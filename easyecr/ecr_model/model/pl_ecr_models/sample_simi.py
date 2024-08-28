import os
import json
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Optional


import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW
from omegaconf import DictConfig
from transformers import AutoConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import LongformerPreTrainedModel, LongformerModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_model import PlEcrModel


class DataSelector(Dataset):
    def __init__(self, data: EcrData) -> None:
        self.dataset_name = data.name
        self.index_type = data.meta["index_type"]

        self.documents = data.documents
        self.mentions = data.mentions
        self.events = data.events
        self.data = self.adapt_data()

    def adapt_data(self):
        # 可能存在doc中不存在mention的情况，这是在ace2005eng数据集中存在的现象。
        doc2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            doc_id = mention.doc_id
            doc2mentions[doc_id].append(mention)

        data = []
        for doc_id, mentions in doc2mentions.items():
            document = self.documents[doc_id]
            events = [
                {
                    "event_id": m.mention_id,
                    "char_start": m.anchor.start,
                    "char_end": m.anchor.end,
                    "trigger": m.anchor.text,
                    "cluster_id": m.meta["event_id"],
                }
                for m in mentions
            ]
            data.append({"id": doc_id, "document": document.text, "events": events, "index_type": self.index_type})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LongformerSelector(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        # encoder & pooler
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.post_init()

    def _cal_circle_loss(self, event_1_reps, event_2_reps, coref_labels, l=20.0):
        norms_1 = (event_1_reps**2).sum(axis=1, keepdims=True) ** 0.5
        event_1_reps = event_1_reps / norms_1
        norms_2 = (event_2_reps**2).sum(axis=1, keepdims=True) ** 0.5
        event_2_reps = event_2_reps / norms_2
        event_cos = torch.sum(event_1_reps * event_2_reps, dim=1) * l
        # calculate the difference between each pair of Cosine values
        event_cos_diff = event_cos[:, None] - event_cos[None, :]
        # find (noncoref, coref) index
        select_idx = coref_labels[:, None] < coref_labels[None, :]
        select_idx = select_idx.float()

        event_cos_diff = event_cos_diff - (1 - select_idx) * 1e12
        event_cos_diff = event_cos_diff.view(-1)
        event_cos_diff = torch.cat((torch.tensor([0.0], device=self.use_device), event_cos_diff), dim=0)
        return torch.logsumexp(event_cos_diff, dim=0)

    def forward(self, batch_inputs, batch_events, batch_event_cluster_ids=None):
        outputs = self.longformer(**batch_inputs)
        self.use_device = self.longformer.device
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list, batch_event_mask = [], [], []
        max_len = 0
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            for events, event_cluster_ids in zip(batch_events, batch_event_cluster_ids):
                event_1_list, event_2_list, coref_labels = [], [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                max_len = max(max_len, len(coref_labels))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        else:
            for events in batch_events:
                event_1_list, event_2_list = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                max_len = max(max_len, len(event_1_list))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_event_mask.append([1] * len(event_1_list))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        # extract events
        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        # calculate loss
        loss, batch_labels = None, None
        if batch_event_cluster_ids is not None and max_len > 0:
            # Only keep active parts of the loss
            active_loss = batch_mask.view(-1) == 1
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
            active_labels = batch_labels.view(-1)[active_loss]
            active_event_1_reps = batch_event_1_reps.view(-1, self.hidden_size)[active_loss]
            active_event_2_reps = batch_event_2_reps.view(-1, self.hidden_size)[active_loss]
            loss = self._cal_circle_loss(active_event_1_reps, active_event_2_reps, active_labels)
        return loss, batch_event_1_reps, batch_event_2_reps, batch_mask, batch_labels


class SampleSimiModule(pl.LightningModule):
    def __init__(self, conf, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        config = AutoConfig.from_pretrained(conf["transformer_model"])
        self.sample_select_model = LongformerSelector(config)

    def forward(self, batch_inputs, batch_events, batch_event_cluster_ids=None, new_events=None):
        loss, batch_event_1_reps, batch_event_2_reps, batch_mask, batch_labels = self.sample_select_model(
            batch_inputs, batch_events, batch_event_cluster_ids
        )
        torch.cuda.empty_cache()
        return loss, batch_event_1_reps, batch_event_2_reps, batch_mask, batch_labels

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        forward_output = self.forward(**batch)
        if forward_output[0] is not None:
            self.log("train_loss", forward_output[0].item())
            return forward_output[0]
        else:
            return None

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        forward_output = self.forward(**batch)
        if forward_output[0] is not None:
            self.log("val_loss", forward_output[0].item())
            return forward_output[0]
        else:
            return None

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    def get_optimizer_and_scheduler(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if self.hparams.optimizer == "adam":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                self.hparams.learning_rate,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                eps=self.hparams.adam_epsilon,
            )
        else:
            raise NotImplementedError

        lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=250, num_training_steps=300000)
        return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        return [optimizer], [lr_scheduler]

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        if all(not v for k, v in batch.items()):
            return [], [], []
        new_events = batch["new_events"]
        outputs = self(**batch)
        _, event_1_reps, event_2_reps, _, _ = outputs
        norms_1 = (event_1_reps**2).sum(axis=-1, keepdims=True) ** 0.5
        event_1_reps = event_1_reps / norms_1
        norms_2 = (event_2_reps**2).sum(axis=-1, keepdims=True) ** 0.5
        event_2_reps = event_2_reps / norms_2

        event_pair_cos = torch.squeeze(torch.sum(event_1_reps * event_2_reps, dim=-1), dim=0).cpu().numpy().tolist()
        if len(new_events) > 1:
            assert len(event_pair_cos) == len(new_events) * (len(new_events) - 1) / 2
        event_id_pairs = []
        for i in range(len(new_events) - 1):
            for j in range(i + 1, len(new_events)):
                event_id_pairs.append(f"{new_events[i]}###{new_events[j]}")
        return new_events, event_id_pairs, event_pair_cos


class SampleSimitor(PlEcrModel):
    def __init__(
        self,
        mode: str,
        model_dir: str,
        model_filename: str,
        trainer_parameters: Dict[str, Any],
        best_model_filename: str = "best.ckpt",
        conf: Optional[DictConfig] = None,
    ):
        super().__init__(mode, model_dir, model_filename, trainer_parameters, best_model_filename, conf)
        seed_everything(42)

    def build_tokenizer(self):
        self.config = AutoConfig.from_pretrained(self.conf["module"]["transformer_model"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf["module"]["transformer_model"])

    def instanciate_module(self):
        module = SampleSimiModule(conf=self.conf["module"])
        return module

    def load_module(self, filepath: str):
        result = SampleSimiModule.load_from_checkpoint(checkpoint_path=filepath)
        return result

    def collate_fn(self, batch_samples):
        tokenizer = self.tokenizer
        batch_sentences, batch_events = [], []
        index_type = None
        for sample in batch_samples:
            batch_sentences.append(sample["document"])
            batch_events.append(sample["events"])
            index_type = sample["index_type"]
        batch_inputs = tokenizer(
            batch_sentences,
            max_length=self.conf["dataset"]["max_seq_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_filtered_events = []
        batch_filtered_event_cluster_id = []
        for sentence, events in zip(batch_sentences, batch_events):
            encoding = tokenizer(sentence, max_length=self.conf["dataset"]["max_seq_length"], truncation=True)
            filtered_events = []
            filtered_event_cluster_id = []
            for e in events:
                if index_type == "char":
                    token_start = encoding.char_to_token(e["char_start"])
                    if not token_start:
                        token_start = encoding.char_to_token(e["char_start"] + 1)
                    token_end = encoding.char_to_token(e["char_end"])
                    if not token_start or not token_end:
                        continue
                elif index_type == "word":
                    token_start = encoding.word_to_tokens(e["char_start"])
                    if not token_start:
                        token_start = encoding.word_to_tokens(e["char_start"] + 1)
                    token_end = encoding.word_to_tokens(e["char_end"])
                    if not token_start or not token_end:
                        continue
                    token_start = token_start[0]
                    token_end = token_end[1]
                filtered_events.append([token_start, token_end])
                filtered_event_cluster_id.append(e["cluster_id"])
            batch_filtered_events.append(filtered_events)
            batch_filtered_event_cluster_id.append(filtered_event_cluster_id)

        batch_data = {
            "batch_inputs": {k: v for k, v in batch_inputs.items()},
            "batch_events": batch_filtered_events,
            "batch_event_cluster_ids": batch_filtered_event_cluster_id,
        }
        return batch_data

    def collate_fn_predict(self, batch_samples):
        tokenizer = self.tokenizer
        batch_sentence, batch_events = [], []
        index_type = None
        for sample in batch_samples:
            batch_sentence.append(sample["document"])
            batch_events.append(sample["events"])
            index_type = sample["index_type"]
        inputs = tokenizer(
            batch_sentence, max_length=self.conf["dataset"]["max_seq_length"], truncation=True, return_tensors="pt"
        )

        filtered_events = []
        new_events = []
        for events in batch_events:
            for event in events:
                event_id = event["event_id"]
                char_start = event["char_start"]
                char_end = event["char_end"]
                if index_type == "char":
                    token_start = inputs.char_to_token(char_start)
                    if not token_start:
                        token_start = inputs.char_to_token(char_start + 1)
                    token_end = inputs.char_to_token(char_end)
                    if not token_start or not token_end:
                        continue
                elif index_type == "word":
                    token_start = inputs.word_to_tokens(char_start)
                    if not token_start:
                        token_start = inputs.word_to_tokens(char_start + 1)
                    token_end = inputs.word_to_tokens(char_end)
                    if not token_start or not token_end:
                        continue
                    token_start = token_start[0]
                    token_end = token_end[1]
                filtered_events.append([token_start, token_end])
                new_events.append(event_id)
        if not new_events:
            return {"batch_inputs": [], "batch_events": [], "new_events": []}
        batch_data = {
            "batch_inputs": {k: v for k, v in inputs.items()},
            "batch_events": [filtered_events],
            "new_events": new_events,
        }
        return batch_data

    def prepare_data(self, data: EcrData, mode: str):
        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]

        dataset = DataSelector(data=data)
        if mode != "predict":
            collate_fn = self.collate_fn
        else:
            collate_fn = self.collate_fn_predict
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=(mode == "train"),
            num_workers=num_workers,
        )
        dataloaders = [dataloader]
        return dataset, dataloaders

    def instantiate_model_checkpoint_callback(self):
        callback = ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            dirpath=self.conf["common"]["model_dir"],
            filename=self.model_filename,
            mode="min",
        )
        return callback

    def predict(self, data: EcrData) -> EcrData:
        trainer = pl.Trainer(
            accelerator="gpu",
            # devices=args.devices
        )
        mentions = data.mentions
        documents = data.documents
        events = data.events

        doc2mentions = defaultdict(list)
        for _, mention in mentions.items():
            doc_id = mention.doc_id
            doc2mentions[doc_id].append(mention)

        event2mentions = defaultdict(list)
        for event in events:
            event_id = event.event_id
            for mention in event.mentions:
                mention_id = mention.mention_id
                event2mentions[event_id].append(mention_id)

        dataset, dataloaders = self.prepare_data(data, mode="predict")
        predictions = trainer.predict(self.module, dataloaders)

        for i, prediction in enumerate(predictions):
            sample = dataset[i]
            doc_id = sample["id"]
            document = documents[doc_id]

            new_events, event_id_pairs, event_pair_cos = prediction
            select_events = [men for m_id, men in mentions.items() if m_id in new_events]
            document.add_tag(name="select_events", value=select_events)
            select_clusters = []
            for event_id, event_mentions in event2mentions.items():
                doc_events = [e_id for e_id in event_mentions if e_id in new_events]
                if len(doc_events) > 0:
                    select_clusters.append({"hopper_id": event_id, "events": doc_events})
            document.add_tag(name="clusters", value=select_clusters)
            document.add_tag(name="event_pairs_id", value=event_id_pairs)
            document.add_tag(name="event_pairs_cos", value=event_pair_cos)
            documents[doc_id] = document

        # # file
        # results = []
        # for i, prediction in enumerate(predictions):
        #     jsonl = {}
        #     sample = dataset[i]
        #     doc_id = sample["id"]
        #     doc = documents[doc_id]
        #     jsonl["doc_id"] = doc_id
        #     jsonl["document"] = doc.text
        #     if data.name in ["kbpmix", "mavenere"]:
        #         jsonl["sentences"] = doc.meta["sentences"]
        #     elif data.name in ["ace2005eng"]:
        #         jsonl["sentences"] = doc.meta["doc_token"]
        #     else:
        #         raise NotImplementedError
        #     jsonl["events"] = doc.meta["select_events"]
        #     jsonl["clusters"] = doc.meta["clusters"]
        #     jsonl["event_pairs_id"] = doc.meta["event_pairs_id"]
        #     jsonl["event_pairs_cos"] = doc.meta["event_pairs_cos"]
        #     results.append(jsonl)
        # if not os.path.exists(self.conf["output"]["dir"]):
        #     os.makedirs(self.conf["output"]["dir"])
        # with open(
        #     os.path.join(self.conf["output"]["dir"], self.conf["output"]["file_name"]), "wt", encoding="utf-8"
        # ) as f:
        #     for example_result in results:
        #         f.write(json.dumps(example_result) + "\n")
        return data
