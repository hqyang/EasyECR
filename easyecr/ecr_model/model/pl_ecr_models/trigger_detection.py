import os
import copy
import json
from typing import Any
from typing import Dict
from typing import Optional
from collections import defaultdict

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AdamW
from omegaconf import DictConfig
from transformers import AutoConfig
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import DataLoader
from pytorch_lightning.profilers import SimpleProfiler
from transformers import LongformerPreTrainedModel, LongformerModel
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from omegaconf import OmegaConf

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_model import PlEcrModel
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_complex_tagger import EcrComplexTagger


CATEGORIES = [
    "artifact",
    "transferownership",
    "transaction",
    "broadcast",
    "contact",
    "demonstrate",
    "injure",
    "transfermoney",
    "transportartifact",
    "attack",
    "meet",
    "elect",
    "endposition",
    "correspondence",
    "arrestjail",
    "startposition",
    "transportperson",
    "die",
]

MAVENERE = [
    "Control",
    "Achieve",
    "Creating",
    "Self_motion",
    "Motion",
    "Process_start",
    "Process_end",
    "Cause_to_amalgamate",
    "Aiming",
    "Bringing",
    "Participation",
    "Ratification",
    "Conquering",
    "Using",
    "Deciding",
    "Cause_to_make_progress",
    "Dispersal",
    "Coming_to_be",
    "Causation",
    "Damaging",
    "Placing",
    "Manufacturing",
    "Name_conferral",
    "Becoming",
    "Destroying",
    "Cause_change_of_strength",
    "Arriving",
    "Motion_directional",
    "Preserving",
    "Reporting",
    "Killing",
    "Escaping",
    "Recovering",
    "Warning",
    "Removing",
    "Reveal_secret",
    "Patrolling",
    "Carry_goods",
    "Attack",
    "Know",
    "Criminal_investigation",
    "Committing_crime",
    "Judgment_communication",
    "Coming_to_believe",
    "Legal_rulings",
    "Death",
    "Rescuing",
    "Bodily_harm",
    "Statement",
    "Check",
    "Telling",
    "Preventing_or_letting",
    "Communication",
    "Arrest",
    "Breathing",
    "Assistance",
    "Giving",
    "Change",
    "Use_firearm",
    "Body_movement",
    "Wearing",
    "Violence",
    "Come_together",
    "Publishing",
    "Supporting",
    "Getting",
    "Request",
    "Perception_active",
    "Rite",
    "Earnings_and_losses",
    "Recording",
    "Military_operation",
    "Cause_change_of_position_on_a_scale",
    "Risk",
    "Hindering",
    "Hostile_encounter",
    "Reforming_a_system",
    "Change_of_leadership",
    "Labeling",
    "Social_event",
    "Change_sentiment",
    "Extradition",
    "Commitment",
    "Defending",
    "Agree_or_refuse_to_act",
    "Commerce_sell",
    "Expressing_publicly",
    "Temporary_stay",
    "Cure",
    "Commerce_pay",
    "Collaboration",
    "Response",
    "Convincing",
    "Writing",
    "Presence",
    "Catastrophe",
    "Connect",
    "Departing",
    "Influence",
    "Supply",
    "Sending",
    "Hold",
    "Justifying",
    "Rewards_and_punishments",
    "Quarreling",
    "Releasing",
    "Arranging",
    "Action",
    "Employment",
    "Change_event_time",
    "Surrounding",
    "Being_in_operation",
    "Scrutiny",
    "Adducing",
    "GiveUp",
    "Expansion",
    "Confronting_problem",
    "Receiving",
    "Building",
    "Robbery",
    "Competition",
    "Legality",
    "Commerce_buy",
    "Scouring",
    "Choosing",
    "Institutionalization",
    "Cause_to_be_included",
    "Besieging",
    "Prison",
    "Bearing_arms",
    "Vocalizations",
    "Practice",
    "Create_artwork",
    "Becoming_a_member",
    "Protest",
    "Theft",
    "Education_teaching",
    "Forming_relationships",
    "Traveling",
    "Award",
    "Revenge",
    "Emptying",
    "Sign_agreement",
    "Suspicion",
    "Submitting_documents",
    "Hiding_objects",
    "Testing",
    "Resolve_problem",
    "Terrorism",
    "GetReady",
    "Incident",
    "Surrendering",
    "Having_or_lacking_access",
    "Imposing_obligation",
    "Ingestion",
    "Kidnapping",
    "Openness",
    "Exchange",
    "Emergency",
    "Cost",
    "Research",
    "Change_tool",
    "Containing",
    "Filling",
    "Renting",
    "Expend_resource",
    "Lighting",
    "Limiting",
]

id2label = None
label2id = None


class DataTrigger(Dataset):
    def __init__(self, data: EcrData):
        self.dataset_name = data.name
        self.index_type = data.meta["index_type"]
        global id2label
        global label2id
        if self.dataset_name in ["ace2005eng", "kbpmix"]:
            id2label = {0: "O"}
            for c in CATEGORIES:
                id2label[len(id2label)] = f"B-{c}"
                id2label[len(id2label)] = f"I-{c}"
            label2id = {v: k for k, v in id2label.items()}
        elif self.dataset_name in ["mavenere"]:
            id2label = {0: "O"}
            for c in MAVENERE:
                id2label[len(id2label)] = f"B-{c}"
                id2label[len(id2label)] = f"I-{c}"
            label2id = {v: k for k, v in id2label.items()}
            print("id2label:", len(id2label), "label2id:", len(label2id))
        else:
            raise NotImplementedError
        self.documents = data.documents
        self.mentions = data.mentions
        self.events = data.events
        self.data = self.adapt_data()

    def adapt_data(self):
        doc2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            doc_id = mention.doc_id
            doc2mentions[doc_id].append(mention)
        data = []

        for doc_id, mentions in doc2mentions.items():
            document = self.documents[doc_id]
            text = document.text
            if self.dataset_name in ["ace2005eng", "kbpmix"]:
                tags = [
                    (
                        mention.anchor.start,
                        mention.anchor.end,
                        mention.anchor.text,
                        mention.meta["subtype"],
                    )
                    for mention in mentions
                    if mention.meta["subtype"] in CATEGORIES
                ]
            elif self.dataset_name in ["mavenere"]:
                tags = [
                    (
                        mention.anchor.start,
                        mention.anchor.end,
                        mention.anchor.text,
                        mention.meta["event_type"],
                    )
                    for mention in mentions
                    if mention.meta["event_type"] in MAVENERE
                ]
            else:
                raise NotImplementedError
            data.append({"id": doc_id, "document": text, "tags": tags, "index_type": self.index_type})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TriggerDetectionModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(conf)
        config = AutoConfig.from_pretrained(conf["transformer_model"])

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = conf["num_labels"]
        self.classifier = nn.Linear(config.hidden_size, conf["num_labels"])
        self.loss_fct = CrossEntropyLoss()
        self.longformer.post_init()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        torch.cuda.empty_cache()
        return loss, logits

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        loss, logits = self.forward(**batch)

        if loss is not None:
            self.log("train_loss", loss.item())
            return loss
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
        offsets = batch.pop("offset_mapping").squeeze(0)
        result = self(**batch)
        logits = result[1]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        predictions = logits.argmax(dim=-1)[0].cpu().numpy().tolist()
        pred_label = []

        idx = 1
        while idx < len(predictions) - 1:
            pred = predictions[idx]
            label = id2label[pred]
            if label != "O":
                label = label[2:]  # Remove the B- or I-
                start, end = offsets[idx]
                all_scores = [probabilities[idx][pred]]
                # Grab all the tokens labeled with I-label
                while idx + 1 < len(predictions) - 1 and id2label[predictions[idx + 1]] == f"I-{label}":
                    all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                    _, end = offsets[idx + 1]
                    idx += 1

                score = np.mean(all_scores).item()
                start, end = start.item(), end.item()
                pred_label.append({"start": start, "end": end, "subtype": label, "score": score})
            idx += 1
        return pred_label


class TriggerDetection(PlEcrModel):
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
        module = TriggerDetectionModule(conf=self.conf["module"])
        return module

    def load_module(self, filepath: str):
        result = TriggerDetectionModule.load_from_checkpoint(filepath)
        return result

    def collate_fn(self, batch_samples):
        batch_sentence, batch_tags = [], []
        index_type = None
        for sample in batch_samples:
            batch_sentence.append(sample["document"])
            batch_tags.append(sample["tags"])
            index_type = sample["index_type"]
        batch_inputs = self.tokenizer(
            batch_sentence,
            max_length=self.conf["dataset"]["max_seq_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_label = np.zeros(batch_inputs["input_ids"].shape, dtype=int)
        for s_idx, sentence in enumerate(batch_sentence):
            encoding = self.tokenizer(sentence, max_length=self.conf["dataset"]["max_seq_length"], truncation=True)
            for char_start, char_end, _, tag in batch_tags[s_idx]:
                if index_type == "char":
                    token_start = encoding.char_to_token(char_start)
                    token_end = encoding.char_to_token(char_end)
                    if not token_start or not token_end:
                        continue
                elif index_type == "word":
                    token_start = encoding.word_to_tokens(char_start)
                    token_end = encoding.word_to_tokens(char_end)
                    if not token_start or not token_end:
                        continue
                    token_start = token_start[0]
                    token_end = token_end[1]
                batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
                batch_label[s_idx][token_start + 1 : token_end + 1] = label2id[f"I-{tag}"]
        batch_inputs["labels"] = torch.tensor(batch_label)
        return batch_inputs

    def collate_fn_predict(self, batch_samples):
        batch_sentence = []
        for sample in batch_samples:
            batch_sentence.append(sample["document"])
        batch_inputs = self.tokenizer(
            batch_sentence,
            max_length=self.conf["dataset"]["max_seq_length"],
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        return batch_inputs

    def prepare_data(self, data: EcrData, mode: str):
        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]

        dataset = DataTrigger(data=data)
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
        test_dataset, test_dataloaders = self.prepare_data(data, mode="predict")
        predictions = trainer.predict(self.module, test_dataloaders)
        results = []
        for i, prediction in enumerate(predictions):
            document = test_dataset[i]["document"]
            doc_id = test_dataset[i]["id"]
            tag = test_dataset[i]["tags"]
            new_prediction = []
            for pred in prediction:
                start, end = pred["start"], pred["end"]
                word = document[start:end]
                pred.pop("end")
                pred["trigger"] = word
                new_prediction.append(pred)
            results.append({"doc_id": doc_id, "document": document, "pred_label": new_prediction, "true_label": tag})
        if not os.path.exists(self.conf["output"]["dir"]):
            os.makedirs(self.conf["output"]["dir"])
        with open(
            os.path.join(self.conf["output"]["dir"], self.conf["output"]["file_name"]), "wt", encoding="utf-8"
        ) as f:
            for example_result in results:
                f.write(json.dumps(example_result) + "\n")
        # add_tag
        for res in results:
            doc_id = res["doc_id"]
            document = data.documents[doc_id]
            document.add_tag(name="trigger_true_label", value=res["true_label"])
            document.add_tag(name="trigger_pred_label", value=res["pred_label"])
        return data
