#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/13 10:09
"""
import random
import sys
from collections import defaultdict
import copy
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from typing import Any

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback

from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoConfig
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup
import torch
from torch import nn
import pytorch_lightning as pl

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_model import PlEcrModel
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.utils import text_encoder
from easyecr.utils import log_utils
from easyecr.ecr_model.sample_generator.sample_generator import NearestNeighborGenerator
from easyecr.ecr_model.sample_generator.sample_generator import SimpleGenerator
from easyecr.ecr_model.cluster.cluster_model import EcrAgglomerativeClustering

logger = log_utils.get_logger(__file__)


class ClusterEcrDataset(Dataset):
    def __init__(
        self, ecr_data: EcrData, tokenizer, coarse_type: str, train: bool = False, without_singleton: bool = False
    ):
        self.ecr_data = ecr_data
        self.tokenizer: AutoTokenizer = tokenizer
        self.coarse_type = coarse_type
        self.train = train
        self.include_singleton = not without_singleton
        self.max_sentence_len = 512

        self.samples = []
        self.generate_samples()

    def generate_samples(self):
        """

        Returns:

        """
        self.samples.extend(list(self.ecr_data.mentions.values()))

    def __len__(self):
        result = len(self.samples)
        return result

    def find_left_right_index(self, masks: List[float]):
        """

        :param masks:
        :return:
        """
        if 1.0 not in masks:
            print()
        left = masks.index(1.0)
        right = left
        for i in range(left + 1, len(masks)):
            if masks[i] == 1.0:
                right = i
            else:
                break
        return left, right

    def tokenize(self, texts: List[Tuple[str, str, str]]):
        """

        Args:
            texts:

        Returns:

        """
        input_ids = []
        attention_masks = []
        trigger1_masks = []
        for i, text in enumerate(texts):
            tokens = self.tokenizer(text)
            input_id_start = 0
            input_id_end = len(tokens["input_ids"])
            if i == 0:
                input_id_end -= 1
            elif i == len(texts) - 1:
                input_id_start += 1
            else:
                input_id_end -= 1
                input_id_start += 1
            input_ids.extend(tokens["input_ids"][input_id_start:input_id_end])
            attention_masks.extend(tokens["attention_mask"][input_id_start:input_id_end])

            # trigger1_mask = 1 if i == 1 else 0
            temp_trigger_mask = [0] * (input_id_end - input_id_start)
            if i == 1:
                temp_trigger_mask[0] = 1
                temp_trigger_mask[-1] = 1
            trigger1_masks.extend(temp_trigger_mask)

            cur_input_len = len(input_ids)
            if cur_input_len > self.max_sentence_len:
                more = cur_input_len - self.max_sentence_len
                for i in range(cur_input_len):
                    if trigger1_masks[i] != 1:
                        del input_ids[i]
                        del attention_masks[i]
                        del trigger1_masks[i]
                        more -= 1
                    if more == 0:
                        break

        return input_ids, attention_masks, trigger1_masks

    def get_context(self, mention: Mention):
        """

        Args:
            mention:

        Returns:

        """
        temp = self.ecr_data.get_mention_context(
            mention_id=mention.mention_id, local_context_type="doc", max_local_context_len=250
        )
        context = [temp[1], temp[2], temp[3]]
        return context

    def __getitem__(self, idx: int):
        """

        :param idx:
        :return:
        """
        mention = self.samples[idx]

        context = self.get_context(mention)
        mention_repr = self.tokenize(context)
        if mention.contain("label_cluster_label_updator"):
            label = mention.get_tag("label_cluster_label_updator")
            label_repr, cluster_id = label
        else:
            label_repr = None
            cluster_id = None

        result = []
        result.extend(mention_repr)
        result.extend([label_repr])

        result = [torch.tensor(e) if e is not None else e for e in result]

        meta = {"mention_id": mention.mention_id, "cluster_id": cluster_id}

        return result, meta

    @staticmethod
    def collate_fn(data_and_meta):
        """

        :param data_and_meta:
        :return:
        """
        data = [e[0] for e in data_and_meta]
        meta = [e[1] for e in data_and_meta]
        field_num = len(data[0])
        fields = []
        for i in range(field_num):
            temp = [e[i] for e in data]
            if temp[0] is not None:
                field_data = pad_sequence(temp, padding_value=0, batch_first=True)
                fields.append(field_data)
            else:
                fields.append(None)
        fields.append(meta)
        return fields


class ClusterEcrModule(pl.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.config = AutoConfig.from_pretrained(conf["transformer_model"])
        self.encoder = AutoModel.from_config(self.config)
        self.hidden_size = self.config.hidden_size
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.validation_step_outputs = []

    def get_representations(self, input_ids, attention_masks, trigger1_masks):
        """

        Args:
            input_ids:
            attention_masks:
            trigger1_masks:

        Returns:

        """
        hiddens = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_masks,
        )
        last_hidden_states = hiddens.last_hidden_state
        context_repr = last_hidden_states[:, 0, :]

        batch_size = last_hidden_states.shape[0]
        # mention_representation1 = last_hidden_states * torch.unsqueeze(trigger1_masks, dim=2)
        # mention_representation1 = torch.sum(mention_representation1, dim=1)
        # repr = torch.cat(mention_representation1)
        trigger1_index = torch.nonzero(trigger1_masks == 1, as_tuple=False).tolist()
        res = {}
        for idx in trigger1_index:
            res[idx[0]] = res.get(idx[0], 0) + 1

        lis = []
        for idx in trigger1_index:
            if res[idx[0]] == 1:
                lis.append(idx)
                lis.append(idx)
            else:
                lis.append(idx)

        aug_trigger1_index = torch.tensor(lis)

        selected_values = last_hidden_states[aug_trigger1_index[:, 0], aug_trigger1_index[:, 1], :]
        concatenated_tensor = selected_values.view(batch_size, 2, context_repr.shape[1])
        repr = torch.cat([concatenated_tensor[:, i, :] for i in range(concatenated_tensor.size(1))], dim=1)
        return repr

    def compute_inner_product(self, tensor1, tensor2):
        """

        Args:
            tensor1:
            tensor2:

        Returns:

        """
        batch_size = tensor1.shape[0]
        tensor_dim = tensor1.shape[1]
        result = torch.bmm(tensor1.view(batch_size, 1, tensor_dim), tensor2.view(batch_size, tensor_dim, 1))
        result = torch.squeeze(result, dim=-1)
        return result

    def forward(
        self, input_ids, attention_masks, trigger1_masks, labels, metas, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        result = {"meta": metas}
        mention_representations = self.get_representations(input_ids, attention_masks, trigger1_masks)

        # mention_representations = self.linear(mention_representations)
        result["mention_representations"] = mention_representations

        if labels is not None:
            positive_losses = self.compute_inner_product(mention_representations, labels) * -1
            negative_losses = []
            for i, meta_i in enumerate(metas):
                cluster_id_i = meta_i["cluster_id"]
                negative_tensors = []
                for j, meta_j in enumerate(metas):
                    if i == j:
                        continue
                    cluster_id_j = meta_i["cluster_id"]
                    if cluster_id_j == cluster_id_i:
                        continue
                    negative_tensors.append(labels[j].unsqueeze(dim=0))
                if len(negative_tensors) > 0:
                    negative_tensors = torch.cat(negative_tensors, dim=0)
                    anchors = torch.cat(
                        [torch.unsqueeze(mention_representations[i], dim=0) for _ in range(len(negative_tensors))],
                        dim=0,
                    )
                    negative_loss = self.compute_inner_product(anchors, negative_tensors)
                    negative_loss = torch.log(torch.sum(torch.exp(negative_loss)))
                    negative_losses.append(negative_loss)
                else:
                    negative_losses.append(torch.tensor([[0.0]]).to(self.device))
            negative_losses = torch.cat(negative_losses, dim=0)
            loss = positive_losses + negative_losses
            result["instance_loss"] = loss

            loss = torch.mean(loss)
            result["loss"] = loss
        else:
            result["loss"] = 0
        return result

    def training_step(self, batch, batch_idx):
        forward_output = self.forward(*batch[0])
        self.log("train_loss", forward_output["loss"])
        return forward_output

    def validation_step(self, batch, batch_idx: int, *args, **kwargs):
        forward_output = self.forward(*batch)
        self.log("val_loss", forward_output["loss"])
        self.validation_step_outputs.append(forward_output)
        return forward_output

    def on_validation_epoch_end(self):
        """

        :return:
        """
        instance_losses = [e["instance_loss"] for e in self.validation_step_outputs]
        all_losses = torch.cat(instance_losses)
        val_loss = torch.mean(all_losses)
        self.log("val_loss", val_loss)
        print(f"val_loss: {val_loss}")
        self.validation_step_outputs.clear()

    def get_optimizer_and_scheduler(self):
        no_decay = self.hparams.no_decay_params

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, self.hparams.learning_rate)
        else:
            raise NotImplementedError

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )

        return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        if lr_scheduler is None:
            return optimizer
        return [optimizer], [{"interval": "step", "scheduler": lr_scheduler}]

    def predict_step(self, batch, batch_idx: int):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        result = self(*batch)
        return result


class ClusterLabelUpdator(Callback):
    """
    2021-EMNLP-Focus on what matters: Applying Discourse Coherence Theory to Cross Document Coreference
    """

    def __init__(self, ecr_model: "ClusterEcrModel", train_data: EcrData, dev_data: EcrData, predict_topic: str):
        """

        Args:
            ecr_model:
            train_data:
            dev_data:
            predict_topic:
        """
        self.ecr_model = ecr_model
        self.train_data = train_data
        self.dev_data = dev_data
        self.predict_topic = predict_topic
        self.cluster_model = EcrAgglomerativeClustering(best_distance=0.5)

    def update_label(self):
        """

        Returns:

        """
        for data in [self.train_data, self.dev_data]:
            data_tag = self.ecr_model.get_predict_type() + "_cluster_label_updator"
            data = self.ecr_model.predict(data, data_tag)
            output_tag = "event_id_predict_cluster_label_updator"
            data = self.cluster_model.predict(data, output_tag, data_tag, self.predict_topic)
            topic_mentions = data.group_mention_by_tag(output_tag)
            for topic, mentions in topic_mentions.items():
                reprs = []
                for mention in mentions:
                    repr = mention.get_tag(data_tag)
                    reprs.append(np.expand_dims(repr, axis=0))
                reprs = np.concatenate(reprs, axis=0)
                mean_repr = np.mean(reprs, axis=0)
                for mention in mentions:
                    mention.add_tag("label_cluster_label_updator", (mean_repr, mention.get_tag(output_tag)))

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """

        Args:
            trainer:
            pl_module:

        Returns:

        """
        self.update_label()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.update_label()


class ClusterEcrModel(PlEcrModel):
    """
    Contrastive Representation Learning for Cross-Document Coreference Resolution of Events and Entities
    """

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

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf["module"]["transformer_model"])
        # self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)

    def instanciate_module(self):
        module = ClusterEcrModule(self.conf["module"])
        # module.encoder.resize_token_embeddings(len(self.tokenizer))
        return module

    def load_module(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        result = ClusterEcrModule.load_from_checkpoint(filepath)
        return result

    def prepare_data(self, data: EcrData, mode: str):
        """

        Args:
            data:
            mode:

        Returns:

        """
        if data is None:
            return None, None
        coarse_type = self.conf["common"][f"{mode}_topic"]
        dataset = ClusterEcrDataset(data, self.tokenizer, coarse_type, train=(mode == "train"))

        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=ClusterEcrDataset.collate_fn,
            num_workers=num_workers,
            shuffle=(mode == "train"),
        )
        dataloaders = [dataloader]
        return dataset, dataloaders

    def predict_reprs(self, data: EcrData) -> Dict[str, List[float]]:
        """

        Args:
            data:

        Returns:

        """
        trainer = pl.Trainer(
            accelerator="gpu",
            # devices=args.devices
        )

        test_dataset, test_dataloaders = self.prepare_data(data, mode="predict")

        model_copy = self.instanciate_module()
        model_copy.load_state_dict(self.module.state_dict())
        model_copy.to(self.module.device)

        predictions = trainer.predict(model_copy, dataloaders=test_dataloaders)
        del model_copy
        torch.cuda.empty_cache()

        result = {}
        for e in predictions:
            meta = e["meta"]
            mention_representations = e["mention_representations"]
            for i, mention_meta in enumerate(meta):
                result[mention_meta["mention_id"]] = mention_representations[i].cpu().numpy()
        return result

    def get_predict_type(self) -> str:
        result = Mention.mention_repr_tag_name
        return result

    def add_callbacks(self, train_data: EcrData, dev_data: EcrData) -> List[Callback]:
        """

        Args:
            train_data:
            dev_data:

        Returns:

        """
        result = []
        label_updator = ClusterLabelUpdator(self, train_data, dev_data, self.conf["common"]["predict_topic"])
        result.append(label_updator)
        return result
