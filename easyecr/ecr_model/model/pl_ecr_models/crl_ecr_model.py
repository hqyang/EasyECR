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
from typing import Any
from typing import Optional

from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
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
from easyecr.ecr_model.model.event_coreference_pl_module import EventCoreferenceModule
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_model import PlEcrModel
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.utils import text_encoder
from easyecr.utils import log_utils
from easyecr.ecr_model.sample_generator.sample_generator import HardGenerator
from easyecr.ecr_model.sample_generator.sample_generator import SimpleGenerator


logger = log_utils.get_logger(__file__)


class EventCoreferenceDataset(Dataset):
    def __init__(
        self, ecr_data: EcrData, tokenizer, coarse_type: str, train: bool = False, without_singleton: bool = False
    ):
        self.ecr_data = ecr_data
        self.tokenizer: AutoTokenizer = tokenizer
        self.coarse_type = coarse_type
        self.train = train
        self.include_singleton = not without_singleton
        self.distance_tag_name = "embedding_distance"
        self.max_mention_token_num = 128

        self.samples = []
        self.generate_samples()

    def generate_samples(self):
        """

        Returns:

        """
        if self.train:
            sample_generator = HardGenerator()
            samples = sample_generator.generate(
                self.ecr_data, self.coarse_type, self.include_singleton, distance_tag_name=self.distance_tag_name
            )
            positive_samples = [e + [1] for e in samples["positive"]]
            negative_samples = [e + [0] for e in samples["negative"]]
            self.samples.extend(positive_samples)
            self.samples.extend(negative_samples)
        else:
            self.samples.extend([[e] for e in list(self.ecr_data.mentions.values())])

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

    def tokenize(self, texts: List[str]):
        """

        :param texts:
        :return:
        """
        left_tokens = self.tokenizer(texts[0])
        tokenized_left = [
            left_tokens["input_ids"][:-1],
            left_tokens["attention_mask"][:-1],
            # tokens['token_type_ids'],
        ]
        tokens = self.tokenizer(texts[1])
        tokenized_text = [
            tokens["input_ids"][1:-1],
            tokens["attention_mask"][1:-1],
            # tokens['token_type_ids'],
        ]
        right_tokens = self.tokenizer(texts[2])
        tokenized_right = [
            right_tokens["input_ids"][1:],
            right_tokens["attention_mask"][1:],
            # tokens['token_type_ids'],
        ]
        result = [
            tokenized_left[0] + tokenized_text[0] + tokenized_right[0],
            tokenized_left[1] + tokenized_text[1] + tokenized_right[1],
            [0.0] * len(tokenized_left[1]) + [1.0] * len(tokenized_text[1]) + [0.0] * len(tokenized_right[1]),
        ]
        result[0] = result[0][-512:]
        result[1] = result[1][-512:]
        result[2] = result[2][-512:]
        left, right = self.find_left_right_index(result[2])
        result.extend([[left], [right]])
        return result

    def get_context(self, mention: Mention):
        """

        Args:
            mention:

        Returns:

        """
        temp = self.ecr_data.get_mention_context(
            mention.mention_id,
            local_context_type="sentence",
            global_context_type="first_sentences",
            first_sentence_num=2,
        )
        context = [temp[1] + " [E] ", temp[2], " [/E] " + temp[3]]
        context_words = " ".split("".join(context))

        global_context = temp[0]
        if global_context:
            global_words = " ".split(global_context)
        else:
            global_words = []

        minus = self.max_mention_token_num - len(context_words)
        if minus > 0:
            context[0] = " ".join(global_words[:minus]) + " [SEP] " + context[0]
        return context

    def __getitem__(self, idx: int):
        """

        :param idx:
        :return:
        """
        sample = self.samples[idx]
        mention1: Mention = sample[0]
        context1 = self.get_context(mention1)
        result = []
        first_tokenized_result = self.tokenize(context1)
        result.extend(first_tokenized_result)
        if self.train:
            mention2: Mention = sample[1]
            context2 = self.get_context(mention2)
            result.extend(self.tokenize(context2))
            result.append([sample[2]])
        else:
            result.extend(first_tokenized_result)
            result.append([0])
        meta = {"mention_id1": mention1.mention_id}
        result = [torch.tensor(e) for e in result]
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
            else:
                field_data = None
            fields.append(field_data)
        fields.append(meta)
        return fields


class EventCoreferenceModule(pl.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.config = AutoConfig.from_pretrained(conf["transformer_model"])
        self.encoder = AutoModel.from_config(self.config)
        self.margin = 0.4
        self.validation_step_outputs = []
        self.hidden_size = self.config.hidden_size
        self.single_repr_dim = 1024
        self.mention_mlp = nn.Linear(self.hidden_size, self.single_repr_dim)
        self.context_mlp = nn.Linear(self.hidden_size, self.single_repr_dim)
        self.pdist = nn.PairwiseDistance(p=2)

    def get_mention_representions(self, mentions, attention_masks, mention_masks):
        """

        :param mentions:
        :param attention_masks:
        :param mention_masks:
        :return:
        """
        hiddens = self.encoder(
            input_ids=mentions,
            attention_mask=attention_masks,
            # token_type_ids=token_type_ids
        )
        last_hidden_states = hiddens.last_hidden_state
        context_repr = last_hidden_states[:, 0, :]
        context_repr = self.context_mlp(context_repr)

        mention_representation = last_hidden_states * torch.unsqueeze(mention_masks, dim=2)
        mention_representation = torch.sum(mention_representation, dim=1)
        mention_representation = self.mention_mlp(mention_representation)
        repr = torch.cat([context_repr, mention_representation], dim=1)
        return repr

    def forward(
        self,
        mentions1,
        attention_masks1,
        mention_masks1,
        mention_left1,
        mention_right1,
        # token_type_ids1,
        mentions2,
        attention_masks2,
        mention_masks2,
        mention_left2,
        mention_right2,
        # token_type_ids2,
        labels,
        meta,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        result = {"meta": meta}
        mention_representations1 = self.get_mention_representions(mentions1, attention_masks1, mention_masks1)
        result["mention_representations1"] = mention_representations1
        if mentions2 is not None:
            mention_representations2 = self.get_mention_representions(mentions2, attention_masks2, mention_masks2)
            distances = 1 - torch.cosine_similarity(
                mention_representations1,
                mention_representations2,
            )
            # distances = self.pdist(mention_representations1, mention_representations2)
            distances = torch.unsqueeze(distances, dim=1)
            distances_square = torch.square(distances)
            # print()
            # print(f'label: {labels.detach().cpu().numpy()}')
            # print(f'distances: {distances.detach().cpu().numpy()}')
            # print(f'distances_square: {distances_square.detach().cpu().numpy()}')
            result["labels"] = labels
            result["distances"] = distances
            result["distances_square"] = distances_square
            one_minus_labels = 1 - labels
            loss = labels * distances_square + one_minus_labels * torch.square(
                torch.clamp(self.margin - distances, min=0)
            )
            # loss = labels * distances_square + one_minus_labels * (1.0 - distances_square)
            result["instance_loss"] = loss
            loss = torch.mean(loss)
            result["loss"] = loss
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

        labels = torch.cat([e["labels"] for e in self.validation_step_outputs])
        distances = torch.cat([e["distances"] for e in self.validation_step_outputs])
        positive_distance = torch.sum((labels * distances)) / torch.sum(labels)
        negative_distance = torch.sum(((1 - labels) * distances)) / torch.sum((1 - labels))
        self.log("positive_distance", positive_distance)
        self.log("negative_distance", negative_distance)
        print()
        print(f"positive_distance: {positive_distance}")
        print(f"negative_distance: {negative_distance}")

        distances_square = torch.cat([e["distances_square"] for e in self.validation_step_outputs])
        positive_distance_square = torch.sum((labels * distances_square)) / torch.sum(labels)
        negative_distance_square = torch.sum(((1 - labels) * distances_square)) / torch.sum((1 - labels))
        self.log("positive_distance_square", positive_distance_square)
        self.log("negative_distance_square", negative_distance_square)
        print()
        print(f"positive_distance_square: {positive_distance_square}")
        print(f"negative_distance_square: {negative_distance_square}")

        self.validation_step_outputs.clear()

    def get_optimizer_and_scheduler(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), self.hparams.learning_rate)
        else:
            raise NotImplementedError
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch // 10))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 调度器更新的间隔（可以是'epoch'或'step'）
                "frequency": 1,  # 调度器更新的频率
                "reduce_on_plateau": False,  # 是否使用 ReduceLROnPlateau 调度器
            },
        }

    def configure_optimizers(self):
        optimizer = self.get_optimizer_and_scheduler()
        return optimizer

    def predict_step(self, batch, batch_idx: int):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        result = self(*batch)
        return result


class ContrastiveRepresentationLearningEcrModel(PlEcrModel):
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

    def instanciate_module(self):
        module = EventCoreferenceModule(self.conf["module"])
        return module

    def load_module(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        result = EventCoreferenceModule.load_from_checkpoint(filepath)
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
        dataset = EventCoreferenceDataset(data, self.tokenizer, coarse_type, train=(mode == "train"))

        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=EventCoreferenceDataset.collate_fn,
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
        test_dataset, test_dataloaders = self.prepare_data(data, mode=self.mode)
        predictions = trainer.predict(self.module, dataloaders=test_dataloaders)
        result = {}
        for e in predictions:
            meta = e["meta"]
            mention_representations1 = e["mention_representations1"]
            for i, mention_meta in enumerate(meta):
                result[mention_meta["mention_id1"]] = mention_representations1[i].cpu().numpy()
        return result

    def get_predict_type(self) -> str:
        result = Mention.mention_repr_tag_name
        return result
