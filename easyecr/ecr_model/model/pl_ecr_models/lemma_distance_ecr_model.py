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
from easyecr.utils import common_metrics


logger = log_utils.get_logger(__file__)


class LemmaDistanceEcrDataset(Dataset):

    def __init__(
        self,
        ecr_data: EcrData,
        tokenizer,
        coarse_type: str,
        train: bool = False,
        without_singleton: bool = False,
        distance_tag_name: str = "distance_lemma",
        local_context: str = "doc",
        global_context: str = "",
    ):
        self.ecr_data = ecr_data
        self.tokenizer: AutoTokenizer = tokenizer
        self.coarse_type = coarse_type
        self.train = train
        self.include_singleton = not without_singleton
        self.start_id = self.tokenizer.encode("<m>", add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode("</m>", add_special_tokens=False)[0]
        self.space_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
        self.distance_tag_name = distance_tag_name
        self.max_sentence_len = 512
        self.local_context = local_context
        self.global_context = global_context

        self.samples = []
        self.generate_samples()

    def generate_samples(self):
        """

        Returns:

        """
        if self.train:
            sample_generator = NearestNeighborGenerator(threshold=0.5)
        else:
            sample_generator = SimpleGenerator(times=0)
        samples = sample_generator.generate(
            self.ecr_data, self.coarse_type, self.include_singleton, distance_tag_name=self.distance_tag_name
        )
        positive_samples = [e + [1] for e in samples["positive"]]
        negative_samples = [e + [0] for e in samples["negative"]]
        self.samples.extend(positive_samples)
        self.samples.extend(negative_samples)

    def __len__(self):
        result = len(self.samples)
        return result

    def tokenize(self, context1: List[Tuple[str, str, str]], context2: List[Tuple[str, str, str]]):
        """

        Args:
            context1:
            context2:

        Returns:

        """
        global_flag = "<g>"
        doc_start = "<doc-s>"
        doc_end = "</doc-s>"
        texts = [
            f"{global_flag} {doc_start} {context1[0]}",
            context1[1],
            f"{context1[2]} {doc_end} {doc_start} {context2[0]}",
            context2[1],
            f"{context2[2]} {doc_end}",
        ]

        text_1 = f"{global_flag} {doc_start} {context1[0]}{context1[1]}{context1[2]} {doc_end}"
        tokens_1 = self.tokenizer(text_1, add_special_tokens=False)
        m1s_index = tokens_1["input_ids"].index(self.start_id)
        m1e_index = tokens_1["input_ids"].index(self.end_id)
        text_2 = f"{global_flag} {doc_start} {context2[0]}{context2[1]}{context2[2]} {doc_end}"
        tokens_2 = self.tokenizer(text_2, add_special_tokens=False)
        m2s_index = tokens_2["input_ids"].index(self.start_id) + len(tokens_1["input_ids"])
        m2e_index = tokens_2["input_ids"].index(self.end_id) + len(tokens_1["input_ids"])

        input_ids = tokens_1["input_ids"] + tokens_2["input_ids"]
        if len(input_ids) > self.max_sentence_len:
            n = len(input_ids) - self.max_sentence_len
            input_ids = input_ids[: -n - 1] + [input_ids[-1]]

        attention_masks = [0] * len(input_ids)
        trigger1_masks = [0] * len(input_ids)
        trigger2_masks = [0] * len(input_ids)

        for i in range(len(input_ids)):
            if m1s_index <= i <= m1e_index:
                attention_masks[i] = 1
            if m2s_index <= i <= m2e_index:
                attention_masks[i] = 1
            if m1s_index < i < m1e_index:
                trigger1_masks[i] = 1
            if m2s_index < i < m2e_index:
                trigger2_masks[i] = 1

        return input_ids, attention_masks, trigger1_masks, trigger2_masks

    def get_context(self, mention: Mention):
        """

        Args:
            mention:

        Returns:

        """
        temp = self.ecr_data.get_mention_context(
            mention.mention_id, local_context_type=self.local_context, global_context_type=self.global_context
        )
        context = [temp[1] + " <m> ", temp[2], " </m> " + temp[3]]
        return context

    def __getitem__(self, idx: int):
        """

        :param idx:
        :return:
        """
        sample = self.samples[idx]
        mention1: Mention = sample[0]
        mention2: Mention = sample[1]

        context1 = self.get_context(mention1)
        context2 = self.get_context(mention2)

        result = []
        input = self.tokenize(context1, context2)
        result.extend(input)
        reversed_input = self.tokenize(context2, context1)
        result.extend(reversed_input)
        result.append([sample[2]])
        result = [torch.tensor(e) for e in result]

        meta = {"mention_id1": mention1.mention_id, "mention_id2": mention2.mention_id}

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
            field_data = pad_sequence(temp, padding_value=0, batch_first=True)
            fields.append(field_data)
        fields.append(meta)
        return fields


class LemmaDistanceEcrModule(pl.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.config = AutoConfig.from_pretrained(conf["transformer_model"])
        self.encoder = AutoModel.from_config(self.config)
        self.encoder.resize_token_embeddings(50270)
        self.hidden_size = self.config.hidden_size
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.linear.apply(LemmaDistanceEcrModule.init_weights)

        self.bce_loss = torch.nn.BCELoss(reduction="none")

        self.training_step_outputs = []
        self.validation_step_outputs = []

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias)

    def get_representations(self, input_ids, attention_masks, trigger1_masks, trigger2_masks):
        """

        Args:
            input_ids:
            attention_masks:
            trigger1_masks:
            trigger2_masks:

        Returns:

        """
        hiddens = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_masks,
        )
        last_hidden_states = hiddens.last_hidden_state
        context_repr = hiddens.pooler_output

        mention_representation1 = last_hidden_states * torch.unsqueeze(trigger1_masks, dim=2)
        mention_representation1 = torch.sum(mention_representation1, dim=1)

        mention_representation2 = last_hidden_states * torch.unsqueeze(trigger2_masks, dim=2)
        mention_representation2 = torch.sum(mention_representation2, dim=1)

        mention_representation = mention_representation1 * mention_representation2
        repr = torch.cat(
            [context_repr, mention_representation1, mention_representation2, mention_representation], dim=1
        )
        return repr

    def forward(
        self,
        input_ids,
        attention_masks,
        trigger1_masks,
        trigger2_masks,
        reversed_input_ids,
        reversed_attention_masks,
        reversed_trigger1_masks,
        reversed_trigger2_masks,
        labels,
        meta,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        result = {"meta": meta}
        mention_representations = self.get_representations(input_ids, attention_masks, trigger1_masks, trigger2_masks)
        r_mention_representations = self.get_representations(
            reversed_input_ids, reversed_attention_masks, reversed_trigger1_masks, reversed_trigger2_masks
        )

        pred1 = self.linear(mention_representations)
        pred2 = self.linear(r_mention_representations)

        scores_mean = (pred1 + pred2) / 2
        loss = self.bce_loss(scores_mean, labels.float())

        distances = 1 - scores_mean
        distances_square = torch.square(distances)
        result["predictions"] = scores_mean
        result["labels"] = labels
        result["distances"] = distances
        result["distances_square"] = distances_square

        result["instance_loss"] = loss
        loss = torch.mean(loss)
        result["loss"] = loss
        return result

    def training_step(self, batch, batch_idx):
        forward_output = self.forward(*batch[0])
        self.log("train_loss", forward_output["loss"])
        self.training_step_outputs.append(forward_output)
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
        self.log("val_positive_distance", positive_distance)
        self.log("val_negative_distance", negative_distance)
        print()
        print(f"val_positive_distance: {positive_distance}")
        print(f"val_negative_distance: {negative_distance}")

        distances_square = torch.cat([e["distances_square"] for e in self.validation_step_outputs])
        positive_distance_square = torch.sum((labels * distances_square)) / torch.sum(labels)
        negative_distance_square = torch.sum(((1 - labels) * distances_square)) / torch.sum((1 - labels))
        self.log("val_positive_distance_square", positive_distance_square)
        self.log("val_negative_distance_square", negative_distance_square)
        print()
        print(f"val_positive_distance_square: {positive_distance_square}")
        print(f"val_negative_distance_square: {negative_distance_square}")

        predictions = torch.cat([e["predictions"] for e in self.validation_step_outputs]) > 0.5
        predictions = torch.squeeze(predictions)
        labels = torch.squeeze(torch.cat([e["labels"] for e in self.validation_step_outputs]))
        print("val_accuracy:", common_metrics.accuracy(predictions, labels))
        print("val_precision:", common_metrics.precision(predictions, labels))
        print("val_recall:", common_metrics.recall(predictions, labels))
        print("val_f1:", common_metrics.f1_score(predictions, labels))

        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        """

        :return:
        """
        instance_losses = [e["instance_loss"] for e in self.training_step_outputs]
        all_losses = torch.cat(instance_losses)
        val_loss = torch.mean(all_losses)
        self.log("train_loss", val_loss)

        labels = torch.cat([e["labels"] for e in self.training_step_outputs])
        distances = torch.cat([e["distances"] for e in self.training_step_outputs])
        positive_distance = torch.sum((labels * distances)) / torch.sum(labels)
        negative_distance = torch.sum(((1 - labels) * distances)) / torch.sum((1 - labels))
        self.log("train_positive_distance", positive_distance)
        self.log("train_negative_distance", negative_distance)
        print()
        print(f"train_positive_distance: {positive_distance}")
        print(f"train_negative_distance: {negative_distance}")

        distances_square = torch.cat([e["distances_square"] for e in self.training_step_outputs])
        positive_distance_square = torch.sum((labels * distances_square)) / torch.sum(labels)
        negative_distance_square = torch.sum(((1 - labels) * distances_square)) / torch.sum((1 - labels))
        self.log("train_positive_distance_square", positive_distance_square)
        self.log("train_negative_distance_square", negative_distance_square)
        print()
        print(f"train_positive_distance_square: {positive_distance_square}")
        print(f"train_negative_distance_square: {negative_distance_square}")

        predictions = torch.cat([e["predictions"] for e in self.training_step_outputs]) > 0.5
        predictions = torch.squeeze(predictions)
        labels = torch.squeeze(torch.cat([e["labels"] for e in self.training_step_outputs]))
        print("train_accuracy:", common_metrics.accuracy(predictions, labels))
        print("train_precision:", common_metrics.precision(predictions, labels))
        print("train_recall:", common_metrics.recall(predictions, labels))
        print("train_f1:", common_metrics.f1_score(predictions, labels))

        self.training_step_outputs.clear()

    def get_optimizer_and_scheduler(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                [
                    {"params": self.encoder.parameters(), "lr": self.hparams.lr_lm},
                    {"params": self.linear.parameters(), "lr": self.hparams.lr_class},
                ]
            )
        else:
            raise NotImplementedError

        return optimizer

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


class LemmaDistanceEcrModel(PlEcrModel):
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
        # if mode == "train":
        self.tokenizer.add_tokens(["<m>", "</m>"], special_tokens=True)
        self.tokenizer.add_tokens(["<doc-s>", "</doc-s>"], special_tokens=True)
        self.tokenizer.add_tokens(["<g>"], special_tokens=True)
        self.module.encoder.resize_token_embeddings(len(self.tokenizer))

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf["module"]["transformer_model"])

    def instanciate_module(self):
        module = LemmaDistanceEcrModule(self.conf["module"])
        return module

    def load_module(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        result = LemmaDistanceEcrModule.load_from_checkpoint(filepath)
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
        dataset = LemmaDistanceEcrDataset(
            ecr_data=data,
            tokenizer=self.tokenizer,
            coarse_type=coarse_type,
            train=(mode == "train"),
            distance_tag_name=self.conf["common"]["framework1_tag"],
            local_context=self.conf["common"]["local_context"],
            global_context=self.conf["common"]["global_context"],
        )

        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=LemmaDistanceEcrDataset.collate_fn,
            num_workers=num_workers,
            shuffle=(mode == "train"),
            pin_memory=True,
        )
        dataloaders = [dataloader]
        return dataset, dataloaders

    def predict_distances(self, data: EcrData) -> Dict[str, Dict[str, float]]:
        """

        Args:
            data:

        Returns:

        """
        trainer = pl.Trainer(
            accelerator="gpu",
            # devices=args.devices
        )
        test_dataset, dataloaders = self.prepare_data(data, mode=self.mode)
        predictions = trainer.predict(self.module, dataloaders=dataloaders)
        distances = []
        metas = []
        for prediction in predictions:
            distances.extend(prediction["distances"].cpu().numpy().tolist())
            metas.extend(prediction["meta"])
        id_id_distance = {}
        for i, meta in enumerate(metas):
            mention_id1 = meta["mention_id1"]
            mention_id2 = meta["mention_id2"]
            if mention_id1 not in id_id_distance:
                id_id_distance[mention_id1] = {}
            if mention_id2 not in id_id_distance:
                id_id_distance[mention_id2] = {}
            distance = distances[i][0]
            id_id_distance[mention_id1][mention_id2] = distance
            id_id_distance[mention_id2][mention_id1] = distance
        return id_id_distance

    def get_predict_type(self) -> str:
        result = Mention.mention_distance_tag_name
        return result
