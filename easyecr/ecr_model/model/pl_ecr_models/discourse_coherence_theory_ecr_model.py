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


logger = log_utils.get_logger(__file__)


class DiscourseCoherenceTheoryEcrDataset(Dataset):

    def __init__(
        self,
        ecr_data: EcrData,
        tokenizer,
        coarse_type: str,
        train: bool = False,
        without_singleton: bool = False,
        distance_tag_name: str = "embedding_distance",
    ):
        self.ecr_data = ecr_data
        self.tokenizer: AutoTokenizer = tokenizer
        self.coarse_type = coarse_type
        self.train = train
        self.include_singleton = not without_singleton
        self.distance_tag_name = distance_tag_name
        self.max_sentence_len = 512

        self.samples = []
        self.generate_samples()

    def generate_samples(self):
        """

        Returns:

        """
        if self.train:
            sample_generator = NearestNeighborGenerator(top_k=10)
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

    def tokenize(self, context1: List[Tuple[str, str, str]], context2: List[Tuple[str, str, str]]):
        """

        Args:
            context1:
            context2:

        Returns:

        """
        texts = [context1[0], context1[1], f"{context1[2]} </s> {context2[0]}", context2[1], context2[2]]

        input_ids = []
        attention_masks = []
        trigger1_masks = []
        trigger2_masks = []
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

            trigger1_mask = 1 if i == 1 else 0
            trigger2_mask = 1 if i == 3 else 0
            trigger1_masks.extend([trigger1_mask] * (input_id_end - input_id_start))
            trigger2_masks.extend([trigger2_mask] * (input_id_end - input_id_start))

        cur_input_len = len(input_ids)
        if cur_input_len > self.max_sentence_len:
            more = cur_input_len - self.max_sentence_len
            # print("cur_input_len", cur_input_len, "more", more)
            last_one_index = min(trigger1_masks[::-1].index(1), trigger2_masks[::-1].index(1))
            first_one_index = min(trigger1_masks.index(1), trigger2_masks.index(1))
            if last_one_index + 1 > more:
                input_ids = input_ids[:-more]
                attention_masks = attention_masks[:-more]
                trigger1_masks = trigger1_masks[:-more]
                trigger2_masks = trigger2_masks[:-more]
            elif first_one_index + 1 > more:
                input_ids = input_ids[more:]
                attention_masks = attention_masks[more:]
                trigger1_masks = trigger1_masks[more:]
                trigger2_masks = trigger2_masks[more:]
            else:
                for i in range(cur_input_len):
                    if trigger1_masks[i] != 1 and trigger2_masks[i] != 1:
                        del input_ids[i]
                        del attention_masks[i]
                        del trigger1_masks[i]
                        del trigger2_masks[i]
                        more -= 1
                    if more == 0:
                        break

        return input_ids, attention_masks, trigger1_masks, trigger2_masks

    def get_context(self, mention: Mention):
        """

        Args:
            mention:

        Returns:

        """
        temp = self.ecr_data.get_mention_context(mention.mention_id, local_context_type="doc")
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


class DiscourseCoherenceTheoryEcrModule(pl.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.config = AutoConfig.from_pretrained(conf["transformer_model"])
        self.encoder = AutoModel.from_config(self.config)
        self.word_embedding_dim = self.encoder.embeddings.word_embeddings.embedding_dim
        self.mention_dim = self.word_embedding_dim * 2
        self.hidden_size = self.config.hidden_size
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.mention_dim),
            nn.ReLU(),
            nn.Linear(self.mention_dim, self.mention_dim),
            nn.ReLU(),
            nn.Linear(self.mention_dim, 1),
            nn.Sigmoid(),
        )
        # self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")  # integrate Sigmoid()
        self.bce_loss = torch.nn.BCELoss(reduction="none")

        self.validation_step_outputs = []

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
        context_repr = last_hidden_states[:, 0, :]

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
        self, input_ids, attention_masks, trigger1_masks, trigger2_masks, labels, meta, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        result = {"meta": meta}
        mention_representations = self.get_representations(input_ids, attention_masks, trigger1_masks, trigger2_masks)

        pred1 = self.linear(mention_representations)

        loss = self.bce_loss(pred1, labels.float())

        distances = 1 - pred1
        distances_square = torch.square(distances)
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


class DiscourseCoherenceTheoryEcrModel(PlEcrModel):
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
        module = DiscourseCoherenceTheoryEcrModule(self.conf["module"])
        # module.encoder.resize_token_embeddings(len(self.tokenizer))
        return module

    def load_module(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        result = DiscourseCoherenceTheoryEcrModule.load_from_checkpoint(filepath)
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
        dataset = DiscourseCoherenceTheoryEcrDataset(
            data,
            self.tokenizer,
            coarse_type,
            train=(mode == "train"),
            distance_tag_name=self.conf["common"]["framework1_tag"],
        )

        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=DiscourseCoherenceTheoryEcrDataset.collate_fn,
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
        test_dataset, dataloaders = self.prepare_data(data, mode="predict")
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
