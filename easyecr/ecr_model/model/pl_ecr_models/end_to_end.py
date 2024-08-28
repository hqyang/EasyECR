#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/12 15:08
"""
import os
from typing import Dict
from typing import List
from typing import Any
from typing import Optional
import copy
import shutil
from enum import Enum
from collections import defaultdict
import random

from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
import numpy as np
from torch import nn
from tqdm import tqdm

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.data.end_to_end_event_coreference_dataset import EndToEndEventCoreferenceDataset
from easyecr.ecr_model.model.end_to_end_event_coreference_pl_module import EndToEndEventCoreferenceModule
from easyecr.utils import log_utils
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_model import PlEcrModel
from easyecr.ecr_model.aggregation.aggregation import Attention
from easyecr.ecr_model.sample_generator.sample_generator import SimpleGenerator
from easyecr.utils import common_metrics


logger = log_utils.get_logger(__file__)


class EndToEndEventCoreferenceDataset(Dataset):
    def __init__(
        self, ecr_data: EcrData, coarse_type: str, train: bool = False, include_singleton: bool = True, times: int = 0
    ):
        self.ecr_data = ecr_data
        self.coarse_type = coarse_type
        self.train = train
        self.include_singleton = include_singleton
        self.times = times

        self.samples = []
        self.generate_samples()

    def generate_samples(self):
        """

        :return:
        """
        if self.train:
            sample_generator = SimpleGenerator(times=self.times)
        else:
            sample_generator = SimpleGenerator(times=0)
        samples = sample_generator.generate(self.ecr_data, self.coarse_type, self.include_singleton)
        positive_samples = [e + [1] for e in samples["positive"]]
        negative_samples = [e + [0] for e in samples["negative"]]
        self.samples.extend(positive_samples)
        print(len(positive_samples), len(negative_samples))
        self.samples.extend(negative_samples)

    def __len__(self):
        result = len(self.samples)
        return result

    def get_mention_repr(self, mention: Mention):
        """

        Args:
            mention:

        Returns:

        """
        meta = mention.meta
        mention_hiddens = meta["mention_hiddens"]
        hiddens_mask = [1] * mention_hiddens.shape[0]
        mention_length = meta["mention_length"]
        mention_hiddens_first = meta["mention_hiddens_first"]
        mention_hiddens_last = meta["mention_hiddens_last"]
        result = [mention_hiddens, hiddens_mask, [mention_length], mention_hiddens_first, mention_hiddens_last]
        return result

    def __getitem__(self, idx: int):
        """

        :param idx:
        :return:
        """
        sample = self.samples[idx]
        mention1: Mention = sample[0]
        mention2: Mention = sample[1]

        result = []
        first_tokenized_result = self.get_mention_repr(mention1)
        result.extend(first_tokenized_result)
        result.extend(self.get_mention_repr(mention2))
        result.append([sample[2]])
        result = [torch.tensor(e) for e in result]
        meta = {"mention_id1": mention1.mention_id, "mention_id2": mention2.mention_id}
        return result, meta

    @staticmethod
    def collate_fn(data_and_meta):
        """

        :param data:
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


class EndToEndEventCoreferenceModule(pl.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.validation_step_outputs = []
        self.bert_hidden_size = 1024
        self.embedding_dimension = 20
        self.dropout = 0.1
        self.self_attention_layer = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(self.bert_hidden_size, 1),
        )

        self.width_feature = nn.Embedding(5, self.embedding_dimension)

        self.with_width_embedding = True
        self.use_head_attention = True
        if self.use_head_attention:
            self.input_layer = self.bert_hidden_size * 3
        else:
            self.input_layer = self.bert_hidden_size * 2

        if self.with_width_embedding:
            self.input_layer += self.embedding_dimension

        self.input_layer *= 3

        self.mlp = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.input_layer, self.bert_hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
            # nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.bert_hidden_size, 1),
        )

        # self.head_finding_attention = Attention(in_features=768)
        self.criteria = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def get_mention_representions(
        self, mention_hiddens, hidden_mask, mention_length, mention_hiddens_first, mention_hiddens_last
    ):
        """

        Args:
            mention_hiddens:
            hidden_mask:
            mention_length:
            mention_hiddens_first:
            mention_hiddens_last:

        Returns:

        """
        repr = torch.cat([mention_hiddens_first, mention_hiddens_last], dim=1)

        if self.use_head_attention:
            attention_scores = self.self_attention_layer(mention_hiddens).squeeze(-1)
            attention_scores *= hidden_mask
            attention_scores = torch.where(
                attention_scores != 0, attention_scores, torch.tensor(-9e9, device=self.device)
            )
            attention_scores = torch.softmax(attention_scores, dim=1)
            attention_weighted_sum = (attention_scores.unsqueeze(-1) * mention_hiddens).sum(dim=1)

            # attention_weights = self.head_finding_attention(mention_hiddens, hidden_mask)
            # attention_weighted_sum = self.element_wise_mul(mention_hiddens, attention_weights)

            repr = torch.cat([repr, attention_weighted_sum], dim=1)

        if self.with_width_embedding:
            width = torch.clamp(mention_length, max=4)
            width_embedding = self.width_feature(width).squeeze(1)
            repr = torch.cat([repr, width_embedding], dim=1)
        return repr

    def print_tensor(self, tensor):
        """

        Args:
            tensor:

        Returns:

        """
        data = tensor.detach().cpu().numpy().tolist()
        for i, nums in enumerate(data):
            print(f'{i}: {["%.4f" % e for e in nums]}')

    def forward(
        self,
        mention_hiddens1,
        hidden_mask1,
        mention_length1,
        mention_hiddens_first1,
        mention_hiddens_last1,
        mention_hiddens2,
        hidden_mask2,
        mention_length2,
        mention_hiddens_first2,
        mention_hiddens_last2,
        labels,
        meta,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        result = {"meta": meta}
        mention_representations1 = self.get_mention_representions(
            mention_hiddens1, hidden_mask1, mention_length1, mention_hiddens_first1, mention_hiddens_last1
        )
        mention_representations2 = self.get_mention_representions(
            mention_hiddens2, hidden_mask2, mention_length2, mention_hiddens_first2, mention_hiddens_last2
        )
        repr_multiplication = mention_representations1 * mention_representations2
        pair_repr = torch.cat([mention_representations1, mention_representations2, repr_multiplication], dim=1)
        similarities = self.mlp(pair_repr)
        predictions = torch.sigmoid(similarities)
        result["predictions"] = predictions

        distances = 1 - predictions
        distances = distances
        distances_square = torch.square(distances)
        result["labels"] = labels
        result["distances"] = distances
        result["distances_square"] = distances_square

        result["instance_loss"] = self.criteria(similarities, labels.float())
        loss = torch.mean(result["instance_loss"])
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

    def on_ecr_epoch_end(self, step_outputs, mode: str):
        """

        Args:
            step_outputs:
            mode:

        Returns:

        """
        instance_losses = [e["instance_loss"] for e in step_outputs]
        all_losses = torch.cat(instance_losses)
        val_loss = torch.mean(all_losses)
        self.log(f"{mode}_loss", val_loss)

        labels = torch.cat([e["labels"] for e in step_outputs])
        print()
        print(f"{mode}_positive_instance_num: {torch.sum(labels)}")
        print(f"{mode}_negative_instance_num: {torch.sum(1 - labels)}")

        distances = torch.cat([e["distances"] for e in step_outputs])
        positive_distance = torch.sum((labels * distances)) / torch.sum(labels)
        negative_distance = torch.sum(((1 - labels) * distances)) / torch.sum((1 - labels))
        print()
        print(f"{mode}_positive_distance: {positive_distance}")
        print(f"{mode}_negative_distance: {negative_distance}")

        distances_square = torch.cat([e["distances_square"] for e in step_outputs])
        positive_distance_square = torch.sum((labels * distances_square)) / torch.sum(labels)
        negative_distance_square = torch.sum(((1 - labels) * distances_square)) / torch.sum((1 - labels))
        print()
        print(f"{mode}_positive_distance_square: {positive_distance_square}")
        print(f"{mode}_negative_distance_square: {negative_distance_square}")

        predictions_real = torch.cat([e["predictions"] for e in step_outputs])
        predictions = predictions_real > 0.5
        predictions = torch.squeeze(predictions)
        labels = torch.squeeze(torch.cat([e["labels"] for e in step_outputs]))
        print(f"{mode}_accuracy:", common_metrics.accuracy(predictions, labels))
        print(f"{mode}_precision:", common_metrics.precision(predictions, labels))
        print(f"{mode}_recall:", common_metrics.recall(predictions, labels))
        print(f"{mode}_f1:", common_metrics.f1_score(predictions, labels))

        step_outputs.clear()

    def on_train_epoch_end(self):
        """

        :return:
        """
        self.on_ecr_epoch_end(self.training_step_outputs, "train")

    def on_validation_epoch_end(self):
        """

        :return:
        """
        self.on_ecr_epoch_end(self.validation_step_outputs, "val")

    def get_optimizer_and_scheduler(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), self.hparams.learning_rate)
        else:
            raise NotImplementedError

        return optimizer, None

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        return optimizer

    def predict_step(self, batch, batch_idx: int):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        result = self(*batch)
        return result


class EndToEndEcrModel(PlEcrModel):
    """
    2021-findings-of-acl-Cross-document Coreference Resolution over Predicted Mention
    2021-NAACL-WEC: Deriving a Large-scale Cross-document Event Coreference dataset from Wikipedia
    2017-emnlp-End-to-end Neural Coreference Resolution
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
        pass

    def instanciate_module(self):
        module = EndToEndEventCoreferenceModule(self.conf["module"])
        return module

    def load_module(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        result = EndToEndEventCoreferenceModule.load_from_checkpoint(filepath)
        return result

    def prepare_data(self, data: EcrData, mode: str):
        """

        Args:
            data:
            mode: train, evaluate, predict

        Returns:

        """
        if data is None:
            return None, None
        coarse_type = self.conf["common"][f"{mode}_topic"]
        times = self.conf["dataset"]["times"]
        include_singleton = self.conf["dataset"]["include_singleton"]
        dataset = EndToEndEventCoreferenceDataset(
            data, coarse_type, train=(mode == "train"), include_singleton=include_singleton, times=times
        )

        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=EndToEndEventCoreferenceDataset.collate_fn,
            num_workers=num_workers,
            shuffle=(mode == "train"),
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

    # def train(self, train_data: EcrData, dev_data: EcrData):
    #     """
    #
    #     Args:
    #         train_data:
    #         dev_data:
    #
    #     Returns:
    #
    #     """
    #     trainer_parameters = self.trainer_parameters
    #     model_checkpoint_callback = self.instantiate_model_checkpoint_callback()
    #     trainer_parameters['callbacks'] = [model_checkpoint_callback]
    #     trainer_parameters['callbacks'].extend(self.add_callbacks(train_data, dev_data))
    #     trainer = pl.Trainer(**trainer_parameters)
    #
    #     # train_dataset, train_dataloaders = self.prepare_data(train_data, mode='train')
    #     dev_dataset, dev_dataloaders = self.prepare_data(dev_data, mode='train')
    #     train_dataloaders = dev_dataloaders
    #
    #     epoch_num = 10
    #     module: EndToEndEventCoreferenceModule = self.module
    #     optimizer = module.configure_optimizers()
    #     for i in range(epoch_num):
    #         logger.info(f'epoch {i} start')
    #         module.train()
    #         for j, batch_data in enumerate(tqdm(train_dataloaders[0])):
    #             forward_output = module.training_step([batch_data], j)
    #             loss = forward_output['loss']
    #             loss.backward()
    #             optimizer.step()
    #
    #             torch.cuda.empty_cache()
    #         module.on_train_epoch_end()
    #
    #         module.eval()
    #         for j, batch_data in enumerate(tqdm(dev_dataloaders[0])):
    #             val_forward_output = module.validation_step(batch_data, j)
    #             # forward_output = module.training_step([batch_data], j)
    #         # module.on_validation_epoch_end()
    #         module.on_validation_epoch_end()
    #         logger.info(f'epoch {i} end')
