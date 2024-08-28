from typing import Dict, Union, List
import torch
from torch import nn
import pytorch_lightning as pl

from transformers import AutoConfig
from transformers import AutoModel
from transformers import RobertaModel
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup

from easyecr.ecr_model.aggregation.aggregation import Attention


class EndToEndEventCoreferenceModule(pl.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.validation_step_outputs = []
        self.hidden_size = 768 * 9
        self.mlp2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp1 = nn.Linear(self.hidden_size, 1)
        self.head_finding_attention = Attention(in_features=768)
        self.criteria = torch.nn.BCEWithLogitsLoss(reduce=False)

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

    def get_mention_representions(self, mention_hiddens, hidden_mask, mention_length, mention_hiddens_first,
                                  mention_hiddens_last):
        """

        Args:
            mention_hiddens:
            hidden_mask:
            mention_length:
            mention_hiddens_first:
            mention_hiddens_last:

        Returns:

        """
        attention_weights = self.head_finding_attention(mention_hiddens, hidden_mask)
        attention_weighted_sum = self.element_wise_mul(mention_hiddens, attention_weights)
        repr = torch.cat([mention_hiddens_first, mention_hiddens_last, attention_weighted_sum], dim=1)
        return repr

    def forward(self, mention_hiddens1, hidden_mask1, mention_length1, mention_hiddens_first1, mention_hiddens_last1,
                mention_hiddens2, hidden_mask2, mention_length2, mention_hiddens_first2, mention_hiddens_last2,
                labels,
                *args, **kwargs) \
            -> Dict[str, torch.Tensor]:
        result = {}
        mention_representations1 = self.get_mention_representions(mention_hiddens1, hidden_mask1, mention_length1,
                                                                  mention_hiddens_first1, mention_hiddens_last1)
        mention_representations2 = self.get_mention_representions(mention_hiddens2, hidden_mask2, mention_length2,
                                                                  mention_hiddens_first2, mention_hiddens_last2)
        repr_multiplication = mention_representations1 * mention_representations2
        pair_repr = torch.cat([mention_representations1, mention_representations2, repr_multiplication], dim=1)
        similarities = self.mlp1(torch.relu(self.mlp2(pair_repr)))

        distances = 1 - torch.sigmoid(similarities)
        distances = distances
        distances_square = torch.square(distances)
        result['labels'] = labels
        result['distances'] = distances
        result['distances_square'] = distances_square

        result['instance_loss'] = self.criteria(similarities, labels.float())
        loss = torch.mean(result['instance_loss'])
        result['loss'] = loss
        return result

    def training_step(self, batch, batch_idx):
        forward_output = self.forward(*batch[0])
        self.log('train_loss', forward_output["loss"])
        return forward_output

    def validation_step(self, batch, batch_idx: int, *args, **kwargs):
        forward_output = self.forward(*batch)
        self.log('val_loss', forward_output["loss"])
        self.validation_step_outputs.append(forward_output)
        return forward_output

    def on_validation_epoch_end(self):
        """

        :return:
        """
        instance_losses = [e['instance_loss'] for e in self.validation_step_outputs]
        all_losses = torch.cat(instance_losses)
        val_loss = torch.mean(all_losses)
        self.log('val_loss', val_loss)

        labels = torch.cat([e['labels'] for e in self.validation_step_outputs])
        distances = torch.cat([e['distances'] for e in self.validation_step_outputs])
        positive_distance = torch.sum((labels * distances)) / torch.sum(labels)
        negative_distance = torch.sum(((1 - labels) * distances)) / torch.sum((1 - labels))
        self.log('positive_distance', positive_distance)
        self.log('negative_distance', negative_distance)
        print()
        print(f'positive_distance: {positive_distance}')
        print(f'negative_distance: {negative_distance}')

        distances_square = torch.cat([e['distances_square'] for e in self.validation_step_outputs])
        positive_distance_square = torch.sum((labels * distances_square)) / torch.sum(labels)
        negative_distance_square = torch.sum(((1 - labels) * distances_square)) / torch.sum((1 - labels))
        self.log('positive_distance_square', positive_distance_square)
        self.log('negative_distance_square', negative_distance_square)
        print()
        print(f'positive_distance_square: {positive_distance_square}')
        print(f'negative_distance_square: {negative_distance_square}')

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
