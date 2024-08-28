from typing import Dict, Union, List
import torch
from torch import nn
import pytorch_lightning as pl

from transformers import AutoConfig
from transformers import AutoModel
from transformers import RobertaModel
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup


class EventCoreferenceModule(pl.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.config = AutoConfig.from_pretrained(conf['transformer_model'])
        self.encoder = AutoModel.from_config(self.config)
        self.margin = 1.0
        self.validation_step_outputs = []
        self.hidden_size = self.config.hidden_size
        self.mention_mlp = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_mlp = nn.Linear(self.hidden_size, self.hidden_size)
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

    def forward(self, mentions1, attention_masks1,
                mention_masks1, mention_left1, mention_right1,
                # token_type_ids1,
                mentions2, attention_masks2,
                mention_masks2, mention_left2, mention_right2,
                # token_type_ids2,
                labels,
                *args, **kwargs) \
            -> Dict[str, torch.Tensor]:
        result = {}
        mention_representations1 = self.get_mention_representions(mentions1, attention_masks1, mention_masks1)
        result['mention_representations1'] = mention_representations1
        if mentions2 is not None:
            mention_representations2 = self.get_mention_representions(mentions2, attention_masks2, mention_masks2)
            distances = 1 - torch.cosine_similarity(mention_representations1,
                                                    mention_representations2,
                                                    )
            # distances = self.pdist(mention_representations1, mention_representations2)
            distances = torch.unsqueeze(distances, dim=1)
            distances_square = torch.square(distances)
            # print()
            # print(f'label: {labels.detach().cpu().numpy()}')
            # print(f'distances: {distances.detach().cpu().numpy()}')
            # print(f'distances_square: {distances_square.detach().cpu().numpy()}')
            result['labels'] = labels
            result['distances'] = distances
            result['distances_square'] = distances_square
            one_minus_labels = 1 - labels
            loss = labels * distances_square \
                   + one_minus_labels * torch.square(torch.clamp(self.margin - distances, min=0))
            # loss = labels * distances_square + one_minus_labels * (1.0 - distances_square)
            result['instance_loss'] = loss
            loss = torch.mean(loss)
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
