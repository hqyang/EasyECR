import copy
from typing import List

from omegaconf import OmegaConf
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.utils import log_utils

logger = log_utils.get_logger(__file__)


class EcrComplexTagger(EcrTagger):
    def __init__(self, conf: DictConfig):
        self.conf = conf
        self.tokenizer = None
        self.build_tokenizer()

        self.build_module()

    def build_tokenizer(self):
        raise NotImplementedError()

    def build_module(self):
        if self.conf["common"]["mode"] == "train":
            self.module = self.instanciate_module()
        else:
            model_checkpoint = self.conf["module"]["model_checkpoint"]
            self.module = self.load_module(model_checkpoint)

    def instanciate_module(self):
        raise NotImplementedError()

    def load_module(self, filepath: str):
        raise NotImplementedError()

    def prepare_data(self, data: EcrData, mode: str):
        raise NotImplementedError()

    def instanciate_callbacks(self, callbacks_config):
        result = []
        for callback_name, config in callbacks_config.items():
            if callback_name == "EarlyStopping":
                callback = EarlyStopping(**config)
            elif callback_name == "ModelCheckpoint":
                callback = ModelCheckpoint(**config)
            else:
                raise NotImplementedError(callback_name)
            result.append(callback)
        return result

    def train(self, train_data: EcrData, dev_data: EcrData):
        trainer_parameters = copy.deepcopy(self.conf["trainer_parameters"])
        trainer_parameters = OmegaConf.to_container(trainer_parameters, resolve=True)
        trainer_parameters["callbacks"] = self.instanciate_callbacks(trainer_parameters["callbacks"])
        trainer = pl.Trainer(**trainer_parameters)

        train_dataset, train_dataloaders = self.prepare_data(train_data, mode="train")
        dev_dataset, dev_dataloaders = self.prepare_data(dev_data, mode="train")

        trainer.fit(self.module, train_dataloaders=train_dataloaders, val_dataloaders=dev_dataloaders)
