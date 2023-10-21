import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from datasets.KMNIST import KMNIST
from models.MLP import MLP
from datasets.utils.prepare_transforms import prepare_transforms


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.__prepare_data(self.cfg.dataset_cfg)
        self.__prepare_model(self.cfg.model_cfg)

    def __prepare_data(self, dataset_cfg):
        """ Подготовка обучающих и тестовых данных """
        self.train_dataset = KMNIST(dataset_cfg, 'train',
                                    transforms=prepare_transforms(dataset_cfg.transforms['train']))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        self.test_dataset = KMNIST(dataset_cfg, 'test', transforms=prepare_transforms(dataset_cfg.transforms['test']))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False)

    def __prepare_model(self, model_cfg):
        """ Подготовка нейронной сети"""
        self.model = MLP(model_cfg)
        self.criterion = nn.CrossEntropyLoss()

    def save_model(self, filename):
        """
            Сохранение весов модели с помощью torch.save()
            :param filename: str - название файла
            TODO: реализовать сохранение модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        raise NotImplementedError

    def load_model(self, filename):
        """
            Загрузка весов модели с помощью torch.load()
            :param filename: str - название файла
            TODO: реализовать выгрузку весов модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        raise NotImplementedError

    def make_step(self, batch):
        """
            Инференс нейронной сети
            :param batch: dict of data keys ("image", "label")
            :return: выход
            # TODO: реализуйте инференс модели для данных batch, посчитайте значение целевой функции
        """
        raise NotImplementedError


if __name__ == '__main__':
    from configs.train_cfg import cfg

    trainer = Trainer(cfg)

    batch = next(iter(trainer.train_dataloader))
    trainer.make_step(batch)
