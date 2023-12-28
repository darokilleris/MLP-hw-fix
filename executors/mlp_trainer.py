
import os
import pickle
import numpy as np
# from sklearn.base import accuracy_score

import torch
import torch.nn as nn
# import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets.KMNIST import KMNIST
from models.MLP import MLP
from datasets.utils.prepare_transforms import prepare_transforms
from utils.metrics import accuracy
from utils.visualization import confusion_matrix
from logs.Logger import Logger
from typing import Union


class Trainer():
    hyperparameters :dict
    
    def __init__(self, cfg):
        self.cfg = cfg
        # TODO: настройте логирование с помощью класса Logger
        self.logger = Logger(env_path='env.env', project='danipek/nothin')
        #  (пример: https://github.com/KamilyaKharisova/mllib_f2023/blob/master/logginig_example.py)

        # TODO: залогируйте используемые гиперпараметры в neptune.ai через метод log_hyperparameters
        self.hyperparameters = {
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'n_epochs': cfg.epochs,
            'optimizer': cfg.optimizer_name,
        }
        self.logger.log_hyperparameters(self.hyperparameters)

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
        if (not cfg.load_existing_model):
            print('chose to initialize new model. Initializing...')
            self.model = MLP(model_cfg)
            print('model initialized')
        else:
            print('chose to load existing model. loading...')
            self.model = self.load_model(model_cfg, filename='best_model')
            print('model loaded')
            
        self.criterion = nn.CrossEntropyLoss()

        # TODO: инициализируйте оптимайзер через getattr(torch.optim, self.cfg.optimizer_name)
        self.optimizer = getattr(torch.optim, self.cfg.optimizer_name)(self.model.parameters(),self.cfg.lr)

    def save_model(self, filename,override_only_if_better=False, stats:dict = None):
        """
            Сохранение весов модели с помощью torch.save()
            :param filename: str - название файла
            TODO: реализовать сохранение модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        print('\nSaving model...')
        for key,value in self.hyperparameters.items():
            stats[key]=value
        
        stats['experiment_name'] = 'best_by_now'
        self.logger.run._name = stats['experiment_name']
        path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        if (not override_only_if_better):
            torch.save(self.model.state_dict(), path)
            pickle.dump(stats,open(os.path.join(self.cfg.exp_dir, f"{filename}_stats.pkl"),"wb"))
            new_accuracy = stats['accuracy']
            print(f'Model saved to {path}.\nAccuracy: {new_accuracy}')
        else:
            try:
                prev_stats = pickle.load(open(os.path.join(self.cfg.exp_dir, f"{filename}_stats.pkl"),"rb"))
                print(prev_stats)
                prev_accuracy = prev_stats['accuracy']
                new_accuracy = stats['accuracy']   
            except EOFError:
                torch.save(self.model.state_dict(), path)
                pickle.dump(stats,open(os.path.join(self.cfg.exp_dir, f"{filename}_stats.pkl"),"wb"))
                new_accuracy = stats['accuracy']
                print(f'Model saved to {path}.\n accuracy: {new_accuracy}')
                return
            if (stats['accuracy'] > prev_stats['accuracy']):
                torch.save(self.model.state_dict(), path)
                pickle.dump(stats,open(os.path.join(self.cfg.exp_dir, f"{filename}_stats.pkl"),"wb"))
                print(f'Model saved to {path} because accuracy increased from {prev_accuracy} to {new_accuracy}')
            else : print(f'Model was not saved because accuracy did not increase. \nPrevious accuracy: {prev_accuracy} \nNew accuracy: {new_accuracy}')
                    
            
            

    def load_model(self, model_cfg, filename) -> MLP:
        """
            Загрузка весов модели с помощью torch.load()
            :param filename: str - название файла
            TODO: реализовать выгрузку весов модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        # Получение весов из файла filename, в который ранее были сохранены веса модели
        path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        model = MLP(model_cfg)
        model.load_state_dict(torch.load(path))
        return model

    def make_step(self, batch, update_model=True):
        """
            Этот метод выполняет один шаг обучения, включая forward pass, вычисление целевой функции,
            backward pass и обновление весов модели (если update_model=True).

            :param batch: dict of data with keys ["image", "label"]
            :param update_model: bool - если True, необходимо сделать backward pass и обновить веса модели
            :return: значение функции потерь, выход модели
            # TODO: реализуйте инференс модели для данных batch, посчитайте значение целевой функции
        """
        if (update_model):
            self.optimizer.zero_grad()
            forward = self.model.forward(batch["image"])
            loss = self.criterion(forward, batch["label"])
            if update_model:
                loss.backward()
                self.optimizer.step()
                # print(f'Loss: {loss.item():.4f}')
                
                output = self.model.predict_forward(forward)
            return loss, output
        else:
            with torch.no_grad():
                forward = self.model.forward(batch["image"])
                loss = self.criterion(forward, batch["label"])
                output = self.model.predict_forward(forward)
            return loss, output

    def train_epoch(self, *args, **kwargs):
        """
            Обучение модели на self.train_dataloader в течение одной эпохи. Метод проходит через все обучающие данные и
            вызывает метод self.make_step() на каждом шаге.

            TODO: реализуйте функцию обучения с использованием метода self.make_step(batch, update_model=True),
                залогируйте на каждом шаге значение целевой функции и accuracy на batch
        """
        self.model.train()
        for i, (batch_data) in enumerate(self.train_dataloader):
            loss = self.make_step(batch_data, update_model=True)[0]
            if (i%100 == 0):
                print(f'Loss: {loss:.4f}')
        pass

    def evaluate(self, *args, **kwargs):
        """
            Метод используется для проверки производительности модели на обучающих/тестовых данных. Сначала модель
            переводится в режим оценки (model.eval()), затем данные последовательно подаются на вход модели, по
            полученным выходам вычисляются метрики производительности, такие как значение целевой функции, accuracy

            :param dataset: str - имя датасета, на котором будет производиться оценка, может принимать значения 'train', 'test'
            :return: значение целевой функции, accuracy
            
            TODO: реализуйте функцию оценки с использованием метода self.make_step(batch, update_model=False),
                залогируйте значения целевой функции и accuracy, постройте confusion_matrix
        """
        self.model.eval()
        if (kwargs['dataset'] == 'train'):
            raise NotImplementedError
        elif(kwargs['dataset'] == 'test'):
            loader = self.train_dataloader
            predicted = 0
            total_loss = 0
            for i, batch_data in enumerate(loader):
                loss, predictions = self.make_step(batch_data, update_model=False)
                predicted += torch.sum(predictions == batch_data["label"])
                total_loss += loss.item()
                accuracy_on_batch = accuracy(predictions, batch_data["label"])
                if (i%100 == 0):
                    print(f'loss on batch No.{i}: {loss}. accuracy on batch No.{i}: {accuracy_on_batch}')
                    self.logger.save_param(
                        kwargs['dataset'],
                        [f'loss on Batch No.{i}',f'accuracy on Batch No.{i}'],
                        [loss,accuracy_on_batch])
                    self.logger.save_plot(kwargs['dataset'],
                                        'confusion_matrix',
                                        confusion_matrix(predictions, batch_data["label"],
                                                        cfg.dataset_cfg.nrof_classes
                                                        )
                                        )
                    
                    
            final_accuracy = predicted/len(loader.dataset)
            print(f'final loss: {total_loss}, final accuracy: {final_accuracy}')
            
            stats = {
                'loss': total_loss,
                'accuracy': final_accuracy
                }
            self.logger.save_param(
                kwargs['dataset'],
                ['final loss','final accuracy'],
                [total_loss, final_accuracy]
            )
            self.save_model('best_model',override_only_if_better=True, stats = stats)
            return stats


    def fit(self, *args, **kwargs):
        """
            Основной цикл обучения модели. Данная функция должна содержать один цикл на заданное количество эпох.
            На каждой эпохе сначала происходит обучение модели на обучающих данных с помощью метода self.train_epoch(),
            а затем оценка производительности модели на тестовых данных с помощью метода self.evaluate()

            # TODO: реализуйте основной цикл обучения модели, сохраните веса модели с лучшим значением accuracy на
                тестовой выборке
        """
        for epoch in range(self.cfg.epochs):
            # Обучение на одной эпохе
            self.train_epoch()
            print(f'Epoch {epoch+1}/{self.cfg.epochs}')
        pass

    def overfitting_on_batch(self, max_step=100):
        """
            Оверфиттинг на одном батче. Эта функция может быть полезна для отладки и оценки способности вашей
            модели обучаться и обновлять свои веса в ответ на полученные данные.
        """
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 10 == 0:
                acc = accuracy(output, batch['label'])
                print('[{:d}]: loss - {:.4f}, {:.4f}'.format(step, loss, acc))


if __name__ == '__main__':
    from configs.train_cfg import cfg

    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    # trainer.overfitting_on_batch()

    # обучение нейронной сети
    # trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate(dataset = 'test')
    
    # trainer.
