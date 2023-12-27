import matplotlib.pyplot as plt
import numpy as np
import torch


def confusion_matrix(predictions: torch.Tensor, ground_truth: torch.Tensor, num_classes: int):
    """
        Построение и визуализация confusion matrix
            confusion_matrix - матрица NxN, где N - кол-во классов в наборе данных
            confusion_matrix[i, j] - кол-во элементов класса "i", которые классифицируются как класс "j"

        :return plt.gcf() - matplotlib figure
        TODO: реализуйте построение и визуализацию confusion_matrix, подпишите оси на полученной визуализации, добавьте значение confusion_matrix[i, j] в соотвествующие ячейки на изображении
    """
    conf_matrix = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            count = 0
            for k in range(len(predictions)):
                if predictions[k] == i and ground_truth[k] == j:
                    count += 1
            conf_matrix[i,j] = count
        
    plt.cla(), plt.clf()
    plt.imshow(conf_matrix)
    return plt.gcf()
