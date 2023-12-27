import numpy as np
import torch


# def accuracy(*args, **kwargs):
#     """
#         Вычисление точности:
#             accuracy = sum( predicted_class == ground_truth ) / N, где N - размер набора данных
#         TODO: реализуйте подсчет accuracy
#     """
#     predictions = kwargs['predictions']
#     ground_truth = kwargs['ground_truth']
#     num_of_classes = len(predictions)
    
#     return np.sum(predictions == ground_truth) / num_of_classes

def accuracy(predictions, ground_truth):
    """
        Вычисление точности:
            accuracy = sum( predicted_class == ground_truth ) / N, где N - размер набора данных
        TODO: реализуйте подсчет accuracy
    """
    num_of_classes = len(predictions)
    with torch.no_grad():
        return torch.sum(predictions == ground_truth) / num_of_classes