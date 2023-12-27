import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()

        self.cfg = cfg

        self.layers = self._init_layers()
        self.apply(self._init_weights)

    def _init_layers(self):
        """
            Инициализация слоев нейронной сети.

            Описание необходимых слоев задается в self.cfg.layers в следующем виде:
                 [
                    (name_1, params_1),
                    ...,
                    (name_N, params_N)
                ],
                где name_i (str) - название класса из nn (Linear, ReLU и т.д.), params_i (dict) - параметры этого слоя
            :return: список инициализированных слоев

            TODO: необходимо инициализировать слои, заданные с помощью self.cfg.layers, данная функция должна быть
                универсальной (описание модели вынесено в конфиг, основной код не должен меняться в зависимости от
                эксперимента), можно использовать getattr(nn, name) и nn.Sequential/nn.ModuleList
        """
        layers = nn.ModuleList()
        for layer_config in self.cfg.layers:
            #            name, params = layer_config
            layer = getattr(nn, layer_config[0])(**layer_config[1])
            layers.append(layer)
        return layers

    @torch.no_grad()
    def _init_weights(self, m: nn.Module):
        """
            Инициализация параметров линейный слоев согласно заданному типу self.cfg.init_type.
            TODO: реализуйте этот метод, можно использовать getattr(nn.init, self.cfg.init_type)
        """
        if (isinstance(m,nn.Linear)):
            getattr(nn.init, self.cfg.init_type)(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_number_of_parameters(self):
        print(f'Общее число обучаемых параметров модели: {self.number_of_parameters()}')

    def forward(self, inputs: torch.Tensor):
        """
            Forward pass нейронной сети, все вычисления производятся для батча
            :param inputs: torch.Tensor(batch_size, height, weight)
            :return output of the model: torch.Tensor(batch_size, nrof_classes) - уверенность

            TODO: реализуйте этот метод
        """
        res = inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2])
        
        for layer in self.layers:
            res = layer(res)
        return res

    def predict_forward(self, forward: torch.Tensor):
        """
        returns the prediction for the forward pass
        """
        new_var = torch.argmax(forward, dim=1)
        return new_var

if __name__ == '__main__':
    from configs.mlp_cfg import cfg

    model = MLP(cfg)

    # TODO: вывести количество обучаемых параметров нейронной сети
    
    nrof_params = model.number_of_parameters()
    print(f'number of trainable parameters: {nrof_params}')