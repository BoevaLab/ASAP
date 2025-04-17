from torch import nn

class ModelConfig:
    def __init__(self, name: str, model: nn.Module, bin_size: int, output_window: int, batch_size: int, learning_rate: float, use_map: bool):
        self.name = name
        self.model = model
        self.bin_size = bin_size
        self.output_window = output_window
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_map = use_map

    def __repr__(self):
        return f'ModelDataSource={self.__dict__.__str__()}'
