from pyexpat import model


class ModelArch:
    names = 1
    def __init__(self, config_dict, name = None, acc = None) -> None:
        self.name = str(ModelArch.names) if name is None else name
        self.accuracy = acc
        self.config_dict = config_dict
        ModelArch.names += 1

    @property
    def depth(self):
        return self.config_dict['depths']
    
    @property
    def kernel(self):
        return self.config_dict['kernel_sizes']

    @property
    def expansion_ratio(self):
        return self.config_dict['expansion_ratios']