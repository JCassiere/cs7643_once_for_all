import copy
import time
import torch
from datasets import get_dataloaders
from utils import get_device
from mobilenetv3_ofa import MobileNetV3OFA

class Experiment:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
        self.device = get_device()
        # dataset setup
        self.dataset_name = kwargs.get("dataset_name", "cifar10")
        self.train_data_loader, self.val_data_loader = get_dataloaders(self.device, self.dataset_name)
        
        self.experiment_name = kwargs.get("experiment_name", time.time())
        self.save_dir = "./checkpoint/" + self.experiment_name + "/"
        self.base_net_epochs = kwargs.get("base_net_epochs", 180)
        self.base_net_lr = kwargs.get("base_net_lr", .026)
        self.overall_kernel_choices = sorted(kwargs.get("kernel_choices", [3, 5]), reverse=True)
        self.elastic_kernel_epochs = kwargs.get("elastic_kernel_epochs", 125)
        self.elastic_kernel_lr = kwargs.get("elastic_kernel_lr", 0.0096)
        self.overall_depth_choices = sorted(kwargs.get("depth_choices", [2, 3, 4]), reverse=True)
        self.elastic_depth_epochs_stage_1 = kwargs.get("elastic_depth_epochs_stage_1", 25)
        self.elastic_depth_lr_stage_1 = kwargs.get("elastic_depth_lr_stage_1", 0.0008)
        self.elastic_depth_epochs_stage_2 = kwargs.get("elastic_depth_epochs_stage_2", 125)
        self.elastic_depth_lr_stage_2 = kwargs.get("elastic_depth_lr_stage_2", 0.0024)
        self.overall_expansion_ratio_choices = sorted(kwargs.get("expansion_ratio_choices", [3, 4, 6]), reverse=True)
        self.elastic_width_epochs_stage_1 = kwargs.get("elastic_width_epochs_stage_1", 25)
        self.elastic_width_lr_stage_1 = kwargs.get("elastic_width_lr_stage_1", 0.0008)
        self.elastic_width_epochs_stage_2 = kwargs.get("elastic_width_epochs_stage_2", 125)
        self.elastic_width_lr_stage_2 = kwargs.get("elastic_width_lr_stage_2", 0.0024)
        self.dropout = kwargs.get("dropout", 0.1)
        self.width_mult = kwargs.get("width_mult", 1)
        
        # Network initialization
        self.net_output_widths = kwargs.get("output_widths", [16, 16, 24, 40, 80, 112, 160, 960, 1280])
        self.use_squeeze_excites = kwargs.get("use_squeeze_excites", [False, False, False, True, False, True, True, False, False])
        self.use_hard_swishes = kwargs.get("use_hard_swishes", [True, False, False, False, True, True, True, True, True])
        self.strides = kwargs.get("strides", [2, 1, 2, 2, 2, 1, 2, 1, 1])
        
        input_data_channels = 3
        if self.dataset_name in ["mnist", "cifar10"]:
            num_classes = 10
        elif self.dataset_name == "cifar100":
            num_classes = 100
        else:
            raise ValueError("{} dataset not supported".format(self.dataset_name))
        
        max_kernel_size = self.overall_kernel_choices[0]
        max_depth = self.overall_depth_choices[0]
        max_expansion_ratio = self.overall_expansion_ratio_choices[0]
        
        self.net = MobileNetV3OFA(
            output_widths=self.net_output_widths,
            use_squeeze_excites=self.use_squeeze_excites,
            use_hard_swishes=self.use_hard_swishes,
            strides=self.strides,
            input_data_channels=input_data_channels,
            num_classes=num_classes,
            width_mult=self.width_mult,
            max_kernel_size=max_kernel_size,
            max_expansion_ratio=max_expansion_ratio,
            max_depth=max_depth,
            dropout=self.dropout
        )
        self.current_stage_train_accuracies = []
        self.current_stage_val_accuracies = []
        self.teacher = None
    
    def log(self, stage: str):
        torch.save(self.net.state_dict(), self.save_dir + stage + '.pt')
    
    def load_net_post_stage(self, stage: str):
        checkpoint_file = "experiments/{}.pt".format(stage)
        self.net.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
        
    def append_train_accuracy(self, accuracy: float):
        self.current_stage_train_accuracies.append(accuracy)
        
    def append_val_accuracies(self, accuracies: dict):
        self.current_stage_val_accuracies.append(accuracies)
    
    def set_teacher(self):
        self.teacher = copy.deepcopy(self.net)
    
    def get_teacher(self):
        # TODO - update teacher after each stage?
        #  maybe check whether the original teacher or current net gives better results
        #  for the full network
        return self.teacher
        
    # TODO
    def from_config(self):
        pass
    
    def save_config(self):
        pass
    