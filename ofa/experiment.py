import copy
import os
import time
import json
import torch
from ofa.datasets import get_dataloaders
from ofa.utils import get_device
from ofa.mobilenetv3_ofa import MobileNetV3OFA


class Experiment:
    EXPERIMENTS_DIR = "./experiments/"
    
    def __init__(self, save=True, **kwargs):
        self.device = get_device()
        # dataset setup
        # choices = {cifar10, cifar100, mnist}
        self.dataset_name = kwargs.get("dataset_name", "cifar10")
        self.train_data_loader, self.val_data_loader = get_dataloaders(self.device, self.dataset_name)
        
        self.experiment_name = kwargs.get("experiment_name", "default_{}".format(int(time.time())))
        self.save_dir = self.EXPERIMENTS_DIR + self.experiment_name + "/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
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
        
        self.net = None
        self.set_net()
        self.current_stage_train_accuracies = []
        self.current_stage_val_accuracies = {}
        self.teacher = None
        if save:
            self.save_config()
    
    def set_net(self):
        self.net = None
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
        self.net.to(self.device)
    
    def log(self, stage: str, clear: bool = True):
        torch.save(self.net.state_dict(), self.save_dir + stage + '.pt')
        
        results = {
            'train_accuracies': self.current_stage_train_accuracies,
            'val_accuracies': self.current_stage_val_accuracies
        }
        with open(self.save_dir + stage + "_results.json", 'w') as file:
            json.dump(results, file)
        
        if clear:
            self.clear_results()
        
    def clear_results(self):
        self.current_stage_train_accuracies = []
        self.current_stage_val_accuracies = {}
    
    def load_net_post_stage(self, stage: str):
        checkpoint_file = self.save_dir + stage + ".pt"
        self.net.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
        
    def append_train_accuracy(self, accuracy: float):
        self.current_stage_train_accuracies.append(accuracy)
        
    def append_val_accuracies(self, accuracies: dict):
        for key, value in accuracies.items():
            if key in self.current_stage_val_accuracies:
                self.current_stage_val_accuracies[key].append(value)
            else:
                self.current_stage_val_accuracies[key] = [value]
    
    def set_teacher(self):
        self.teacher = copy.deepcopy(self.net)
    
    def load_teacher(self, stage):
        if not self.teacher:
            self.set_teacher()
        big_net_checkpoint = self.save_dir + stage + ".pt"
        self.teacher.load_state_dict(torch.load(big_net_checkpoint, map_location=self.device))
        
    def get_teacher(self, stage="big_network"):
        self.load_teacher(stage)
        return self.teacher
        
    def save_config(self):
        config = copy.deepcopy(self.__dict__)
        config.pop("net")
        config.pop("device")
        config.pop("train_data_loader")
        config.pop("val_data_loader")
        config.pop("current_stage_train_accuracies")
        config.pop("current_stage_val_accuracies")
        config.pop("teacher")
        with open(self.save_dir + "config.json", 'w') as file:
            json.dump(config, file, indent=4)
    
    def load_from_config(self, experiment_name, teacher_stage="big_network"):
        load_dir = self.EXPERIMENTS_DIR + experiment_name + "/"
        with open(load_dir + "config.json") as file:
            self.__dict__ = json.load(file)
        self.device = get_device()
        self.train_data_loader, self.val_data_loader = get_dataloaders(self.device, self.dataset_name)
        self.set_net()
        self.current_stage_train_accuracies = []
        self.current_stage_val_accuracies = {}
        self.teacher = None
        self.load_teacher(teacher_stage)
        return self
    
    def load_from_previous_experiment(self, previous_experiment, stage_to_start_from, teacher_stage="big_network"):
        previous_experiment.load_net_post_stage(stage_to_start_from)
        previous_experiment.load_teacher(teacher_stage)
        self.net = copy.deepcopy(previous_experiment.net)
        self.teacher = copy.deepcopy(previous_experiment.teacher)

def experiment_from_config(experiment_name, load_stage, teacher_stage="big_network"):
    experiment = Experiment(save=False).load_from_config(experiment_name, teacher_stage=teacher_stage)
    experiment.load_net_post_stage(load_stage)
    return experiment
    