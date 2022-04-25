
class Experiment:
    def __init__(self, name, **kwargs):
        self.base_net_epochs = kwargs.get("base_net_epochs", 180)
        self.base_net_lr = kwargs.get("base_net_lr", 2.6)
        self.overall_kernel_choices = kwargs.get("kernel_choices", [3, 5])
        self.elastic_kernel_epochs = kwargs.get("elastic_kernel_epochs", 125)
        self.elastic_kernel_lr = kwargs.get("elastic_kernel_lr", 0.96)
        self.overall_depth_choices = kwargs.get("depth_choices", [2, 3, 4])
        self.elastic_depth_epochs_stage_1 = kwargs.get("elastic_depth_epochs_stage_1", 25)
        self.elastic_depth_lr_stage_1 = kwargs.get("elastic_depth_lr_stage_1", 0.08)
        self.elastic_depth_epochs_stage_2 = kwargs.get("elastic_depth_epochs_stage_2", 125)
        self.elastic_depth_lr_stage_2 = kwargs.get("elastic_depth_lr_stage_2", 0.24)
        self.overall_expansion_ratio_choices = kwargs.get("expansion_ratio_choices", [3, 4, 6])
        self.elastic_width_epochs_stage_1 = kwargs.get("elastic_width_epochs_stage_1", 25)
        self.elastic_width_lr_stage_1 = kwargs.get("elastic_width_lr_stage_1", 0.08)
        self.elastic_width_epochs_stage_2 = kwargs.get("elastic_width_epochs_stage_2", 125)
        self.elastic_width_lr_stage_2 = kwargs.get("elastic_width_lr_stage_2", 0.24)
        self.dataset_name = kwargs.get("dataset_name", "cifar10")
    
        