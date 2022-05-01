import time
from ofa.experiment import Experiment
from ofa.progressive_shrinking import train_loop

def train_sub_network(experiment: Experiment):
    kernel_choices = experiment.overall_kernel_choices[1:2]
    depth_choices = experiment.overall_depth_choices[0:1]
    expansion_ratio_choices = experiment.overall_expansion_ratio_choices[2:3]
    train_loop(
        experiment,
        lr=experiment.base_net_lr,
        epochs=10,
        depth_choices=depth_choices,
        kernel_choices=kernel_choices,
        expansion_ratio_choices=expansion_ratio_choices,
        eval_first=False,
        weight_decay=3e-4
    )
    experiment.log(stage="sub_network")
    experiment.set_teacher()

exp_kwargs = {
        "dataset_name": "cifar10",
        "experiment_name": "sub_network_cifar10_{}".format(int(time.time()))
    }
experiment = Experiment(**exp_kwargs)
train_sub_network(experiment)
