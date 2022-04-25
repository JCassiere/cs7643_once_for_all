import torch
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import random
from ofa.experiment import Experiment


# https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
def add_weight_decay(net, alpha=3e-5):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': alpha}]


def cross_entropy_loss(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


# https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
def smooth_labels(targets, num_classes, alpha=0.1):
    one_hot = F.one_hot(targets, num_classes)
    soft_target = (1 - alpha) * one_hot + alpha / num_classes
    return soft_target
    
    
def get_network_config(num_blocks, kernel_choices, depth_choices, expansion_ratio_choices):
    config = {"depths": [], "kernel_sizes": [], "expansion_ratios": []}
    for i in range(num_blocks):
        depth = random.choice(depth_choices)
        block_kernels = []
        block_expansion_ratios = []
        for _ in range(depth):
            block_kernels.append(random.choice(kernel_choices))
            block_expansion_ratios.append(random.choice(expansion_ratio_choices))
        config["depths"].append(depth)
        config["kernel_sizes"].append(block_kernels)
        config["expansion_ratios"].append(block_expansion_ratios)
    
    return config


def eval_one_epoch(experiment: Experiment, epoch, test_loader, depth_choices,
                   kernel_choices, expansion_ratio_choices):
    net = experiment.net
    net.eval()
    high_level_configurations = itertools.product(kernel_choices, depth_choices, expansion_ratio_choices)
    
    current_epoch_val_accuracies = {}
    # TODO - add top5 accuracy alongside top1
    with torch.no_grad():
        # test using the same depth across blocks and same kernel and expansion ratio at each layer
        # do this for each possible combination of depth, kernel size, and expansion ratio
        # then take the average
        for config in high_level_configurations:
            test_correct = []
            num_blocks = net.num_blocks
            max_depth = net.max_depth
            kernels = [[config[0] for _ in range(max_depth)] for _ in range(num_blocks)]
            depths = [config[1] for _ in range(num_blocks)]
            expansion_ratios = [[config[2] for _ in range(max_depth)] for _ in range(num_blocks)]
            for (idx, batch) in enumerate(test_loader):
                images, targets = batch
                output = net.forward(images, depths, kernels, expansion_ratios)
                pred = torch.argmax(output, dim=1)
                test_correct.append((pred == targets).int())
            test_correct = torch.cat(test_correct, dim=-1)
            accuracy = torch.mean(torch.sum(test_correct) / test_correct.shape[0]).item()
            config_str = "K{}-D{}-ExR{}".format(config[0], config[1], config[2])
            print("Epoch {} {} accuracy: {}".format(epoch, config_str, accuracy))
            current_epoch_val_accuracies[config_str] = accuracy
    experiment.append_val_accuracies(current_epoch_val_accuracies)


def train_loop(experiment: Experiment, lr, epochs,
               depth_choices, kernel_choices, expansion_ratio_choices,
               teacher=None, weight_decay=3e-5, num_subnetworks_per_minibatch=1,
               eval_first=True):
    net = experiment.net
    train_loader = experiment.train_data_loader
    test_loader = experiment.val_data_loader
    params = add_weight_decay(net, alpha=weight_decay)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    steps_per_epoch = len(train_loader)
    max_iterations = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)
    
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = cross_entropy_loss
    if eval_first:
        eval_one_epoch(experiment, -1, test_loader, depth_choices,
                       kernel_choices, expansion_ratio_choices)
    for epoch in range(epochs):
        # TODO - add top5 accuracy alongside top1
        with tqdm(total=steps_per_epoch,
                  desc="Train Epoch #{}".format(epoch)) as t:
            net.train()
            train_accuracies = []
            for (idx, batch) in enumerate(train_loader):
                optimizer.zero_grad()
                for _ in range(num_subnetworks_per_minibatch):
                    # seeding code taken from original repo
                    subnet_seed = int('%d%.3d%.3d' % (epoch * steps_per_epoch + idx, _, 0))
                    random.seed(subnet_seed)
                    
                    config = get_network_config(net.num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
                    depths = config['depths']
                    kernels = config['kernel_sizes']
                    expansion_ratios = config['expansion_ratios']
                    
                    images, targets = batch
                    output = net.forward(images, depths, kernels, expansion_ratios)
                    if teacher:
                        with torch.no_grad():
                            teacher_pred = teacher.forward(images).detach()
                        soft_label = F.softmax(teacher_pred, dim=1)
                        
                        dist_loss = criterion(output, soft_label)
                        smoothed_labels = smooth_labels(targets, output.size(1))
                        student_loss = criterion(output, smoothed_labels)
                        loss = dist_loss + student_loss
                    else:
                        smoothed_labels = smooth_labels(targets, output.size(1))
                        loss = criterion(output, smoothed_labels)
                    pred = torch.argmax(output, dim=1)
                    correct = (pred == targets).int()
                    train_accuracies.append(torch.sum(correct) / correct.shape[0])
                    
                    loss.backward()
                optimizer.step()
                scheduler.step()
                current_epoch_train_acc = torch.mean(torch.tensor(train_accuracies)).item()
                t.set_postfix({
                    'accuracy': current_epoch_train_acc
                })
                t.update(1)
                
        experiment.append_train_accuracy(current_epoch_train_acc)
        
        eval_one_epoch(experiment, epoch, test_loader, depth_choices,
                       kernel_choices, expansion_ratio_choices)


def train_big_network(experiment: Experiment):
    kernel_choices = experiment.overall_kernel_choices[:1]
    depth_choices = experiment.overall_depth_choices[:1]
    expansion_ratio_choices = experiment.overall_expansion_ratio_choices[:1]
    train_loop(
        experiment,
        lr=experiment.base_net_lr,
        epochs=experiment.base_net_epochs,
        depth_choices=depth_choices,
        kernel_choices=kernel_choices,
        expansion_ratio_choices=expansion_ratio_choices,
        eval_first=False,
        weight_decay=3e-4
    )
    experiment.log(stage="big_network")
    experiment.set_teacher()
    
    
def train_elastic_kernel(experiment: Experiment, load_stage=None):
    if load_stage:
        experiment.load_net_post_stage(load_stage)
    kernel_choices = experiment.overall_kernel_choices[:]
    depth_choices = experiment.overall_depth_choices[:1]
    expansion_ratio_choices = experiment.overall_expansion_ratio_choices[:1]
    train_loop(
        experiment,
        lr=experiment.elastic_kernel_lr,
        epochs=experiment.elastic_kernel_epochs,
        depth_choices=depth_choices,
        kernel_choices=kernel_choices,
        expansion_ratio_choices=expansion_ratio_choices,
        weight_decay=3e-5,
        teacher=experiment.get_teacher()
    )
    experiment.log(stage="elastic_kernel")


def train_elastic_depth_stage_1(experiment: Experiment, load_stage=None):
    if load_stage:
        experiment.load_net_post_stage(load_stage)
    kernel_choices = experiment.overall_kernel_choices[:]
    depth_choices = experiment.overall_depth_choices[:2]
    expansion_ratio_choices = experiment.overall_expansion_ratio_choices[:1]
    train_loop(
        experiment,
        lr=experiment.elastic_depth_lr_stage_1,
        epochs=experiment.elastic_depth_epochs_stage_1,
        depth_choices=depth_choices,
        kernel_choices=kernel_choices,
        expansion_ratio_choices=expansion_ratio_choices,
        weight_decay=3e-5,
        teacher=experiment.get_teacher(),
        num_subnetworks_per_minibatch=2
    )
    experiment.log(stage="elastic_depth_stage_1")


def train_elastic_depth_stage_2(experiment: Experiment, load_stage=None):
    if load_stage:
        experiment.load_net_post_stage(load_stage)
    kernel_choices = experiment.overall_kernel_choices[:]
    depth_choices = experiment.overall_depth_choices[:]
    expansion_ratio_choices = experiment.overall_expansion_ratio_choices[:1]
    train_loop(
        experiment,
        lr=experiment.elastic_depth_lr_stage_2,
        epochs=experiment.elastic_depth_epochs_stage_2,
        depth_choices=depth_choices,
        kernel_choices=kernel_choices,
        expansion_ratio_choices=expansion_ratio_choices,
        weight_decay=3e-5,
        teacher=experiment.get_teacher(),
        num_subnetworks_per_minibatch=2
    )
    experiment.log(stage="elastic_depth_stage_2")


def train_elastic_width_stage_1(experiment: Experiment, load_stage=None):
    if load_stage:
        experiment.load_net_post_stage(load_stage)
    kernel_choices = experiment.overall_kernel_choices[:]
    depth_choices = experiment.overall_depth_choices[:]
    expansion_ratio_choices = experiment.overall_expansion_ratio_choices[:2]
    train_loop(
        experiment,
        lr=experiment.elastic_width_lr_stage_1,
        epochs=experiment.elastic_width_epochs_stage_1,
        depth_choices=depth_choices,
        kernel_choices=kernel_choices,
        expansion_ratio_choices=expansion_ratio_choices,
        weight_decay=3e-5,
        teacher=experiment.get_teacher(),
        num_subnetworks_per_minibatch=4
    )
    experiment.log(stage="elastic_width_stage_1")


def train_elastic_width_stage_2(experiment: Experiment, load_stage=None):
    if load_stage:
        experiment.load_net_post_stage(load_stage)
    kernel_choices = experiment.overall_kernel_choices[:]
    depth_choices = experiment.overall_depth_choices[:]
    expansion_ratio_choices = experiment.overall_expansion_ratio_choices[:]
    train_loop(
        experiment,
        lr=experiment.elastic_width_lr_stage_2,
        epochs=experiment.elastic_width_epochs_stage_2,
        depth_choices=depth_choices,
        kernel_choices=kernel_choices,
        expansion_ratio_choices=expansion_ratio_choices,
        weight_decay=3e-5,
        teacher=experiment.get_teacher(),
        num_subnetworks_per_minibatch=4
    )
    experiment.log(stage="elastic_width_stage_2")

    
def progressive_shrinking_from_scratch(experiment: Experiment):
    train_big_network(experiment)
    train_elastic_kernel(experiment)
    train_elastic_depth_stage_1(experiment)
    train_elastic_depth_stage_2(experiment)
    train_elastic_width_stage_1(experiment)
    train_elastic_width_stage_2(experiment)
