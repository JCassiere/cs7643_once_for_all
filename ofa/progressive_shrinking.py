import torch
import copy
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import random


# https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
def add_weight_decay(net, alpha=3e-5, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
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


def eval_one_epoch(net, epoch, test_loader, depth_choices,
                   kernel_choices, expansion_ratio_choices):
    net.eval()
    high_level_configurations = itertools.product(kernel_choices, depth_choices, expansion_ratio_choices)
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
            accuracy = torch.sum(test_correct) / test_correct.shape[0]
            config_str = "K{}-D{}-ExR{}".format(config[0], config[1], config[2])
            print("Epoch {} {} accuracy: {}".format(epoch, config_str, torch.mean(accuracy)))


def train_loop(net, train_loader, test_loader, lr, epochs,
               depth_choices, kernel_choices, expansion_ratio_choices,
               teacher=None, weight_decay=3e-5, num_subnetworks_per_minibatch=1):
    params = add_weight_decay(net, alpha=weight_decay)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    steps_per_epoch = len(train_loader)
    max_iterations = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)
    
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = cross_entropy_loss
    eval_one_epoch(net, -1, test_loader, depth_choices,
                   kernel_choices, expansion_ratio_choices)
    for epoch in range(epochs):
        # TODO - include training accuracy and potentially top1 and top5
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
                t.set_postfix({
                    'accuracy': torch.mean(torch.tensor(train_accuracies)).item()
                })
                t.update(1)
                
        eval_one_epoch(net, epoch, test_loader, depth_choices,
                       kernel_choices, expansion_ratio_choices)



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


def progressive_shrinking(train_loader, test_loader, net, experiment_name, **kwargs):
    base_net_epochs = kwargs.get("base_net_epochs", 180)
    base_net_lr = kwargs.get("base_net_lr", 2.6)
    overall_kernel_choices = kwargs.get("kernel_choices", [3, 5])
    elastic_kernel_epochs = kwargs.get("elastic_kernel_epochs", 125)
    elastic_kernel_lr = kwargs.get("elastic_kernel_lr", 0.96)
    overall_depth_choices = kwargs.get("depth_choices", [2, 3, 4])
    elastic_depth_epochs_stage_1 = kwargs.get("elastic_depth_epochs_stage_1", 25)
    elastic_depth_lr_stage_1 = kwargs.get("elastic_depth_lr_stage_1", 0.08)
    elastic_depth_epochs_stage_2 = kwargs.get("elastic_depth_epochs_stage_2", 125)
    elastic_depth_lr_stage_2 = kwargs.get("elastic_depth_lr_stage_2", 0.24)
    overall_expansion_ratio_choices = kwargs.get("expansion_ratio_choices", [3, 4, 6])
    elastic_width_epochs_stage_1 = kwargs.get("elastic_width_epochs_stage_1", 25)
    elastic_width_lr_stage_1 = kwargs.get("elastic_width_lr_stage_1", 0.08)
    elastic_width_epochs_stage_2 = kwargs.get("elastic_width_epochs_stage_2", 125)
    elastic_width_lr_stage_2 = kwargs.get("elastic_width_lr_stage_2", 0.24)
    
    overall_kernel_choices.sort(reverse=True)
    overall_depth_choices.sort(reverse=True)
    overall_expansion_ratio_choices.sort(reverse=True)
    kernel_choices = overall_kernel_choices[:1]
    depth_choices = overall_depth_choices[:1]
    expansion_ratio_choices = overall_expansion_ratio_choices[:1]
    
    directory = './checkpoint/' + experiment_name + '/'
    # big network training
    train_loop(net, train_loader, test_loader, lr=base_net_lr, epochs=base_net_epochs,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices, weight_decay=3e-4)
    torch.save(net.state_dict(), directory + 'big_network.pt')
    
    # elastic kernel
    teacher = copy.deepcopy(net)
    kernel_choices = overall_kernel_choices[:]
    train_loop(net, train_loader, test_loader, lr=elastic_kernel_lr,
               epochs=elastic_kernel_epochs, depth_choices=depth_choices,
               kernel_choices=kernel_choices, expansion_ratio_choices=expansion_ratio_choices,
               teacher=teacher)
    torch.save(net.state_dict(), directory + '/elastic_kernel.pt')
    
    # elastic depth stage 1
    # teacher = copy.deepcopy(net)
    depth_choices = overall_depth_choices[:2]
    train_loop(net, train_loader, test_loader, lr=elastic_depth_lr_stage_1,
               epochs=elastic_depth_epochs_stage_1,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices, teacher=teacher,
               num_subnetworks_per_minibatch=2)
    torch.save(net.state_dict(), directory + 'elastic_depth_stage1.pt')
    
    # elastic depth stage 2
    # TODO - update teacher after each stage?
    #  maybe check whether the original teacher or current net gives better results
    #  for the full network
    # teacher = copy.deepcopy(net)
    depth_choices = overall_depth_choices[:]
    train_loop(net, train_loader, test_loader, lr=elastic_depth_lr_stage_2,
               epochs=elastic_depth_epochs_stage_2,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices, teacher=teacher,
               num_subnetworks_per_minibatch=2)
    torch.save(net.state_dict(), directory + 'elastic_depth_stage2.pt')
    
    # elastic width stage 1
    # teacher = copy.deepcopy(net)
    net.reorder_channels()
    expansion_ratio_choices = overall_expansion_ratio_choices[:2]
    train_loop(net, train_loader, test_loader, lr=elastic_width_lr_stage_1,
               epochs=elastic_width_epochs_stage_1,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices, teacher=teacher,
               num_subnetworks_per_minibatch=4)
    torch.save(net.state_dict(), directory + 'elastic_width_stage1.pt')
    
    # elastic width stage 2
    # teacher = copy.deepcopy(net)
    net.reorder_channels()
    expansion_ratio_choices = overall_expansion_ratio_choices[:]
    train_loop(net, train_loader, test_loader, lr=elastic_width_lr_stage_2,
               epochs=elastic_width_epochs_stage_2,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices, teacher=teacher,
               num_subnetworks_per_minibatch=4)
    torch.save(net.state_dict(), directory + 'elastic_width_stage2.pt')
