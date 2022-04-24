import numpy as np
import torch
import copy
import torch.nn.functional as F
import itertools
from tqdm import tqdm
# from torchviz import make_dot

# https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
def add_weight_decay(net, l2_value=3e-5, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def cross_entropy_loss(pred, target):
    # logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


# https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
def smooth_labels(targets, num_classes, alpha=0.1):
    # soft_labels = torch.argmax(F.softmax(teacher_output, dim=1), dim=1)
    # num_classes = teacher_output.size(1)
    one_hot = F.one_hot(targets, num_classes)
    # batch_size = target.size(0)
    # target = torch.unsqueeze(target, 1)
    # soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    # soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = (1 - alpha) * one_hot + alpha / num_classes
    return soft_target

# def distillation_loss(teacher_output, student_output):
#     return torch.mean(torch.sum(-F.softmax(teacher_output, dim=1) * F.log_softmax(student_output, dim=1), 1))

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
               scheduler=True):
    # TODO - no weight decay on biases or batch norm
    params = add_weight_decay(net)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=3e-5, momentum=0.9, nesterov=True)
    steps_per_epoch = len(train_loader)
    max_iterations = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)
    
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = cross_entropy_loss
    for epoch in range(epochs):
        # TODO - include training accuracy and potentially top1 and top5
        #
        with tqdm(total=len(train_loader),
                  desc="Train Epoch #{}".format(epoch)) as t:
            net.train()
            for (idx, batch) in enumerate(train_loader):
                config = get_network_config(net.num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
                depths = config['depths']
                kernels = config['kernel_sizes']
                expansion_ratios = config['expansion_ratios']
                
                optimizer.zero_grad()
                images, targets = batch
                output = net.forward(images, depths, kernels, expansion_ratios)
                smoothed_labels = smooth_labels(targets, output.size(1))
                loss = criterion(output, smoothed_labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.update(1)
            eval_one_epoch(net, epoch, test_loader, depth_choices,
                           kernel_choices, expansion_ratio_choices)


def train_loop_with_distillation(net, teacher, train_loader, test_loader, lr, epochs,
                                 depth_choices, kernel_choices, expansion_ratio_choices,
                                 num_subs_to_sample=1):
    params = add_weight_decay(net)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=3e-5, momentum=0.9, nesterov=True)
    steps_per_epoch = len(train_loader)
    max_iterations = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = cross_entropy_loss
    teacher.eval()
    eval_one_epoch(net, -1, test_loader, depth_choices,
                   kernel_choices, expansion_ratio_choices)
    # TODO - track training error
    for epoch in range(epochs):
        net.train()
        # print(net.blocks[0].layers[0].depthwise_convolution.base_conv.weight[0][0])
        with tqdm(total=len(train_loader),
                  desc="Train Epoch #{}".format(epoch)) as t:
            # if epoch > 0:
            for (idx, batch) in enumerate(train_loader):
                for _ in range(num_subs_to_sample):
                    config = get_network_config(net.num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
                    depths = config['depths']
                    kernels = config['kernel_sizes']
                    expansion_ratios = config['expansion_ratios']
                    
                    optimizer.zero_grad()
                    images, targets = batch
                    output = net.forward(images, depths, kernels, expansion_ratios)
                    with torch.no_grad():
                        teacher_pred = teacher.forward(images).detach()
                    soft_label = F.softmax(teacher_pred, dim=1)
                    
                    dist_loss = criterion(output, soft_label)
                    smoothed_labels = smooth_labels(targets, output.size(1))
                    student_loss = criterion(output, smoothed_labels)
                    # loss = 0.1 * dist_loss + 0.9 * student_loss
                    loss = dist_loss + student_loss
                    # make_dot(loss, params=dict(net.named_parameters())).render("ofa_torchviz", format="png")
                    
                    loss.backward()
                optimizer.step()
                scheduler.step()
                t.update(1)
            eval_one_epoch(net, epoch, test_loader, depth_choices,
                           kernel_choices, expansion_ratio_choices)


def get_network_config(num_blocks, kernel_choices, depth_choices, expansion_ratio_choices):
    config = {"depths": [], "kernel_sizes": [], "expansion_ratios": []}
    for i in range(num_blocks):
        depth = np.random.choice(depth_choices)
        block_kernels = list(np.random.choice(kernel_choices, depth))
        block_expansion_ratios = list(np.random.choice(expansion_ratio_choices, depth))
        config["depths"].append(depth)
        config["kernel_sizes"].append(block_kernels)
        config["expansion_ratios"].append(block_expansion_ratios)
    
    return config


def progressive_shrinking(train_loader, test_loader, net, **kwargs):
    max_expansion_ratio = kwargs.get("max_expansion_ratio", 6)
    max_depth = kwargs.get("max_depth", 4)
    base_net_epochs = kwargs.get("base_net_epochs", 180)
    base_net_lr = kwargs.get("base_net_lr", 2.6)
    elastic_kernel_epochs = kwargs.get("elastic_kernel_epochs", 125)
    elastic_kernel_lr = kwargs.get("elastic_kernel_lr", 0.96)
    elastic_depth_epochs_stage_1 = kwargs.get("elastic_depth_epochs_stage_1", 25)
    elastic_depth_lr_stage_1 = kwargs.get("elastic_depth_lr_stage_1", 0.08)
    elastic_depth_epochs_stage_2 = kwargs.get("elastic_depth_epochs_stage_2", 125)
    elastic_depth_lr_stage_2 = kwargs.get("elastic_depth_lr_stage_2", 0.24)
    elastic_width_epochs_stage_1 = kwargs.get("elastic_width_epochs_stage_1", 25)
    elastic_width_lr_stage_1 = kwargs.get("elastic_width_lr_stage_1", 0.08)
    elastic_width_epochs_stage_2 = kwargs.get("elastic_width_epochs_stage_2", 125)
    elastic_width_lr_stage_2 = kwargs.get("elastic_width_lr_stage_2", 0.24)
    
    depth_choices = [max_depth]
    kernel_choices = [7]
    # kernel_choices = [5]
    expansion_ratio_choices = [max_expansion_ratio]
    
    # # warm-up
    # train_loop(net, train_loader, test_loader, lr=0.4, epochs=10,
    #            depth_choices=depth_choices, kernel_choices=kernel_choices,
    #            expansion_ratio_choices=expansion_ratio_choices, scheduler=False)
    # big network training
    train_loop(net, train_loader, test_loader, lr=base_net_lr, epochs=base_net_epochs,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/big_network.pt')
    
    teacher = copy.deepcopy(net)
    # elastic kernel
    kernel_choices = [3, 5, 7]
    # kernel_choices = [3, 5]
    train_loop_with_distillation(net, teacher, train_loader, test_loader, lr=elastic_kernel_lr,
                                 epochs=elastic_kernel_epochs, depth_choices=depth_choices,
                                 kernel_choices=kernel_choices, expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/elastic_kernel.pt')
    
    # TODO - sample 2 networks at each step and aggregate gradients
    # elastic depth stage 1
    depth_choices = [max_depth]
    if max_depth - 1 > 0:
        depth_choices.append(max_depth - 1)
    # kernel_choices = [7, 5, 3]
    train_loop_with_distillation(net, teacher, train_loader, test_loader, lr=elastic_depth_lr_stage_1,
                                 epochs=elastic_depth_epochs_stage_1,
                                 depth_choices=depth_choices, kernel_choices=kernel_choices,
                                 expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/elastic_depth_stage1.pt')
    
    # elastic depth stage 2
    depth_choices = [max_depth]
    for i in range(1, 3):
        if max_depth - i > 0:
            depth_choices.append(max_depth - i)
        else:
            break
    # kernel_choices = [7, 5, 3]
    train_loop_with_distillation(net, teacher, train_loader, test_loader, lr=elastic_depth_lr_stage_2,
                                 epochs=elastic_depth_epochs_stage_2,
                                 depth_choices=depth_choices, kernel_choices=kernel_choices,
                                 expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/elastic_depth_stage2.pt')
    
    # TODO - sample 4 networks at each step and aggregate gradients
    # elastic width stage 1
    net.reorder_channels()
    expansion_ratio_choices = [max_expansion_ratio]
    if max_expansion_ratio - 2 > 0:
        expansion_ratio_choices.append(max_expansion_ratio - 2)
    
    # kernel_choices = [7, 5, 3]
    train_loop_with_distillation(net, teacher, train_loader, test_loader, lr=elastic_width_lr_stage_1,
                                 epochs=elastic_width_epochs_stage_1,
                                 depth_choices=depth_choices, kernel_choices=kernel_choices,
                                 expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/elastic_width_stage1.pt')
    
    # elastic width stage 2
    net.reorder_channels()
    expansion_ratio_choices = [max_expansion_ratio]
    for x in [2, 3]:
        if max_expansion_ratio - x > 0:
            expansion_ratio_choices.append(max_expansion_ratio - x)
        else:
            break
    
    train_loop_with_distillation(net, teacher, train_loader, test_loader, lr=elastic_width_lr_stage_2,
                                 epochs=elastic_width_epochs_stage_2,
                                 depth_choices=depth_choices, kernel_choices=kernel_choices,
                                 expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/elastic_width_stage2.pt')
