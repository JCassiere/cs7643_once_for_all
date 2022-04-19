import numpy as np
import torch

def train_loop(net, train_loader, test_loader, lr, epochs,
               depth_choices, kernel_choices, expansion_ratio_choices):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        for (idx, batch) in enumerate(train_loader):
            config = get_network_config(net.num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
            depths = config['depths']
            kernels = config['kernel_sizes']
            expansion_ratios = config['expansion_ratios']
        
            optimizer.zero_grad()
            images, targets = batch
            output = net.forward(images, depths, kernels, expansion_ratios)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        net.eval()
        test_correct = []
        with torch.no_grad():
            for (idx, batch) in enumerate(test_loader):
                config = get_network_config(net.num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
                depths = config['depths']
                kernels = config['kernel_sizes']
                expansion_ratios = config['expansion_ratios']
            
                images, targets = batch
                output = net.forward(images, depths, kernels, expansion_ratios)
                pred = torch.argmax(output, dim=1)
                test_correct.append((pred == targets).int())
            test_correct = torch.cat(test_correct, dim=-1)
            accuracy = torch.sum(test_correct) / test_correct.shape[0]
        print("Epoch {} accuracy: {}".format(epoch, accuracy))
    

def get_network_config(num_blocks, kernel_choices, depth_choices, expansion_ratio_choices):
    config = {"depths": [], "kernel_sizes": [], "expansion_ratios": []}
    for i in range(num_blocks):
        depth = np.random.choice(depth_choices)
        block_kernels = []
        block_expansion_ratios = []
        for j in range(depth):
            block_kernels.append(np.random.choice(kernel_choices))
            block_expansion_ratios.append(np.random.choice(expansion_ratio_choices))
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
    expansion_ratio_choices = [max_expansion_ratio]
    # big network training
    train_loop(net, train_loader, test_loader, lr=base_net_lr, epochs=base_net_epochs,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/big_network.pt')
        
    # elastic kernel
    kernel_choices = [7, 5, 3]
    train_loop(net, train_loader, test_loader, lr=elastic_kernel_lr, epochs=elastic_kernel_epochs,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices)
    
    # elastic depth stage 1
    depth_choices = [max_depth]
    if max_depth - 1 > 0:
        depth_choices.append(max_depth - 1)
    # kernel_choices = [7, 5, 3]
    train_loop(net, train_loader, test_loader, lr=elastic_depth_lr_stage_1, epochs=elastic_depth_epochs_stage_1,
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
    train_loop(net, train_loader, test_loader, lr=elastic_depth_lr_stage_2, epochs=elastic_depth_epochs_stage_2,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/elastic_depth_stage2.pt')
        
    # elastic width stage 1
    net.reorder_channels()
    depth_choices = [max_depth]
    for i in range(1, 3):
        if max_depth - i > 0:
            depth_choices.append(max_depth - i)
        else:
            break
    expansion_ratio_choices = [max_expansion_ratio]
    if max_expansion_ratio - 2 > 0:
        expansion_ratio_choices.append(max_expansion_ratio - 2)
    
    # kernel_choices = [7, 5, 3]
    train_loop(net, train_loader, test_loader, lr=elastic_width_lr_stage_1, epochs=elastic_width_epochs_stage_1,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/elastic_width_stage1.pt')
        
    # elastic width stage 2
    net.reorder_channels()
    depth_choices = [max_depth]
    for i in range(1, 3):
        if max_depth - i > 0:
            depth_choices.append(max_depth - i)
        else:
            break
    expansion_ratio_choices = [max_expansion_ratio]
    for x in [2, 3]:
        if max_expansion_ratio - x > 0:
            expansion_ratio_choices.append(max_expansion_ratio - x)
        else:
            break
    
    # kernel_choices = [7, 5, 3]
    train_loop(net, train_loader, test_loader, lr=elastic_width_lr_stage_2, epochs=elastic_width_epochs_stage_2,
               depth_choices=depth_choices, kernel_choices=kernel_choices,
               expansion_ratio_choices=expansion_ratio_choices)
    torch.save(net.state_dict(), './checkpoint/elastic_width_stage2.pt')
