import numpy as np
import torch


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
    
def training_stage(net, optimizer, train_loader, test_loader, epochs,
                   depths, kernels, expansion_ratios):
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        for (idx, batch) in enumerate(train_loader):
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
                images, targets = batch
                output = net.forward(images, depths, kernels, expansion_ratios)
                pred = torch.argmax(output, dim=1)
                test_correct.append((pred == targets).int())
            test_correct = torch.cat(test_correct, dim=-1)
            accuracy = torch.sum(test_correct) / test_correct.shape[0]
        print("Epoch {} accuracy: {}".format(epoch, accuracy))
    
def progressive_shrinking(train_loader, test_loader, net, **kwargs):
    num_blocks = kwargs.get("num_blocks", 5)
    max_expansion_ratio = kwargs.get("max_expansion_ratio", 6)
    max_depth = kwargs.get("max_depth", 4)
    # batch_size = kwargs.get("batch_size", 128)
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
    
    optimizer = torch.optim.SGD(net.parameters(), lr=base_net_lr, weight_decay=1e-5, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    default_depths = [max_depth for _ in range(num_blocks)]
    default_kernels = [[7 for _ in range(4)] for _ in range(num_blocks)]
    default_expansion_ratios = [[max_expansion_ratio for _ in range(4)] for _ in range(num_blocks)]
    # big network training
    for epoch in range(base_net_epochs):
        net.train()
        for (idx, batch) in enumerate(train_loader):
            optimizer.zero_grad()
            images, targets = batch
            output = net.forward(images, default_depths, default_kernels, default_expansion_ratios)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        net.eval()
        test_correct = []
        with torch.no_grad():
            for (idx, batch) in enumerate(test_loader):
                images, targets = batch
                output = net.forward(images, default_depths, default_kernels, default_expansion_ratios)
                pred = torch.argmax(output, dim=1)
                test_correct.append((pred == targets).int())
            test_correct = torch.cat(test_correct, dim=-1)
            accuracy = torch.sum(test_correct) / test_correct.shape[0]
        print("Epoch {} accuracy: {}".format(epoch, accuracy))
        
    # elastic kernel
    optimizer = torch.optim.SGD(net.parameters(), lr=elastic_kernel_lr, weight_decay=1e-5, momentum=0.9)
    for epoch in range(elastic_kernel_epochs):
        net.train()
        for (idx, batch) in enumerate(train_loader):
            config = get_network_config(num_blocks, [7, 5, 3], [max_depth], [max_expansion_ratio])
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
                config = get_network_config(num_blocks, [7, 5, 3], [max_depth], [max_expansion_ratio])
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

    # elastic depth stage 1
    optimizer = torch.optim.SGD(net.parameters(), lr=elastic_depth_lr_stage_1, weight_decay=1e-5, momentum=0.9)
    for epoch in range(elastic_depth_epochs_stage_1):
        depth_choices = [max_depth]
        if max_depth - 1 > 0:
            depth_choices.append(max_depth - 1)
        net.train()
        for (idx, batch) in enumerate(train_loader):
            config = get_network_config(num_blocks, [7, 5, 3], depth_choices, [max_expansion_ratio])
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
                config = get_network_config(num_blocks, [7, 5, 3], depth_choices, [max_expansion_ratio])
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

    # elastic depth stage 2
    optimizer = torch.optim.SGD(net.parameters(), lr=elastic_depth_lr_stage_2, weight_decay=1e-5, momentum=0.9)
    for epoch in range(elastic_depth_epochs_stage_2):
        depth_choices = [max_depth]
        for i in range(1, 3):
            if max_depth - i > 0:
                depth_choices.append(max_depth - i)
            else:
                break
        net.train()
        for (idx, batch) in enumerate(train_loader):
            config = get_network_config(
                num_blocks, [7, 5, 3],
                depth_choices,
                [max_expansion_ratio]
            )
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
                config = get_network_config(
                    num_blocks, [7, 5, 3],
                    depth_choices,
                    [max_expansion_ratio]
                )
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
    
    # elastic width stage 1
    net.reorder_channels()
    optimizer = torch.optim.SGD(net.parameters(), lr=elastic_width_lr_stage_1, weight_decay=1e-5, momentum=0.9)
    for epoch in range(elastic_width_epochs_stage_1):
        depth_choices = [max_depth]
        for i in range(1, 3):
            if max_depth - i > 0:
                depth_choices.append(max_depth - i)
            else:
                break
        expansion_ratio_choices = [max_expansion_ratio]
        if max_expansion_ratio - 2 > 0:
            expansion_ratio_choices.append(max_expansion_ratio - 2)
        net.train()
        for (idx, batch) in enumerate(train_loader):
            config = get_network_config(
                num_blocks, [7, 5, 3],
                depth_choices,
                expansion_ratio_choices
            )
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
                config = get_network_config(
                    num_blocks,
                    [7, 5, 3],
                    depth_choices,
                    expansion_ratio_choices
                )
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
    
    # elastic width stage 2
    net.reorder_channels()
    optimizer = torch.optim.SGD(net.parameters(), lr=elastic_width_lr_stage_2, weight_decay=1e-5, momentum=0.9)
    for epoch in range(elastic_width_epochs_stage_2):
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
        net.train()
        for (idx, batch) in enumerate(train_loader):
            config = get_network_config(
                num_blocks,
                [7, 5, 3],
                depth_choices,
                expansion_ratio_choices
            )
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
                config = get_network_config(
                    num_blocks, [7, 5, 3],
                    depth_choices,
                    expansion_ratio_choices
                )
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
