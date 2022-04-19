import torch.nn
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from ofa.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from ofa.mobilenetv3_ofa import mobilenetv3_ofa
from ofa.progressive_shrinking import progressive_shrinking
from ofa.progressive_shrinking import train_loop
import cProfile

def get_num_params(optimizer):
    params = optimizer.param_groups[0]['params']
    param_count = sum([torch.numel(p) for p in params])
    return param_count
    
def get_cifar_dataloaders(device, subset=False):
    def collate_fn_to_device(batch):
        # adapted from pytorch default_collate_fn
        transposed = list(zip(*batch))
        images = torch.stack(transposed[0], 0).to(device)
        targets = torch.tensor(transposed[1]).to(device)
        
        return images, targets
    
    cifar_train = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            # this performs random shifts by up to 4 pixels
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]))
    if subset:
        indices = torch.arange(6000)
        cifar_train = data_utils.Subset(cifar_train, indices)
    
    cifar_test = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]))
    if subset:
        indices = torch.arange(1000)
        cifar_test = data_utils.Subset(cifar_test, indices)
    train_data_loader = DataLoader(cifar_train, batch_size=128, shuffle=True,
                                   collate_fn=collate_fn_to_device)
    test_data_loader = DataLoader(cifar_test, batch_size=128, shuffle=True,
                                  collate_fn=collate_fn_to_device)
    return train_data_loader, test_data_loader
    
def train_mobilenetv3_cifar100():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_cifar_dataloaders(device)
    criterion = torch.nn.CrossEntropyLoss()
    net = mobilenetv3_small(num_classes=100)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9)
    
    for epoch in range(100):
        net.train()
        for (idx, batch) in enumerate(train_data_loader):
            optimizer.zero_grad()
            images, targets = batch
            output = net.forward(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        net.eval()
        test_correct = []
        with torch.no_grad():
            for (idx, batch) in enumerate(test_data_loader):
                images, targets = batch
                output = net.forward(images)
                pred = torch.argmax(output, dim=1)
                test_correct.append((pred == targets).int())
            test_correct = torch.cat(test_correct, dim=-1)
            accuracy = torch.sum(test_correct) / test_correct.shape[0]
        print("Epoch {} accuracy: {}".format(epoch, accuracy))

def train_mobilenetv3ofa_cifar100():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_cifar_dataloaders(device)
    criterion = torch.nn.CrossEntropyLoss()
    # output_widths = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
    # use_squeeze_excites = [False, False, True, False, True, True, True, True, True]
    # use_hard_swishes = [True, False, False, True, True, True, True, True, True]
    # strides = [1, 1, 1, 2, 1, 1, 1, 1, 1]
    # net = mobilenetv3_ofa(num_classes=100)

    output_widths = [16, 16, 24, 64, 118, 800, 960]
    use_squeeze_excites = [False, False, True, False, True, True, True]
    use_hard_swishes = [True, False, False, True, True, True, True]
    strides = [1, 1, 2, 2, 2, 1, 1]

    net = mobilenetv3_ofa(num_classes=100, output_widths=output_widths, use_squeeze_excites=use_squeeze_excites,
                          use_hard_swishes=use_hard_swishes, strides=strides, width_mult=0.25)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9)
    
    # default_depths = [4 for _ in range(5)]
    # default_kernels = [[7 for _ in range(4)] for _ in range(5)]
    # default_expansion_ratios = [[6 for _ in range(4)] for _ in range(5)]
    default_depths = [4 for _ in range(3)]
    default_kernels = [[7 for _ in range(4)] for _ in range(3)]
    default_expansion_ratios = [[6 for _ in range(4)] for _ in range(3)]
    for epoch in range(100):
        net.train()
        for (idx, batch) in enumerate(train_data_loader):
            optimizer.zero_grad()
            images, targets = batch
            # output = net.forward(images, base_configs)
            output = net.forward(images, default_depths, default_kernels, default_expansion_ratios)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            # scheduler.step()
        net.eval()
        test_correct = []
        with torch.no_grad():
            for (idx, batch) in enumerate(test_data_loader):
                images, targets = batch
                # output = net.forward(images, base_configs)
                output = net.forward(images, default_depths, default_kernels, default_expansion_ratios)
                pred = torch.argmax(output, dim=1)
                test_correct.append((pred == targets).int())
            test_correct = torch.cat(test_correct, dim=-1)
            accuracy = torch.sum(test_correct) / test_correct.shape[0]
        print("Epoch {} accuracy: {}".format(epoch, accuracy))

def learn_on_small_kernel_only():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_cifar_dataloaders(device, subset=True)
    output_widths = [16, 16, 24, 64, 118, 800, 960]
    use_squeeze_excites = [False, False, True, False, True, True, True]
    use_hard_swishes = [True, False, False, True, True, True, True]
    strides = [1, 1, 2, 2, 2, 1, 1]

    net = mobilenetv3_ofa(num_classes=100, output_widths=output_widths, use_squeeze_excites=use_squeeze_excites,
                          use_hard_swishes=use_hard_swishes, strides=strides, width_mult=0.25)
    net.to(device)
    print(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    default_depths = [4 for _ in range(3)]
    default_kernels = [[3 for _ in range(4)] for _ in range(3)]
    default_expansion_ratios = [[6 for _ in range(4)] for _ in range(3)]
    for epoch in range(100):
        net.train()
        for (idx, batch) in enumerate(train_data_loader):
            optimizer.zero_grad()
            images, targets = batch
            # output = net.forward(images, base_configs)
            output = net.forward(images, default_depths, default_kernels, default_expansion_ratios)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            # scheduler.step()
        net.eval()
        test_correct = []
        with torch.no_grad():
            for (idx, batch) in enumerate(test_data_loader):
                images, targets = batch
                # output = net.forward(images, base_configs)
                output = net.forward(images, default_depths, default_kernels, default_expansion_ratios)
                pred = torch.argmax(output, dim=1)
                test_correct.append((pred == targets).int())
            test_correct = torch.cat(test_correct, dim=-1)
            accuracy = torch.sum(test_correct) / test_correct.shape[0]
        print("Epoch {} accuracy: {}".format(epoch, accuracy))
    
# def test_elastic_kernel():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     train_data_loader, test_data_loader = get_cifar_dataloaders(device)
#
#     output_widths = [16, 16, 24, 64, 118, 800, 960]
#     use_squeeze_excites = [False, False, True, False, True, True, True]
#     use_hard_swishes = [True, False, False, True, True, True, True]
#     strides = [1, 1, 2, 2, 2, 1, 1]
#
#     net = mobilenetv3_ofa(num_classes=100, output_widths=output_widths, use_squeeze_excites=use_squeeze_excites,
#                           use_hard_swishes=use_hard_swishes, strides=strides, width_mult=0.25)
#     default_depths = [max_depth for _ in range(num_blocks)]
#     default_kernels = [[7 for _ in range(4)] for _ in range(num_blocks)]
#     default_expansion_ratios = [[max_expansion_ratio for _ in range(4)] for _ in range(num_blocks)]
#
#     train_loop(net, train_data_loader, test_loader, lr=elastic_kernel_lr, epochs=elastic_kernel_epochs,
#                num_blocks=num_blocks, depth_choices=default_depths, kernel_choices=[7, 5, 3],
#                expansion_ratio_choices=default_expansion_ratios)

def test_progressive_shrinking():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_cifar_dataloaders(device)
    # output_widths = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
    # use_squeeze_excites = [False, False, True, False, True, True, True, True, True]
    # use_hard_swishes = [True, False, False, True, True, True, True, True, True]
    # strides = [1, 1, 1, 2, 1, 1, 1, 1, 1]
    # net = mobilenetv3_ofa(num_classes=100)
    
    output_widths = [16, 16, 24, 64, 118, 800, 960]
    use_squeeze_excites = [False, False, True, False, True, True, True]
    use_hard_swishes = [True, False, False, True, True, True, True]
    strides = [1, 1, 2, 2, 2, 1, 1]

    net = mobilenetv3_ofa(num_classes=100, output_widths=output_widths, use_squeeze_excites=use_squeeze_excites,
                          use_hard_swishes=use_hard_swishes, strides=strides, width_mult=0.25)
    net.to(device)
    progressive_shrinking(train_data_loader, test_data_loader, net, base_net_epochs=25,
                          elastic_kernel_epochs=25, elastic_depth_epochs_stage_1=5,
                          elastic_depth_epochs_stage_2=25, elastic_width_epochs_stage_1=5,
                          elastic_width_epochs_stage_2=25, base_net_lr=0.24, elastic_kernel_lr=0.14,
                          elastic_depth_lr_stage_1=0.05, elastic_depth_lr_stage_2=0.1,
                          elastic_width_lr_stage_1=0.05, elastic_width_lr_stage_2=0.1,)
    # progressive_shrinking(train_data_loader, test_data_loader, net, base_net_lr=0.24, elastic_kernel_lr=0.20,
    #                       elastic_depth_lr_stage_1=0.08, elastic_depth_lr_stage_2=0.16,
    #                       elastic_width_lr_stage_1=0.08, elastic_width_lr_stage_2=0.16)

# TODO - try learning just with elastic depth and/or elastic width
#  I think there is a bug with the elastic kernel
if __name__ == "__main__":
    # train_mobilenetv3_cifar100()
    # cProfile.run('train_mobilenetv3ofa_cifar100()', 'profile')
    # train_mobilenetv3ofa_cifar100()
    # learn_on_small_kernel_only()
    
    # 1 epoch/30s on colab
    test_progressive_shrinking()
    
