import copy

import torch.nn
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from ofa.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from ofa.mobilenetv3_ofa import mobilenetv3_ofa
from ofa.progressive_shrinking import progressive_shrinking
from ofa.progressive_shrinking import train_loop, train_loop_with_distillation
import cProfile


def get_num_params(net):
    param_count = sum([torch.numel(p) for p in net.parameters()])
    return param_count


def large_test_ofa_net():
    # default ofa (as seen in paper)
    
    # output_widths = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
    # use_squeeze_excites = [False, False, True, False, True, True, True, True, True]
    # use_hard_swishes = [True, False, False, True, True, True, True, True, True]
    # strides = [1, 1, 2, 2, 1, 2, 1, 1, 1]
    net = mobilenetv3_ofa(num_classes=100)
    return net


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def small_test_ofa_net():
    output_widths = [16, 16, 24, 40, 112, 160, 960, 1280]
    use_squeeze_excites = [False, False, False, True, True, True, False, False]
    use_hard_swishes = [True, False, False, False, True, True, True, True]
    # strides = [1, 1, 2, 2, 2, 1, 1]
    strides = [2, 1, 2, 2, 2, 2, 1]
    
    net = mobilenetv3_ofa(num_classes=100, output_widths=output_widths, use_squeeze_excites=use_squeeze_excites,
                          use_hard_swishes=use_hard_swishes, strides=strides, width_mult=0.75)
    return net


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
    device = get_device()
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


def train_mobilenetv3ofa_cifar100_no_shrinking():
    device = get_device()
    train_data_loader, test_data_loader = get_cifar_dataloaders(device)
    
    net = small_test_ofa_net()
    net.to(device)
    
    train_loop(net, train_data_loader, test_data_loader, lr=0.1, epochs=100,
               depth_choices=[4], kernel_choices=[7], expansion_ratio_choices=[6])


def train_on_small_kernel_only():
    device = get_device()
    train_data_loader, test_data_loader = get_cifar_dataloaders(device, subset=True)
    
    net = small_test_ofa_net()
    net.load_state_dict(torch.load("checkpoint/big_network.pt"))
    net.to(device)
    
    train_loop(net, train_data_loader, test_data_loader, lr=0.3, epochs=100,
               depth_choices=[4], kernel_choices=[3], expansion_ratio_choices=[6])


def test_elastic_kernel():
    device = get_device()
    train_data_loader, test_data_loader = get_cifar_dataloaders(device)
    
    # net = small_test_ofa_net()
    net = large_test_ofa_net()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load("checkpoint/big_network.pt"))
    else:
        net.load_state_dict(torch.load("checkpoint/big_network.pt", map_location=torch.device('cpu')))
    net.to(device)
    teacher = copy.deepcopy(net)
    print("in elastic kernel")
    train_loop_with_distillation(net, teacher, train_data_loader, test_data_loader, lr=0.00096, epochs=125,
                                 depth_choices=[4], kernel_choices=[3, 5, 7],
                                 expansion_ratio_choices=[6])


def test_progressive_shrinking():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_cifar_dataloaders(device)
    
    # net = small_test_ofa_net()
    net = large_test_ofa_net()
    # get_num_params(net)
    # params = [p for p in net.parameters()]
    net.to(device)
    # progressive_shrinking(train_data_loader, test_data_loader, net, base_net_epochs=50,
    #                       elastic_kernel_epochs=50, elastic_depth_epochs_stage_1=5,
    #                       elastic_depth_epochs_stage_2=50, elastic_width_epochs_stage_1=5,
    #                       elastic_width_epochs_stage_2=50, base_net_lr=0.96, elastic_kernel_lr=0.20,
    #                       elastic_depth_lr_stage_1=0.05, elastic_depth_lr_stage_2=0.1,
    #                       elastic_width_lr_stage_1=0.05, elastic_width_lr_stage_2=0.1,)
    # progressive_shrinking(train_data_loader, test_data_loader, net, base_net_epochs=25,
    #                       elastic_kernel_epochs=25, elastic_depth_epochs_stage_1=5,
    #                       elastic_depth_epochs_stage_2=25, elastic_width_epochs_stage_1=5,
    #                       elastic_width_epochs_stage_2=25, base_net_lr=0.24, elastic_kernel_lr=0.20,
    #                       elastic_depth_lr_stage_1=0.05, elastic_depth_lr_stage_2=0.1,
    #                       elastic_width_lr_stage_1=0.05, elastic_width_lr_stage_2=0.1, )
    # progressive_shrinking(train_data_loader, test_data_loader, net, base_net_lr=0.30, elastic_kernel_lr=0.26,
    #                       elastic_depth_lr_stage_1=0.08, elastic_depth_lr_stage_2=0.20,
    #                       elastic_width_lr_stage_1=0.08, elastic_width_lr_stage_2=0.20)
    progressive_shrinking(train_data_loader, test_data_loader, net, base_net_lr=.026,
                          base_net_epochs=65, elastic_kernel_epochs=50, elastic_depth_epochs_stage_2=50,
                          elastic_width_epochs_stage_2=50,
                          elastic_kernel_lr=.03, elastic_depth_lr_stage_1=.0008,
                          elastic_depth_lr_stage_2=.0024, elastic_width_lr_stage_1=.0008,
                          elastic_width_lr_stage_2=.0024)
    
    # Probably the best settings for a final good run
    # progressive_shrinking(train_data_loader, test_data_loader, net, base_net_lr=.026,
    #                       elastic_kernel_lr=.0096, elastic_depth_lr_stage_1=.0008,
    #                       elastic_depth_lr_stage_2=.0024, elastic_width_lr_stage_1=.0008,
    #                       elastic_width_lr_stage_2=.0024)
    
    
def train_smallest_network():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_cifar_dataloaders(device)
    
    net = small_test_ofa_net()
    net.to(device)
    train_loop(net, train_data_loader, test_data_loader, lr=0.3, epochs=100,
               depth_choices=[2], kernel_choices=[3], expansion_ratio_choices=[3])

def fine_tune_smallest_network():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_cifar_dataloaders(device)
    
    net = small_test_ofa_net()
    net.load_state_dict(torch.load("checkpoint/big_network.pt"))
    net.to(device)
    train_loop(net, train_data_loader, test_data_loader, lr=0.3, epochs=100,
               depth_choices=[2], kernel_choices=[3], expansion_ratio_choices=[3])


# TODO - something is still off. Accuracies should be much closer for the 3 different
#  kernel sizes, within 5 or so percentage points
# TODO - implement checkpointing system where best weights are checked and saved
#  after each epoch
if __name__ == "__main__":
    # train_mobilenetv3_cifar100()
    # cProfile.run('train_mobilenetv3ofa_cifar100()', 'profile')
    # train_mobilenetv3ofa_cifar100()
    # learn_on_small_kernel_only()
    
    # test_elastic_kernel()
    # train_smallest_network()
    # train_on_small_kernel_only()
    # 1 epoch/30s on colab
    test_progressive_shrinking()
