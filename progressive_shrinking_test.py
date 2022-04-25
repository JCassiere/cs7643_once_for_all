import copy

import torch.nn
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
# from torchviz import make_dot
from ofa.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from ofa.mobilenetv3_ofa import mobilenetv3_ofa
from ofa.progressive_shrinking import progressive_shrinking_from_scratch
from ofa.progressive_shrinking import train_loop
import cProfile


# def get_graph():
#     device = get_device()
#     train_data_loader, _ = get_cifar_dataloaders(device)
#     net = large_test_ofa_net()
#     for (i, batch) in enumerate(train_data_loader):
#         images, targets = batch
#         out = net.forward(images)
#         make_dot(out, params=dict(net.named_parameters())).render("ofa_torchviz", format="png")
#         break


def get_num_params(net):
    param_count = sum([torch.numel(p) for p in net.parameters()])
    return param_count


def large_test_ofa_net(num_classes=100):
    # default ofa (as seen in paper)
    
    # output_widths = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
    # use_squeeze_excites = [False, False, True, False, True, True, True, True, True]
    # use_hard_swishes = [True, False, False, True, True, True, True, True, True]
    # strides = [1, 1, 2, 2, 1, 2, 1, 1, 1]
    net = mobilenetv3_ofa(num_classes=num_classes)
    return net




def small_test_ofa_net():
    output_widths = [16, 16, 24, 40, 112, 160, 960, 1280]
    use_squeeze_excites = [False, False, False, True, True, True, False, False]
    use_hard_swishes = [True, False, False, False, True, True, True, True]
    # strides = [1, 1, 2, 2, 2, 1, 1]
    strides = [2, 1, 2, 2, 2, 2, 1]
    
    net = mobilenetv3_ofa(num_classes=100, output_widths=output_widths, use_squeeze_excites=use_squeeze_excites,
                          use_hard_swishes=use_hard_swishes, strides=strides, width_mult=0.75)
    return net




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
    train_loop(net, train_data_loader, test_data_loader, lr=0.03, epochs=125,
               depth_choices=[4], kernel_choices=[3, 5, 7],
               expansion_ratio_choices=[6], teacher=teacher)

def test_elastic_kernel_cifar10():
    device = get_device()
    train_data_loader, test_data_loader = get_cifar_dataloaders(device, cifar_dataset=torchvision.datasets.CIFAR10)
    
    # net = small_test_ofa_net()
    net = large_test_ofa_net(num_classes=10)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load("checkpoint/big_network.pt"))
    else:
        net.load_state_dict(torch.load("checkpoint/big_network.pt", map_location=torch.device('cpu')))
    net.to(device)
    teacher = copy.deepcopy(net)
    print("in elastic kernel")
    train_loop(net, train_data_loader, test_data_loader, lr=0.009, epochs=125,
               depth_choices=[4], kernel_choices=[3, 5],
               expansion_ratio_choices=[6], teacher=teacher)

def test_elastic_depth():
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
    print("in elastic depth")
    train_loop(net, train_data_loader, test_data_loader, lr=0.0008, epochs=1,
                                 depth_choices=[3, 4], kernel_choices=[7],
                                 expansion_ratio_choices=[6], teacher=teacher, num_subs_to_sample=2)
    
    train_loop(net, train_data_loader, test_data_loader, lr=0.0024, epochs=20,
                                 depth_choices=[2, 3, 4], kernel_choices=[7],
                                 expansion_ratio_choices=[6], teacher=teacher, num_subs_to_sample=2)


def test_elastic_width_cifar10():
    device = get_device()
    train_data_loader, test_data_loader = get_cifar_dataloaders(device, cifar_dataset=torchvision.datasets.CIFAR10)
    
    # net = small_test_ofa_net()
    net = mobilenetv3_ofa(num_classes=10, max_kernel_size=5)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load("checkpoint/big_network.pt"))
    else:
        net.load_state_dict(torch.load("checkpoint/big_network.pt", map_location=torch.device('cpu')))
    net.to(device)
    teacher = copy.deepcopy(net)
    print("in elastic kernel")
    train_loop(net, train_data_loader, test_data_loader, lr=0.0008, epochs=15,
               depth_choices=[4], kernel_choices=[5],
               expansion_ratio_choices=[6, 4], teacher=teacher)
    train_loop(net, train_data_loader, test_data_loader, lr=0.0024, epochs=125,
               depth_choices=[4], kernel_choices=[5],
               expansion_ratio_choices=[6, 4, 3], teacher=teacher)
    
    
def test_progressive_shrinking_cifar10():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_cifar_dataloaders(device, cifar_dataset=torchvision.datasets.CIFAR10)
    
    # net = large_test_ofa_net(num_classes=10)
    net = mobilenetv3_ofa(num_classes=10, max_kernel_size=5)
    
    net.to(device)
    progressive_shrinking_from_scratch(train_data_loader, test_data_loader, net, base_net_lr=.026,
                                       base_net_epochs=15, elastic_kernel_epochs=0, elastic_depth_epochs_stage_2=50,
                                       elastic_width_epochs_stage_2=50,
                                       elastic_kernel_lr=.009, elastic_depth_lr_stage_1=.0008,
                                       elastic_depth_lr_stage_2=.0024, elastic_width_lr_stage_1=.0008,
                                       elastic_width_lr_stage_2=.0024)
    
    # Probably the best settings for a final good run
    # progressive_shrinking(train_data_loader, test_data_loader, net, base_net_lr=.026,
    #                       elastic_kernel_lr=.0096, elastic_depth_lr_stage_1=.0008,
    #                       elastic_depth_lr_stage_2=.0024, elastic_width_lr_stage_1=.0008,
    #                       elastic_width_lr_stage_2=.0024)

def test_progressive_shrinking():
    exp_kwargs = {
        "dataset_name": "cifar10",
        "experiment_name": "cifar10_100_epochs_per_stage_{}".format(int(time.time())),
        "base_net_epochs": 100,
        "elastic_kernel_epochs": 100,
        "elastic_depth_epochs_stage_1": 25,
        "elastic_depth_epochs_stage_2": 100,
        "elastic_width_epochs_stage_1": 25,
        "elastic_width_epochs_stage_2": 100
    }
    exp_kwargs = {
        "dataset_name": "mnist",
        "experiment_name": "mnist_100_epochs_per_stage_{}".format(int(time.time())),
        "base_net_epochs": 100,
        "elastic_kernel_epochs": 100,
        "elastic_depth_epochs_stage_1": 25,
        "elastic_depth_epochs_stage_2": 100,
        "elastic_width_epochs_stage_1": 25,
        "elastic_width_epochs_stage_2": 100
    }
    experiment = Experiment(**exp_kwargs)
    progressive_shrinking_from_scratch(experiment)


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

# TODO - kernels 3 and 5 learning better than 7

# TODO - initialization like the original?
if __name__ == "__main__":
    # train_mobilenetv3_cifar100()
    # cProfile.run('train_mobileZDnetv3ofa_cifar100()', 'profile')
    # train_mobilenetv3ofa_cifar100()
    # learn_on_small_kernel_only()
    
    # test_elastic_kernel()
    # test_elastic_kernel_cifar10()
    # test_elastic_depth()
    test_elastic_width_cifar10()
    # train_smallest_network()
    # train_on_small_kernel_only()
    # 1 epoch/30s on colab
    # test_progressive_shrinking()
    # test_progressive_shrinking_cifar10()
    # get_graph()
