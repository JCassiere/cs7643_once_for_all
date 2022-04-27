import time
import torch.nn
from ofa.mobilenetv3 import mobilenetv3_small
from ofa.progressive_shrinking import progressive_shrinking_from_scratch, train_big_network, train_elastic_kernel,\
    train_elastic_depth_stage_1, train_elastic_depth_stage_2, train_elastic_width_stage_1, train_elastic_width_stage_2
from ofa.utils import get_device
from ofa.experiment import Experiment, experiment_from_config
from ofa.datasets import get_dataloaders
    
def train_mobilenetv3_MNIST():
    device = get_device()
    train_data_loader, test_data_loader = get_dataloaders(device)
    criterion = torch.nn.CrossEntropyLoss()
    net = mobilenetv3_small(num_classes=10)
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

def train_mobilenetv3ofa_MNIST_no_shrinking():
    exp_kwargs = {
        "dataset_name": "mnist",
        "experiment_name": "big_net_only_MNIST_{}".format(int(time.time()))
    }
    experiment = Experiment(**exp_kwargs)
    train_big_network(experiment)
    
        
def test_elastic_kernel():
    # Note that this will only work if you have previously run an experiment called "my_experiment"
    # and the big_network was trained
    experiment_name = "my_experiment"
    experiment = experiment_from_config(experiment_name, load_stage="big_network")
    train_elastic_kernel(experiment)

def test_elastic_depth():
    # Note that this will only work if you have previously run an experiment called "my_experiment"
    # and the elastic_kernel stage was trained
    experiment_name = "my_experiment"
    experiment = experiment_from_config(experiment_name, load_stage="elastic_kernel")
    train_elastic_depth_stage_1(experiment)
    train_elastic_depth_stage_2(experiment)

def test_elastic_width():
    # Note that this will only work if you have previously run an experiment called "my_experiment"
    # and the elastic_depth_stage_2 stage was trained
    experiment_name = "my_experiment"
    experiment = experiment_from_config(experiment_name, load_stage="elastic_depth_stage_2")
    train_elastic_width_stage_1(experiment)
    train_elastic_width_stage_2(experiment)

def test_progressive_shrinking():
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


if __name__ == "__main__":
    # test_elastic_kernel()
    # test_elastic_depth()
    # test_elastic_width()
    test_progressive_shrinking()
