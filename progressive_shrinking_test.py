import torch.nn
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from ofa.mobilenetv3 import mobilenetv3_large, mobilenetv3_small

def train_mobilenetv3_cifar100():
    # indices = torch.arange(6000)
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
    # cifar_train = data_utils.Subset(cifar_train, indices)
    
    # indices = torch.arange(1000)
    cifar_test = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]))
    # cifar_test = data_utils.Subset(cifar_test, indices)
    train_data_loader = DataLoader(cifar_train, batch_size=128)
    test_data_loader = DataLoader(cifar_test, batch_size=128)
    criterion = torch.nn.CrossEntropyLoss()
    net = mobilenetv3_small(num_classes=100)
    # net = OnceForAll(10, unit_first_layer_strides=[2, 1, 1, 2, 1])
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.99)
    # base_configs = [SampleUnitConfiguration() for _ in range(len(net.conv_units))]
    # base_configs[0].unit_input_channels = 1
    
    
    for epoch in range(100):
        net.train()
        for (idx, batch) in enumerate(train_data_loader):
            optimizer.zero_grad()
            images, targets = batch
            # output = net.forward(images, base_configs)
            output = net.forward(images)
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
                output = net.forward(images)
                pred = torch.argmax(output, dim=1)
                test_correct.append((pred == targets).int())
            test_correct = torch.cat(test_correct, dim=-1)
            accuracy = torch.sum(test_correct) / test_correct.shape[0]
        print("Epoch {} accuracy: {}".format(epoch, accuracy))

if __name__ == "__main__":
    train_mobilenetv3_cifar100()
    