import torch.nn
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from ofa import OnceForAll, SampleUnitConfiguration


if __name__ == "__main__":
    indices = torch.arange(600)
    mnist_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))
    mnist_train = data_utils.Subset(mnist_train, indices)
    
    indices = torch.arange(100)
    mnist_test = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))
    mnist_test = data_utils.Subset(mnist_test, indices)
    train_data_loader = DataLoader(mnist_train, batch_size=128)
    test_data_loader = DataLoader(mnist_test, batch_size=128)
    criterion = torch.nn.CrossEntropyLoss()
    ofa = OnceForAll(10, unit_first_layer_strides=[2, 1, 1, 2, 1])
    optimizer = torch.optim.Adam(ofa.parameters())
    base_configs = [SampleUnitConfiguration() for _ in range(len(ofa.conv_units))]
    base_configs[0].unit_input_channels = 1
    for epoch in range(20):
        ofa.train()
        for (idx, batch) in enumerate(train_data_loader):
            images, targets = batch
            output = ofa.forward(images, base_configs)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ofa.eval()
        test_correct = []
        for (idx, batch) in enumerate(train_data_loader):
            images, targets = batch
            output = ofa.forward(images, base_configs)
            pred = torch.argmax(output, dim=1)
            test_correct.append((pred == targets).int())
        test_correct = torch.cat(test_correct, dim=-1)
        accuracy = torch.sum(test_correct) / test_correct.shape[0]
        print("Epoch {} accuracy: {}".format(epoch, accuracy))
    