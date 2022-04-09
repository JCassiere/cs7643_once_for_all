import torchvision
from torch.utils.data import DataLoader
from ofa import OnceForAll, SampleUnitConfiguration


if __name__ == "__main__":
    mnist_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))
    mnist_test = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))
    data_loader = DataLoader(mnist_train, batch_size=64)
    ofa = OnceForAll(10, unit_first_layer_strides=[2, 1, 1, 2, 1])
    for (idx, batch) in enumerate(data_loader):
        configs = [SampleUnitConfiguration() for _ in range(len(ofa.conv_units))]
        configs[0].unit_input_channels = 1
        x = ofa.forward(batch[0], configs)
        print(1)
    