import torch
import torchvision
from torch.utils.data import DataLoader

def get_dataloaders(device, dataset_name="cifar100"):
    def collate_fn_to_device(batch):
        # adapted from pytorch default_collate_fn
        transposed = list(zip(*batch))
        images = torch.stack(transposed[0], 0).to(device)
        targets = torch.tensor(transposed[1]).to(device)
        
        return images, targets
    
    if dataset_name == "cifar100":
        torch_dataset = torchvision.datasets.CIFAR100
        norm_mean = (0.5071, 0.4865, 0.4409)
        norm_std = (0.2673, 0.2564, 0.2762)
    elif dataset_name == "cifar10":
        torch_dataset = torchvision.datasets.CIFAR10
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2023, 0.1994, 0.2010)
    else:
        raise ValueError("{} dataset not supported".format(dataset_name))
    
    train_dataset = torch_dataset(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            # this performs random shifts by up to 4 pixels
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
        ]))
    
    test_dataset = torch_dataset(
        root='./data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
        ]))
    
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                   collate_fn=collate_fn_to_device)
    test_data_loader = DataLoader(test_dataset, batch_size=256, shuffle=True,
                                  collate_fn=collate_fn_to_device)
    
    return train_data_loader, test_data_loader
