import torch.nn
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from ofa.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from ofa.mobilenetv3_ofa import mobilenetv3_ofa
from ofa.progressive_shrinking import progressive_shrinking_from_scratch
from ofa.progressive_shrinking import train_loop
import cProfile
import os

def get_num_params(optimizer):
    params = optimizer.param_groups[0]['params']
    param_count = sum([torch.numel(p) for p in params])
    return param_count
    
def large_test_ofa_net():
    output_widths = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
    use_squeeze_excites = [False, False, True, False, True, True, True, True, True]
    use_hard_swishes = [True, False, False, True, True, True, True, True, True]
    strides = [1, 1, 1, 2, 1, 1, 1, 1, 1]
    net = mobilenetv3_ofa(num_classes=100, output_widths=output_widths, use_squeeze_excites=use_squeeze_excites,
                          use_hard_swishes=use_hard_swishes, strides=strides)
    return net

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def small_test_ofa_net():
    output_widths = [16, 16, 24, 64, 118, 800, 960]
    use_squeeze_excites = [False, False, True, False, True, True, True]
    use_hard_swishes = [True, False, False, True, True, True, True]
    strides = [1, 1, 2, 2, 2, 1, 1]

    net = mobilenetv3_ofa(num_classes=100, output_widths=output_widths, use_squeeze_excites=use_squeeze_excites,
                          use_hard_swishes=use_hard_swishes, strides=strides, width_mult=0.25)
    return net

def organize_val_folder():
    val_img_dir = os.path.join('./data/ImageNet/tiny-imagenet-200/val', 'images')

    # Open and read val annotations text file
    fp = open(os.path.join('./data/ImageNet/tiny-imagenet-200/val', 'val_annotations.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
    
def get_ImageNet_dataloaders(device, subset=True):
    def collate_fn_to_device(batch):
        # adapted from pytorch default_collate_fn
        transposed = list(zip(*batch))
        images = torch.stack(transposed[0], 0).to(device)
        targets = torch.tensor(transposed[1]).to(device)
        
        return images, targets
    
    ImageNet_train = torchvision.datasets.ImageFolder(
        root='./data/ImageNet/tiny-imagenet-200/train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
        ]))
    if subset:
        indices = torch.arange(6000)
        ImageNet_train = data_utils.Subset(ImageNet_train, indices)

    organize_val_folder()
    
    ImageNet_test = torchvision.datasets.ImageFolder(
        root='./data/ImageNet/tiny-imagenet-200/val',
        transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
        ]))
    if subset:
        indices = torch.arange(1000)
        ImageNet_test = data_utils.Subset(ImageNet_test, indices)
    train_data_loader = DataLoader(ImageNet_train, batch_size=128, shuffle=True,
                                   collate_fn=collate_fn_to_device)
    test_data_loader = DataLoader(ImageNet_test, batch_size=128, shuffle=True,
                                  collate_fn=collate_fn_to_device)
    return train_data_loader, test_data_loader
    
def train_mobilenetv3_ImageNet():
    device = get_device()
    train_data_loader, test_data_loader = get_ImageNet_dataloaders(device)
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

def train_mobilenetv3ofa_ImageNet_no_shrinking():
    device = get_device()
    train_data_loader, test_data_loader = get_ImageNet_dataloaders(device)

    net = small_test_ofa_net()
    net.to(device)
    
    train_loop(net, train_data_loader, test_data_loader, lr=0.1, epochs=100,
               depth_choices=[4], kernel_choices=[7], expansion_ratio_choices=[6])


def learn_on_small_kernel_only():
    device = get_device()
    train_data_loader, test_data_loader = get_ImageNet_dataloaders(device, subset=True)
    
    net = small_test_ofa_net()
    net.to(device)
    
    train_loop(net, train_data_loader, test_data_loader, lr=0.1, epochs=100,
               depth_choices=[4], kernel_choices=[3], expansion_ratio_choices=[6])
    
        
def test_elastic_kernel():
    device = get_device()
    train_data_loader, test_data_loader = get_ImageNet_dataloaders(device)

    net = small_test_ofa_net()
    net.to(device)
    train_loop(net, train_data_loader, test_data_loader, lr=0.64, epochs=10,
               depth_choices=[4], kernel_choices=[7, 5, 3],
               expansion_ratio_choices=[6])

def test_progressive_shrinking():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_loader, test_data_loader = get_ImageNet_dataloaders(device)

    net = small_test_ofa_net()
    # net = large_test_ofa_net()
    net.to(device)
    progressive_shrinking_from_scratch(train_data_loader, test_data_loader, net, base_net_epochs=50,
                                       elastic_kernel_epochs=100, elastic_depth_epochs_stage_1=5,
                                       elastic_depth_epochs_stage_2=50, elastic_width_epochs_stage_1=5,
                                       elastic_width_epochs_stage_2=50, base_net_lr=0.24, elastic_kernel_lr=0.20,
                                       elastic_depth_lr_stage_1=0.05, elastic_depth_lr_stage_2=0.1,
                                       elastic_width_lr_stage_1=0.05, elastic_width_lr_stage_2=0.1, )
    # progressive_shrinking(train_data_loader, test_data_loader, net, base_net_epochs=25,
    #                       elastic_kernel_epochs=25, elastic_depth_epochs_stage_1=5,
    #                       elastic_depth_epochs_stage_2=25, elastic_width_epochs_stage_1=5,
    #                       elastic_width_epochs_stage_2=25, base_net_lr=0.24, elastic_kernel_lr=0.20,
    #                       elastic_depth_lr_stage_1=0.05, elastic_depth_lr_stage_2=0.1,
    #                       elastic_width_lr_stage_1=0.05, elastic_width_lr_stage_2=0.1,)
    # progressive_shrinking(train_data_loader, test_data_loader, net, base_net_lr=0.24, elastic_kernel_lr=0.20,
    #                       elastic_depth_lr_stage_1=0.08, elastic_depth_lr_stage_2=0.16,
    #                       elastic_width_lr_stage_1=0.08, elastic_width_lr_stage_2=0.16)

# TODO - try learning just with elastic depth and/or elastic width
#  I think there is a bug with the elastic kernel
if __name__ == "__main__":
    # train_mobilenetv3_ImageNet()
    # cProfile.run('train_mobilenetv3ofa_ImageNet()', 'profile')
    # train_mobilenetv3ofa_ImageNet()
    # learn_on_small_kernel_only()
    
    # test_elastic_kernel()
    # 1 epoch/30s on colab
    test_progressive_shrinking()
