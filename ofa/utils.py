import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_num_params(net):
    param_count = sum([torch.numel(p) for p in net.parameters()])
    return param_count
