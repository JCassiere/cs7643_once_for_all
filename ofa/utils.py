import torch
import matplotlib.pyplot as plt
import json
import numpy as np

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_num_params(net):
    param_count = sum([torch.numel(p) for p in net.parameters()])
    return param_count
    
def plot_curves(train_accuracy, test_accuracy):
    fig, ax = plt.subplots()
    train = ax.plot(np.arange(0.0, len(train_accuracy)),\
                         train_accuracy, label='Train Accuracy')
    test = ax.plot(np.arange(0.0, len(test_accuracy)),\
                         test_accuracy, label='Test Accuracy')
    ax.set(title='Accuracy Curve', xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(loc='upper right')
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.show()
    fig.savefig("Plot.png")

f = open('../experiments/cifar10_100_epochs_per_stage_1650866734/big_network_results.json')
data = json.load(f)
plot_curves(data['train_accuracies'],data['val_accuracies']['K5-D4-ExR6'])
