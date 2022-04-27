import torch
import matplotlib.pyplot as plt
import json
import numpy as np

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_num_params(net):
    param_count = sum([torch.numel(p) for p in net.parameters()])
    return param_count
    
def plot_curves(train_accuracy, test_accuracy, title):
    fig, ax = plt.subplots(figsize=(10,6))
    keys = list(test_accuracy.keys())
    tests = []
    train = ax.plot(np.arange(0.0, len(train_accuracy)), train_accuracy, label='Train Accuracy')
    for i in keys:
        tests.append(ax.plot(np.arange(0.0, len(test_accuracy[i])), test_accuracy[i], label=i))
    ax.set(title=title, xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    f = open('../experiments/cifar100_100_epochs_per_stage_1650948868-20220427T030902Z-001/big_network_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'],data['val_accuracies'], 'Big Network Accuracy Curve')
    
    f = open('../experiments/cifar100_100_epochs_per_stage_1650948868-20220427T030902Z-001/elastic_depth_stage_1_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Depth Stage 1 Accuracy Curve')
    
    f = open('../experiments/cifar100_100_epochs_per_stage_1650948868-20220427T030902Z-001/elastic_depth_stage_2_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Depth Stage 2 Accuracy Curve')
    
    f = open('../experiments/cifar100_100_epochs_per_stage_1650948868-20220427T030902Z-001/elastic_kernel_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Kernel Curve')
    
    f = open('../experiments/cifar100_100_epochs_per_stage_1650948868-20220427T030902Z-001/elastic_width_stage_1_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Width Stage 1 Curve')
    
    f = open('../experiments/cifar100_100_epochs_per_stage_1650948868-20220427T030902Z-001/elastic_width_stage_2_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Width Stage 2 Curve')
