import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_num_params(net):
    param_count = sum([torch.numel(p) for p in net.parameters()])
    return param_count
    
def plot_curves(train_accuracy, test_accuracy, title):
    fig, ax = plt.subplots(figsize=(10,6))
    keys = list(test_accuracy.keys())
    tests = []
    train_series = pd.Series(train_accuracy)
    r_avg_train = train_series.rolling(5).mean()
    train = ax.plot(np.arange(0.0, len(r_avg_train)), r_avg_train, label='Train Accuracy')
    for i in keys:
        test_series = pd.Series(test_accuracy[i])
        r_avg_test = test_series.rolling(5).mean()
        tests.append(ax.plot(np.arange(0.0, len(r_avg_test)), r_avg_test, label=i))
    ax.set(title=title, xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    '''f = open('../experiments/cifar10_sub_net1651440498/big_network_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'],data['val_accuracies'], 'Big Network Accuracy Curve (5-Epoch Moving Average)')
    
    f = open('../experiments/cifar10_sub_net1651440498/elastic_depth_stage_1_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Depth Stage 1 Accuracy Curve (5-Epoch Moving Average)')
    
    f = open('../experiments/cifar10_sub_net1651440498/elastic_depth_stage_2_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Depth Stage 2 Accuracy Curve (5-Epoch Moving Average)')
    
    f = open('../experiments/cifar10_sub_net1651440498/elastic_kernel_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Kernel Accuracy Curve (5-Epoch Moving Average)')
    
    f = open('../experiments/cifar10_sub_net1651440498/elastic_width_stage_1_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Width Stage 1 Accuracy Curve (5-Epoch Moving Average)')
    
    f = open('../experiments/cifar10_sub_net1651440498/elastic_width_stage_2_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Width Stage 2 Accuracy Curve (5-Epoch Moving Average)')'''

    '''f = open('../experiments/cifar10_k357_elastic_kernel_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Elastic Kernel Accuracy Curve (5-Epoch Moving Average) (kernel_choices = [3,5,7])')

    f = open('../experiments/cifar10_k357_big_network_results.json')
    data = json.load(f)
    plot_curves(data['train_accuracies'], data['val_accuracies'], 'Big Network Accuracy Curve (5-Epoch Moving Average) (kernel_choices = [3,5,7])')'''
