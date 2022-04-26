import torch
import matplotlib.pyplot as plt
import json
import numpy as np

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_num_params(net):
    param_count = sum([torch.numel(p) for p in net.parameters()])
    return param_count
    
def plot_big_network_curves(train_accuracy, test_accuracy):
    fig, ax = plt.subplots()
    train = ax.plot(np.arange(0.0, len(train_accuracy)),\
                         train_accuracy, label='Train Accuracy')
    test = ax.plot(np.arange(0.0, len(test_accuracy)),\
                         test_accuracy, label='Test Accuracy')
    ax.set(title='Accuracy Curve', xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(loc='lower right')
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.show()

def plot_elastic_depth_stage_1_curves(train_accuracy, test_accuracy1, test_accuracy2, test_accuracy3, test_accuracy4):
    fig, ax = plt.subplots()
    train = ax.plot(np.arange(0.0, len(train_accuracy)),\
                         train_accuracy, label='Train Accuracy')
    test1 = ax.plot(np.arange(0.0, len(test_accuracy1)),\
                         test_accuracy1, label='Test Accuracy 1')
    test2 = ax.plot(np.arange(0.0, len(test_accuracy2)),\
                         test_accuracy2, label='Test Accuracy 2')
    test3 = ax.plot(np.arange(0.0, len(test_accuracy3)),\
                         test_accuracy3, label='Test Accuracy 3')
    test4 = ax.plot(np.arange(0.0, len(test_accuracy4)),\
                         test_accuracy4, label='Test Accuracy 4')
    ax.set(title='Accuracy Curve', xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(loc='lower right')
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.show()

def plot_elastic_depth_stage_2_curves(train_accuracy, test_accuracy1, test_accuracy2, test_accuracy3, test_accuracy4, test_accuracy5, test_accuracy6):
    fig, ax = plt.subplots()
    train = ax.plot(np.arange(0.0, len(train_accuracy)),\
                         train_accuracy, label='Train Accuracy')
    test1 = ax.plot(np.arange(0.0, len(test_accuracy1)),\
                         test_accuracy1, label='Test Accuracy 1')
    test2 = ax.plot(np.arange(0.0, len(test_accuracy2)),\
                         test_accuracy2, label='Test Accuracy 2')
    test3 = ax.plot(np.arange(0.0, len(test_accuracy3)),\
                         test_accuracy3, label='Test Accuracy 3')
    test4 = ax.plot(np.arange(0.0, len(test_accuracy4)),\
                         test_accuracy4, label='Test Accuracy 4')
    test5 = ax.plot(np.arange(0.0, len(test_accuracy5)),\
                         test_accuracy5, label='Test Accuracy 5')
    test6 = ax.plot(np.arange(0.0, len(test_accuracy6)),\
                         test_accuracy6, label='Test Accuracy 6')
    ax.set(title='Accuracy Curve', xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(loc='lower right')
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.show()

def plot_elastic_kernel_curves(train_accuracy, test_accuracy1, test_accuracy2):
    fig, ax = plt.subplots()
    train = ax.plot(np.arange(0.0, len(train_accuracy)),\
                         train_accuracy, label='Train Accuracy')
    test1 = ax.plot(np.arange(0.0, len(test_accuracy1)),\
                         test_accuracy1, label='Test Accuracy 1')
    test2 = ax.plot(np.arange(0.0, len(test_accuracy2)),\
                         test_accuracy2, label='Test Accuracy 2')
    ax.set(title='Accuracy Curve', xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(loc='lower right')
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.show()

def plot_elastic_width_stage_1_curves(train_accuracy, test_accuracy1, test_accuracy2, test_accuracy3, test_accuracy4, test_accuracy5, test_accuracy6,\
                                      test_accuracy7, test_accuracy8, test_accuracy9, test_accuracy10, test_accuracy11, test_accuracy12):
    fig, ax = plt.subplots()
    train = ax.plot(np.arange(0.0, len(train_accuracy)),\
                         train_accuracy, label='Train Accuracy')
    test1 = ax.plot(np.arange(0.0, len(test_accuracy1)),\
                         test_accuracy1, label='Test Accuracy 1')
    test2 = ax.plot(np.arange(0.0, len(test_accuracy2)),\
                         test_accuracy2, label='Test Accuracy 2')
    test3 = ax.plot(np.arange(0.0, len(test_accuracy3)),\
                         test_accuracy3, label='Test Accuracy 3')
    test4 = ax.plot(np.arange(0.0, len(test_accuracy4)),\
                         test_accuracy4, label='Test Accuracy 4')
    test5 = ax.plot(np.arange(0.0, len(test_accuracy5)),\
                         test_accuracy5, label='Test Accuracy 5')
    test6 = ax.plot(np.arange(0.0, len(test_accuracy6)),\
                         test_accuracy6, label='Test Accuracy 6')
    test7 = ax.plot(np.arange(0.0, len(test_accuracy7)),\
                         test_accuracy7, label='Test Accuracy 7')
    test8 = ax.plot(np.arange(0.0, len(test_accuracy8)),\
                         test_accuracy8, label='Test Accuracy 8')
    test9 = ax.plot(np.arange(0.0, len(test_accuracy9)),\
                         test_accuracy9, label='Test Accuracy 9')
    test10 = ax.plot(np.arange(0.0, len(test_accuracy10)),\
                         test_accuracy10, label='Test Accuracy 10')
    test11 = ax.plot(np.arange(0.0, len(test_accuracy11)),\
                         test_accuracy11, label='Test Accuracy 11')
    test12 = ax.plot(np.arange(0.0, len(test_accuracy12)),\
                         test_accuracy12, label='Test Accuracy 12')
    ax.set(title='Accuracy Curve', xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(ncol=3, loc='lower right')
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.show()

def plot_elastic_width_stage_2_curves(train_accuracy, test_accuracy1, test_accuracy2, test_accuracy3, test_accuracy4, test_accuracy5, test_accuracy6,\
                                      test_accuracy7, test_accuracy8, test_accuracy9, test_accuracy10, test_accuracy11, test_accuracy12,\
                                      test_accuracy13, test_accuracy14, test_accuracy15, test_accuracy16, test_accuracy17, test_accuracy18):
    fig, ax = plt.subplots()
    train = ax.plot(np.arange(0.0, len(train_accuracy)),\
                         train_accuracy, label='Train Accuracy')
    test1 = ax.plot(np.arange(0.0, len(test_accuracy1)),\
                         test_accuracy1, label='Test Accuracy 1')
    test2 = ax.plot(np.arange(0.0, len(test_accuracy2)),\
                         test_accuracy2, label='Test Accuracy 2')
    test3 = ax.plot(np.arange(0.0, len(test_accuracy3)),\
                         test_accuracy3, label='Test Accuracy 3')
    test4 = ax.plot(np.arange(0.0, len(test_accuracy4)),\
                         test_accuracy4, label='Test Accuracy 4')
    test5 = ax.plot(np.arange(0.0, len(test_accuracy5)),\
                         test_accuracy5, label='Test Accuracy 5')
    test6 = ax.plot(np.arange(0.0, len(test_accuracy6)),\
                         test_accuracy6, label='Test Accuracy 6')
    test7 = ax.plot(np.arange(0.0, len(test_accuracy7)),\
                         test_accuracy7, label='Test Accuracy 7')
    test8 = ax.plot(np.arange(0.0, len(test_accuracy8)),\
                         test_accuracy8, label='Test Accuracy 8')
    test9 = ax.plot(np.arange(0.0, len(test_accuracy9)),\
                         test_accuracy9, label='Test Accuracy 9')
    test10 = ax.plot(np.arange(0.0, len(test_accuracy10)),\
                         test_accuracy10, label='Test Accuracy 10')
    test11 = ax.plot(np.arange(0.0, len(test_accuracy11)),\
                         test_accuracy11, label='Test Accuracy 11')
    test12 = ax.plot(np.arange(0.0, len(test_accuracy12)),\
                         test_accuracy12, label='Test Accuracy 12')
    test13 = ax.plot(np.arange(0.0, len(test_accuracy13)),\
                         test_accuracy13, label='Test Accuracy 13')
    test14 = ax.plot(np.arange(0.0, len(test_accuracy14)),\
                         test_accuracy14, label='Test Accuracy 14')
    test15 = ax.plot(np.arange(0.0, len(test_accuracy15)),\
                         test_accuracy15, label='Test Accuracy 15')
    test16 = ax.plot(np.arange(0.0, len(test_accuracy16)),\
                         test_accuracy16, label='Test Accuracy 16')
    test17 = ax.plot(np.arange(0.0, len(test_accuracy17)),\
                         test_accuracy17, label='Test Accuracy 17')
    test18 = ax.plot(np.arange(0.0, len(test_accuracy18)),\
                         test_accuracy18, label='Test Accuracy 18')
    ax.set(title='Accuracy Curve', xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(ncol=4, loc='lower right')
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.show()

if __name__ == "__main__":
    f = open('../experiments/cifar10_100_epochs_per_stage_1650866734/big_network_results.json')
    data = json.load(f)
    keys = list(data['val_accuracies'].keys())
    plot_big_network_curves(data['train_accuracies'],data['val_accuracies'][keys[0]])
    
    f = open('../experiments/cifar10_100_epochs_per_stage_1650866734/elastic_depth_stage_1_results.json')
    data = json.load(f)
    keys = list(data['val_accuracies'].keys())
    plot_elastic_depth_stage_1_curves(data['train_accuracies'], data['val_accuracies'][keys[0]], data['val_accuracies'][keys[1]],\
                                      data['val_accuracies'][keys[2]], data['val_accuracies'][keys[3]])
    
    f = open('../experiments/cifar10_100_epochs_per_stage_1650866734/elastic_depth_stage_2_results.json')
    data = json.load(f)
    keys = list(data['val_accuracies'].keys())
    plot_elastic_depth_stage_2_curves(data['train_accuracies'], data['val_accuracies'][keys[0]], data['val_accuracies'][keys[1]],\
                                      data['val_accuracies'][keys[2]], data['val_accuracies'][keys[3]], data['val_accuracies'][keys[4]],\
                                      data['val_accuracies'][keys[5]])
    
    f = open('../experiments/cifar10_100_epochs_per_stage_1650866734/elastic_kernel_results.json')
    data = json.load(f)
    keys = list(data['val_accuracies'].keys())
    plot_elastic_kernel_curves(data['train_accuracies'], data['val_accuracies'][keys[0]], data['val_accuracies'][keys[1]])
    
    f = open('../experiments/cifar10_100_epochs_per_stage_1650866734/elastic_width_stage_1_results.json')
    data = json.load(f)
    keys = list(data['val_accuracies'].keys())
    plot_elastic_width_stage_1_curves(data['train_accuracies'], data['val_accuracies'][keys[0]], data['val_accuracies'][keys[1]],\
                                      data['val_accuracies'][keys[2]], data['val_accuracies'][keys[3]], data['val_accuracies'][keys[4]],\
                                      data['val_accuracies'][keys[5]], data['val_accuracies'][keys[6]], data['val_accuracies'][keys[7]],\
                                      data['val_accuracies'][keys[8]], data['val_accuracies'][keys[9]], data['val_accuracies'][keys[10]],\
                                      data['val_accuracies'][keys[11]])
    
    f = open('../experiments/cifar10_100_epochs_per_stage_1650866734/elastic_width_stage_2_results.json')
    data = json.load(f)
    keys = list(data['val_accuracies'].keys())
    plot_elastic_width_stage_2_curves(data['train_accuracies'], data['val_accuracies'][keys[0]], data['val_accuracies'][keys[1]],\
                                      data['val_accuracies'][keys[2]], data['val_accuracies'][keys[3]], data['val_accuracies'][keys[4]],\
                                      data['val_accuracies'][keys[5]], data['val_accuracies'][keys[6]], data['val_accuracies'][keys[7]],\
                                      data['val_accuracies'][keys[8]], data['val_accuracies'][keys[9]], data['val_accuracies'][keys[10]],\
                                      data['val_accuracies'][keys[11]], data['val_accuracies'][keys[12]], data['val_accuracies'][keys[13]],\
                                      data['val_accuracies'][keys[14]], data['val_accuracies'][keys[15]], data['val_accuracies'][keys[16]],\
                                      data['val_accuracies'][keys[17]])
