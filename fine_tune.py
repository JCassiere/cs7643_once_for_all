import time
from ofa.experiment import Experiment
from ofa.progressive_shrinking import train_loop
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

def train_sub_network(experiment: Experiment, load_stage=None):
    if load_stage:
        experiment.load_net_post_stage(load_stage)
    kernel_choices = experiment.overall_kernel_choices[:]
    depth_choices = experiment.overall_depth_choices[:]
    expansion_ratio_choices = experiment.overall_expansion_ratio_choices[:]
    train_loop(
        experiment,
        lr=experiment.base_net_lr,
        epochs=10,
        depth_choices=depth_choices,
        kernel_choices=kernel_choices,
        expansion_ratio_choices=expansion_ratio_choices,
        eval_first=False,
        weight_decay=3e-4
    )
    experiment.log(stage="sub_network")

def plot_curves(train_accuracy, test_accuracy, tuned_train_accuracy, tuned_test_accuracy, title):
    fig, ax = plt.subplots(figsize=(10,6))
    
    keys = list(test_accuracy.keys())
    tests = []
    tuned_keys = list(tuned_test_accuracy.keys())
    tuned_tests = []
    
    train_series = pd.Series(train_accuracy)
    r_avg_train = train_series.rolling(5).mean()
    train = ax.plot(np.arange(0.0, len(r_avg_train)), r_avg_train, label='Train Accuracy')
    for i in keys:
        test_series = pd.Series(test_accuracy[i])
        r_avg_test = test_series.rolling(5).mean()
        tests.append(ax.plot(np.arange(0.0, len(r_avg_test)), r_avg_test, label=i))

    tuned_train_series = pd.Series(tuned_train_accuracy)
    r_avg_tuned_train = tuned_train_series.rolling(5).mean()
    tuned_train = ax.plot(np.arange(0.0, len(r_avg_tuned_train)), r_avg_tuned_train, label='Tuned Train Accuracy')
    for i in tuned_keys:
        tuned_test_series = pd.Series(tuned_test_accuracy[i])
        r_avg_tuned_test = tuned_test_series.rolling(5).mean()
        tuned_tests.append(ax.plot(np.arange(0.0, len(r_avg_tuned_test)), r_avg_tuned_test, label='Tuned '+i))
        
    ax.set(title=title, xlabel='Epochs', ylabel='Accuracy')
    y_start, y_end = ax.get_ylim()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(np.arange(0.0, len(train_accuracy), 5))
    plt.yticks(np.arange(0.0, y_end, 0.1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    '''exp_kwargs = {
        "dataset_name": "cifar10",
        "experiment_name": "cifar10_sub_net1651440498",
        "kernel_choices": [3],
        "depth_choices": [4],
        "expansion_ratio_choices": [3],
    }
    experiment = Experiment(**exp_kwargs)
    train_sub_network(experiment, 'elastic_width_stage_2')'''

    f1 = open('./experiments/cifar10_sub_net1651440498/big_network_results.json')
    data1 = json.load(f1)
    f2 = open('./experiments/cifar10_sub_net1651440498/sub_network_results.json')
    data2 = json.load(f2)
    plot_curves(data1['train_accuracies'], data1['val_accuracies'], data2['train_accuracies'], data2['val_accuracies'], 'Big Network vs Tuned Network Accuracy Curve (5-Epoch Moving Average)')
