from ofa.evolutionary_search import EvoSearch
import torch
from ofa.datasets import get_dataloaders
from ofa.utils import get_device
import random
from ofa.progressive_shrinking import get_network_config
from ofa.model_arch import ModelArch
from ofa.experiment import Experiment
import time
import pickle

def main(population, cycles, samples, dataset_name):
    start = time.time()
    device = get_device()
    exp_kwargs = {
            "dataset_name": dataset_name,
            "experiment_name": "big_net_only_{}_{}".format(dataset_name, int(time.time()))
        }

    exp = Experiment(**exp_kwargs)
    exp.net.load_state_dict(torch.load("checkpoint/elastic_width_stage_2.pt", map_location=torch.device(device)))

    search = EvoSearch(population, cycles, samples)
    # random.seed(100)
    num_blocks = 5
    kernel_choices = [3, 5]
    depth_choices = [2, 3, 4]
    expansion_ratio_choices = [3, 4, 6]

    batchsize = 64
    train_data_loader, test_data_loader = get_dataloaders(device, dataset_name="cifar10")

    result, history, pop, a_list = search.search(net=exp.net, loader=train_data_loader, batchsize=batchsize, num_blocks=num_blocks, device=device, kernel_choices=kernel_choices, depth_choices=depth_choices, expansion_ratio_choices=expansion_ratio_choices)

    print("Seconds:")
    print(time.time() - start)
    print()
    print(result)

    with open(f"{dataset_name}-pop{population}-{cycles}-{samples}.pkl", "wb") as f:
        pickle.dump(pop, f)

    with open(f"{dataset_name}-a_list{population}-{cycles}-{samples}.pkl", "wb") as f:
        pickle.dump(a_list, f)

    with open(f"{dataset_name}-history{population}-{cycles}-{samples}.pkl", "wb") as f:
        pickle.dump(history, f)

if __name__ == '__main__':
    main(1600, 800, 400, "cifar-10")