import random
from cs7643_once_for_all.ofa.accuracy_network import AccNetTrainer
from cs7643_once_for_all.ofa.model_arch import ModelArch
import copy


class EvoSearch:
    def __init__(self, architectures: list, pop: int, cycles: int, samples: int) -> None:
        self.P = pop
        self.C = cycles
        self.S = samples
        self.arch = architectures

    def search(self, net, loader, batchsize=64, num_blocks = 5, kernel_choices = [3, 5, 7], depth_choices = [2, 3, 4], expansion_ratio_choices = [3, 4, 6]):
        population = []
        history = []
        acc_net_trainer = AccNetTrainer(net, loader, batchsize, num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
        acc_net_trainer.train()
        
        for i in range(self.P):
            depths = random.choice(depth_choices)
            kernels = random.choice(kernel_choices)
            ex_ratios = random.choice(expansion_ratio_choices)
            config = {}
            config['depths'] = depths
            config['kernel_sizes'] = kernels
            config['expansion_ratios'] = ex_ratios
            model = ModelArch(config)
            model.acc = acc_net_trainer.model(depths, kernels, ex_ratios)
            population.append(model)
            history.append(model)
        
        for i in range(self.C):
            sample = []
            for j in range(self.S):
                candidate = random.choice(population)
                sample.append(candidate)
            
            parent = max(sample, key=lambda x: x.accuracy)
            child_config_dict = self.mutate(parent.config_dict, kernel_choices, depth_choices, expansion_ratio_choices)
            child = ModelArch(name=parent+"_mutated", config_dict=child_config_dict)
            child.acc = acc_net_trainer.model(child.depth, child.kernel, child.expansion_ratio)
            population.append(child)
            history.append(child)
            dead = population.pop(0)
            del dead

        return max(history, lambda x: x.acc)

    def mutate(self, config_dict, kernel_choices, depth_choices, expansion_ratio_choices):
        new_dict = copy.deepcopy(config_dict)
        choice = random.choice([0, 1, 2])
        if choice == 0:
            new_dict['depths'] = random.choice(depth_choices)
        elif choice == 1:
            new_dict["kernel_sizes"] = random.choice(kernel_choices)
        elif choice == 2:
            new_dict["expansion_ratios"] = random.choice(expansion_ratio_choices)

        return new_dict


