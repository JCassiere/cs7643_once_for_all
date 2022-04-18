import random

from cs7643_once_for_all.ofa.model_arch import ModelArch


class EvoSearch:
    def __init__(self, architectures: list, pop: int, cycles: int, samples: int) -> None:
        self.P = pop
        self.C = cycles
        self.S = samples
        self.arch = architectures

    def search(self):
        population = []
        history = []

        for i in range(self.P):
            model = random.choice(self.arch)
            model.acc = train_eval(model.arch)
            population.append(model)
            history.append(model)
        
        for i in range(self.C):
            sample = []
            for j in range(self.S):
                candidate = random.choice(population)
                sample.append(candidate)
            
            parent = max(sample, key=lambda x: x.accuracy)
            child = ModelArch(parent+"_mutated", parent.arch)
            child.arch = mutate(parent.arch)
            population.append(child)
            history.append(child)
            dead = population.pop(0)
            del dead

        return max(history, lambda x: x.acc)



