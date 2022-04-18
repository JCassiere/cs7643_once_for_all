from pyexpat import model


class ModelArch:
    def __init__(self, name, model, acc = None) -> None:
        self.name = name
        self.arch = model
        self.accuracy = acc