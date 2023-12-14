import torch

class DQNAgent:
    def __init__(self) -> None:
        pass

    def forward(self, x):
        pass

    def move(self):
        pass

    def convertInput(self, x, gameSize):
        gM = []
        for i in range(gameSize+2):
            tmp = []
            for j in range(gameSize+2):
                tmp.append(x[i, j])
            gM.append(tmp)
        return torch.stack([torch.Tensor(a) for a in zip(*gM)])

    def addToMem(self):
        pass

    def sampleMem(self):
        pass

    def optimizeModel(self):
        pass