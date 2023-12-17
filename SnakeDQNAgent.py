import torch
from torch import nn
from collections import deque

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, memSize = 10000) -> None:
        self.cache = deque(maxlen = memSize)
        self.onlineNet = nn.Sequential(
            nn.LazyConv1d(1, 3, device = device),
            nn.LeakyReLU(),
            nn.LazyConv1d(1, 3, device = device),
            nn.LeakyReLU(),
            nn.Flatten(1),
            nn.LazyLinear(64, device = device),
            nn.LeakyReLU(),
            nn.LazyLinear(16, device = device),
            nn.LeakyReLU(),
            nn.LazyLinear(4, device = device)
        ).to(device)
        self.targetNet = nn.Sequential(
            nn.LazyConv1d(1, 3, device = device),
            nn.LeakyReLU(),
            nn.LazyConv1d(1, 3, device = device),
            nn.LeakyReLU(),
            nn.Flatten(1),
            nn.LazyLinear(64, device = device),
            nn.LeakyReLU(),
            nn.LazyLinear(16, device = device),
            nn.LeakyReLU(),
            nn.LazyLinear(4, device = device)
        )
        self.targetNet.load_state_dict(self.onlineNet.state_dict())

    def forward(self, x, online = True):
        """
        Sends an input x to the correct net. And return the net output
        """
        if online:
            return self.onlineNet(x)
        else:
            return self.targetNet(x)

    def move(self, x):
        """
        Does a move with the episolon-greedy approach.
        """
        if random.random() > 0.05:
            with torch.no_grad():
                return torch.argmax(self.forward(self.convertInput(x)))

        else:
            return random.randint(0,3)

    def convertInput(self, x, gameSize = 13):
        """"
        Converts the int[,] from the game to a tensor.
        Inputs: X the int[,] array to be converted, gameSize is the boardsize of the game.
        Returns: The converted tensor
        """
        gM = []
        for i in range(gameSize+2):
            tmp = []
            for j in range(gameSize+2):
                tmp.append(x[i, j])
            gM.append(tmp)
        return torch.div(torch.stack([torch.Tensor(a) for a in zip(*gM)]).to(device), 7) #Need to transpose the game matrix here for now, will be fixed later. Not a high priority

    def optimizeModel(self):
        pass

    def addToMem(self):
        """
        Adds a transition to the replay memory.
        """
        pass

    def sampleMem(self):
        pass