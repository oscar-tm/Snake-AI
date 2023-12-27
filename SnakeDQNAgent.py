import torch
from torch import nn
from collections import deque, namedtuple
import math

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transitionMem = namedtuple("transitionMem", ["pState", "action", "nState", "reward"])


class DQNAgent:
    def __init__(
        self, memSize=10000, batchSize=128, gamma=0.95, lossF=nn.SmoothL1Loss(), lr=1e-4
    ) -> None:
        self.memSize = memSize
        self.cache = deque(maxlen=memSize)
        self.batchSize = batchSize
        self.gamma = gamma
        self.lossF = lossF
        self.nActions = 0

        self.onlineNet = nn.Sequential(
            nn.LazyConv2d(1, 3, device=device),
            nn.LeakyReLU(),
            nn.Flatten(1),
            nn.LazyLinear(512, device=device),
            nn.LeakyReLU(),
            nn.LazyLinear(128, device=device),
            nn.LeakyReLU(),
            nn.LazyLinear(4, device=device),
        ).to(device)
        self.targetNet = nn.Sequential(
            nn.LazyConv2d(1, 3, device=device),
            nn.LeakyReLU(),
            nn.Flatten(1),
            nn.LazyLinear(512, device=device),
            nn.LeakyReLU(),
            nn.LazyLinear(128, device=device),
            nn.LeakyReLU(),
            nn.LazyLinear(4, device=device),
        ).to(device)
        self.targetNet.load_state_dict(self.onlineNet.state_dict())

        self.optim = torch.optim.AdamW(self.onlineNet.parameters(), lr=lr)

    def forward(self, x, online=True):
        """
        Sends an input x to the correct net and returns the output.
        Input: An input tensor x to be sent to a net. A bool online to select which net to use.
        Returns: Returns the tensor from the selected neural net.
        """
        if online:
            return self.onlineNet(x)
        else:
            return self.targetNet(x)

    def move(self, x):
        """
        Does a move with the epsilon-greedy approach. Does a exponential decay of epsilon.
        Input: Input x the current state of the game.
        Returns: A epsilon-greedy move.
        """
        self.nActions += 1
        if random.random() > 0.05 + 0.85 * math.exp(-1 * self.nActions / 20000):
            with torch.no_grad():
                return torch.argmax(self.forward(x))

        else:
            return torch.Tensor([random.randint(0, 3)]).to(torch.long)

    def convertInput(self, x, gameSize=13):
        """ "
        Converts the int[,] from the game to a 3-D tensor.
        Inputs: X the int[,] array to be converted, gameSize is the boardsize of the game.
        Returns: The converted tensor
        """
        gM = []
        # Can be made faster by using only built-in torch functions. Will do later
        for i in range(gameSize + 2):
            tmp = []
            for j in range(gameSize + 2):
                tmp.append(x[i, j])
            gM.append(tmp)
        return (
            torch.stack([torch.Tensor(a) for a in zip(*gM)])
            .to(device)
            .reshape(1, gameSize + 2, gameSize + 2)
        )  # Need to transpose the game matrix here for now, will be fixed later. Not a high priority

    def optimizeModel(self):
        """
        Optimizing the model using the data stored in the memory
        """
        if (
            len(self.cache) < self.memSize * 0.1
        ):  # Only start to optimize model when memory is 10% full
            return

        sample = transitionMem(*zip(*self.sampleMem()))
        pSates = torch.stack(sample.pState)
        actions = torch.stack(sample.action)
        nStates = torch.stack(sample.nState)
        rewards = torch.stack(sample.reward)

        Q = self.forward(pSates).gather(1, actions)

        with torch.no_grad():
            V = self.forward(nStates, False).max(1).values

        V = (V.unsqueeze(1) * self.gamma) + rewards

        loss = self.lossF(Q, V)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def updateTarget(self):
        """
        Updates the target net to the current onlineNet.
        """
        self.targetNet.load_state_dict(self.onlineNet.state_dict())

    def save(self, path):
        """
        Saves the model to a file
        """
        torch.save(self.onlineNet.state_dict(), path)

    def load(self, path):
        """
        Loads the model from a file
        """
        self.onlineNet.load_state_dict(torch.load(path))

    def memLen(self):
        """
        Returns the current size of the memory
        """
        return len(self.cache)

    def addToMem(self, transition):
        """
        Adds a transition to the replay memory.
        """
        self.cache.append(transitionMem(*transition))

    def sampleMem(self, bathSize=64):
        """
        Samples the memory
        Returns: A sample of the size batchsize
        """
        return random.sample(self.cache, bathSize)
