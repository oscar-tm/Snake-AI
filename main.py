snakePath = r"G:\Programmering\.NET\Snake\bin\Debug\net8.0"

import sys

sys.path.append(snakePath)

from time import sleep
import matplotlib.pyplot as plt
from pythonnet import load

load("coreclr")

import clr

clr.AddReference("Snake")
from Snake import SnakeGame  # type: ignore import is correct, telling pylance to ignore error

snake = SnakeGame(13, 3)

from SnakeDQNAgent import *

# Hyperarameters
memSize = 10000
bathSize = 512
gamma = 0.99
lr = 1e-4
lossF = nn.SmoothL1Loss()

agent = DQNAgent(memSize=memSize, batchSize=bathSize, gamma=gamma, lr=lr, lossF=lossF)

nEpisodes = 5000
scores = []
nActions = []
totRew = []
tRew = 0

# Still lacks something to learn completly
for i in range(nEpisodes):
    print("Current episode:", i)
    snake.ResetGame()
    pState = agent.convertInput(snake.SendGame())
    pScore = int(snake.GetScore())
    t = 0
    while True:
        action = agent.move(pState)
        snake.DirectMove(action.item())
        nState = agent.convertInput(snake.SendGame())
        gameOver = snake.GameOver()
        cScore = int(snake.GetScore())
        r = cScore - pScore
        pScore = cScore
        tRew += r

        if t > 2000:  # Stop from running forever
            gameOver = True

        if not gameOver:
            agent.addToMem(
                (
                    pState,
                    torch.tensor([action]).to(device).to(torch.long),
                    nState,
                    torch.tensor([r]).to(device),
                )
            )
        else:
            agent.addToMem(
                (
                    pState,
                    torch.tensor([action]).to(device).to(torch.long),
                    nState,
                    torch.tensor([-1]).to(device),
                )
            )

        agent.optimizeModel()
        pState = nState

        # Soft target update?
        if agent.nActions % 50 == 0 and agent.memLen() > 256:
            agent.updateTarget()

        if gameOver:
            scores.append(cScore)
            nActions.append(t)
            totRew.append(tRew)
            break

        t += 1

print("Total nr of actions:", sum(nActions))
print("Average nr of actions:", sum(nActions) / len(nActions))
plt.plot(nActions, "r")
plt.plot(totRew, "g")
plt.plot(scores, "b")
plt.legend(
    ["Number of actions per episode", "Total cumulative reward", "Episode score"]
)
plt.xlabel("Episode")
plt.show()
