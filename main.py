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

agent = DQNAgent()

nEpisodes = 500
scores = []
nActions = []
totRew = []
tRew = 0

t = 1
for i in range(nEpisodes):
    print("Current episode:", i)
    snake.ResetGame()  # need reset function (despair)
    pState = agent.convertInput(snake.SendGame())
    pScore = snake.GetScore()
    while True:
        action = agent.move(pState)
        snake.DirectMove(action.item())
        nState = agent.convertInput(snake.SendGame())
        gameOver = snake.GameOver()
        cScore = snake.GetScore()
        r = cScore - pScore - 0.05  # Penalty for living without getting any apples.
        pScore = cScore
        tRew += r

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

        # Soft target update?
        if t % 50 == 0 and agent.memLen() > 5000:
            agent.updateTarget()

        if gameOver:
            scores.append(cScore)
            nActions.append(t)
            totRew.append(tRew)
            break

        t += 1

plt.plot(nActions, "r")
plt.plot(totRew, "g")
plt.plot(scores, "b")
plt.legend(["Number of actions per episode", "Total cumulative score", "Episode score"])
plt.xlabel("Episode")
plt.show()
