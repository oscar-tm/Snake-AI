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

testSnake = SnakeGame(13, 3)
testSnake.CreateBoard()

test = testSnake.SendGame()

from SnakeDQNAgent import *

agent = DQNAgent()

nEpisodes = 50
scores = []
nActions = []

for i in range(nEpisodes):
    print("Current episode:", i)
    pState = agent.convertInput(testSnake.SendGame())
    pScore = testSnake.GetScore()
    t = 1
    while True:
        action = agent.move(pState)
        testSnake.DirectMove(action.item())
        nState = agent.convertInput(testSnake.SendGame())
        gameOver = testSnake.GameOver()
        cScore = testSnake.GetScore()
        r = cScore - pScore - 0.05  # Penalty for living without getting any apples.

        agent.addToMem((pState, action, nState, r))

        agent.optimizeModel()

        # Soft target update?
        if t % 50 == 0 and agent.memLen > 5000:
            agent.updateTarget()

        if gameOver:
            scores.append(cScore)
            nActions.append(t)
            break

plt.plot(scores)
