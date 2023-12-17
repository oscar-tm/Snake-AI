snakePath = r"G:\Programmering\.NET\Snake\bin\Debug\net8.0"

import sys
sys.path.append(snakePath)

from time import sleep

from pythonnet import load
load("coreclr")

import clr
clr.AddReference("Snake")

from Snake import SnakeGame

testSnake = SnakeGame(13, 3)
testSnake.CreateBoard()

test = testSnake.SendGame()

from SnakeDQNAgent import *
agent = DQNAgent()

print(agent.convertInput(test, 13).size())
print(agent.forward(agent.convertInput(test, 13)))
print(agent.move(test))