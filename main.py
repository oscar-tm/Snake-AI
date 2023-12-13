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

while not testSnake.GameOver():
    testSnake.DisplayGame()
    testSnake.DirectMove(2)
    sleep(1)

testSnake.DisplayEndStats()