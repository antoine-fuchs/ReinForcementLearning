import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import chess
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sys
sys.path.append('/content/gdrive/mypythondirectory')

import os

from environment import Board
from learn import Q_learning
from agent import Agent 

FEN = "k7/p1p1p1p1/1p1p1p1p/8/8/8/8/RNBQKBNR"

board = Board(FEN=FEN)

agent = Agent()

R = Q_learning(agent, board)
R.learn(iters=2)
