# Enum class for different actions

from enum import Enum

class Action(Enum):
    STAND = 0
    HIT = 1
    DOUBLE_DOWN = 2
    SPLIT = 3