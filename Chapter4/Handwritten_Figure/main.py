import numpy as np
import os, sys


class Net:
    """2層のニュラルネットワーク(入力層は除く)
    """
    def __init__(self) -> None:
        self.W = {
            "1" : np.zeros(64, 100),
            "2" : np.zeros(100, 50)
        }
        
        self.B = {
            "1" : 1,
            "2" : 1
        }
                
        
net = Net()