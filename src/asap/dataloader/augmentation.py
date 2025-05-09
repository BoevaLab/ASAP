import random
import numpy as np
from typing import Tuple

class ReverseComplement():
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p:
            return reverse_complement(x, y)
        return x, y

def reverse_complement(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Simple function that maps indexes of bases to their reverse complement:
    # A=0: f(0) = 3 = T
    # G=1: f(1) = 2 = C
    # C=2: f(2) = 1 = G
    # T=3: f(3) = 0 = A
    # N=4: f(4) = 4 = N
    f = lambda x: (8 - x) % 5
    if (x>4.).any():
        raise ValueError
    if (x<=1).all():
        # No C/T/N in array -- maybe not index-encoded but onehot?
        raise ValueError("No C/T/N in Tensor -- maybe not index-encoded but onehot?")
    return f(np.flip(x, 0)), np.flip(y, (0,))
