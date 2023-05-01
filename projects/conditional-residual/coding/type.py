from typing import List, Tuple, TypeVar

import numpy as np

Floats = TypeVar('Floats', float, np.ndarray)
Ints = TypeVar('Ints', int, np.ndarray)
StatesType = List[Tuple[int, int, bool]]
