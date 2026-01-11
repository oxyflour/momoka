from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EnvBase(ABC):
    @abstractmethod
    def step(self, action: Optional[np.ndarray]) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self, camera: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass
