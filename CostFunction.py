from abc import ABC, abstractmethod
import numpy as np

class CostFuncation(ABC):

    @abstractmethod
    def calculate_error(self, y:np.ndarray, y_pred:np.ndarray) -> float:
        pass

class CostFunctionImpl(CostFuncation):

    def calculate_error(self, y:np.ndarray, y_pred:np.ndarray) -> float:
        return (1/(2*y.shape[0]))*np.sum(np.square(y_pred-y))