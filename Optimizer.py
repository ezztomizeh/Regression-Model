from abc import ABC,abstractmethod
import numpy as np

class Optimizer(ABC):

    @abstractmethod
    def calculate_w(self, x:np.ndarray, y:np.ndarray, y_pred:np.ndarray) -> float:
        pass

    @abstractmethod
    def calculate_b(self,y:np.ndarray, y_pred:np.ndarray) -> float:
        pass

class GradientDecent(Optimizer):

    def calculate_w(self, x:np.ndarray, y:np.ndarray, y_pred:np.ndarray) -> float:
        return (1/y.shape[0]) * np.sum(x*(y_pred-y))
    
    def calculate_b(self, y:np.ndarray, y_pred:np.ndarray) -> float:
        return (1/y.shape[0]) * np.sum(y_pred-y)