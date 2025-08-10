from abc import ABC,abstractmethod
from Optimizer import Optimizer
from CostFunction import CostFuncation
import numpy as np

class Model(ABC):
    def __init__(self, x:int, y:int, optimizer:Optimizer, costFunction: CostFuncation):
        self.__x = x
        self.__y = y
        self.__optimizer = optimizer
        self.__costFunction = costFunction


    def setX(self,x:int) -> None:
        self.__x = x
    
    def setY(self,y:int) -> None:
        self.__y = y

    def setOptimizer(self,optimizer:Optimizer) -> None:
        self.__optimizer = optimizer

    def setCostFUnction(self,costFunction:CostFuncation) -> None:
        self.__costFunction = costFunction

    def getX(self) -> int:
        return self.__x
    
    def getY(self) -> int:
        return self.__y
    
    def getOptimizer(self) -> Optimizer:
        return self.__optimizer
    
    def getCostFunction(self) -> CostFuncation:
        return self.__costFunction

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def predict(self,y:int) -> int:
        pass
