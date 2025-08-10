from Model import Model
from Optimizer import Optimizer
from CostFunction import CostFuncation
import numpy as np

class RegressionModel(Model):
    def __init__(self, x:np.ndarray, y:np.ndarray, optimizer:Optimizer, 
                 costFunction:CostFuncation, numOfRounds:int,alpha:float = 0.01):
        super().__init__(x, y, optimizer, costFunction)
        self.__numOfRounds = numOfRounds
        self.__cost = []
        self.__alpha = alpha
        self.__w = 0
        self.__b = 0

    def setNumOfRounds(self,numOfRounds:int) -> None:
        self.__numOfRounds = numOfRounds

    def getNumOfRounds(self) -> int:
        return self.__numOfRounds
    
    def setAlpha(self, alpha:float) -> None:
        self.__alpha = alpha

    def getAlpha(self) -> float:
        return self.__alpha
    
    def predict(self, x:int) -> int:
        return (self.__w * x) + self.__b


    def fit(self) -> None:
        for i in range(self.getNumOfRounds()):
            y_pred = self.predict(self.getX())
            error = self.getCostFunction().calculate_error(y=self.getY(),y_pred=y_pred)
            self.__cost.append(error)
            print(f'''[!] Round {i} statistics: Error = {error}, w value = {self.__w},
                   b value = {self.__b}''')
            self.__w = self.__w - (self.getAlpha() * 
                                   self.getOptimizer().
                                   calculate_w(x=self.getX(),y=self.getY(),y_pred=y_pred))
            self.__b = self.__b - (self.getAlpha() * 
                                   self.getOptimizer().
                                   calculate_b(y=self.getY(),y_pred=y_pred))
