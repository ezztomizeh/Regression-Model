from CostFunction import CostFunctionImpl
from Optimizer import GradientDecent
from RegressionModel import RegressionModel
import numpy as np

x = np.array([1,2,3])
y = np.array([150,250,350])

model = RegressionModel(x,y,GradientDecent(),CostFunctionImpl(),1000)
model.fit()

y_pred = model.predict(4)
print(f'f({4}): {y_pred}')