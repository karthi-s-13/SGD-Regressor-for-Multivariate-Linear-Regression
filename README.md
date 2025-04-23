# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load California housing data, select features and targets, and split into training and testing sets.
2. Scale both X (features) and Y (targets) using StandardScaler.
3. Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
4. Predict on test data, inverse transform the results, and calculate the mean squared error. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: KARTHIKEYAN S
RegisterNumber:  212224230116
*/
```
```
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
```
```
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousePrice']=dataset.target
df.head()
```
```
x = df.drop(['AveOccup','HousePrice'],axis=1)
x.head()
```
```
y = df[['AveOccup','HousePrice']]
y.head()
```
```
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
```
```
scalar_x = StandardScaler()
scalar_y = StandardScaler()
x_train = scalar_x.fit_transform(x_train)
y_train = scalar_y.fit_transform(y_train)
x_test = scalar_x.transform(x_test)
y_test = scalar_y.transform(y_test)
```
```
sdg = SGDRegressor(max_iter=1000,tol=1e-3)
```
```
multi_output = MultiOutputRegressor(sdg)
```
```
multi_output.fit(x_train,y_train)
```
```
predict = multi_output.predict(x_test)
predict
```
```
#Inverse transform the predictions to get them back to the original scale
predict_inverse = scalar_y.inverse_transform(predict)
y_test = scalar_y.inverse_transform(y_test)
```
```
# Mean square Error
mse = mean_squared_error(y_test,predict_inverse)
print("Mean Square Error: {:.3f}".format(mse))
```
```
# Optional, print some predictions
print("Prediction:\n",predict_inverse[:5])
```











## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
