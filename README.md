# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MERCY A
RegisterNumber:  212223110027
*/

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
dataset.info()
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color='black')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


```

## Output:
#### Head and Tail:
![Screenshot 2024-08-27 111210](https://github.com/user-attachments/assets/1cdec813-4603-41b1-a12f-4565f460ec33)

#### Array value of X and Y:
![Screenshot 2024-08-27 111401](https://github.com/user-attachments/assets/921ba0af-721d-4ca2-8026-c58074a37151)

#### Y prediction and Y test values:
![Screenshot 2024-08-27 111526](https://github.com/user-attachments/assets/a995effc-c335-41f4-a823-40b9d8d25b11)

#### Training set graph:
![Screenshot 2024-08-27 111726](https://github.com/user-attachments/assets/f66f81e9-c5b3-46b2-8ea2-aa574d8a0d80)

#### Testing set graph:
![Screenshot 2024-08-27 111734](https://github.com/user-attachments/assets/babe4f24-27fb-496b-98fd-66b3127e277a)

#### Value of MSE,MAE,RMSE:
![Screenshot 2024-08-27 111800](https://github.com/user-attachments/assets/45242876-ac6f-4365-90b2-cb7bf7e90c56)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.









