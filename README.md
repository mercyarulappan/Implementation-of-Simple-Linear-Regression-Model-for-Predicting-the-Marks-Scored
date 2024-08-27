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
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MERCY A
RegisterNumber:  212223110027
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
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









