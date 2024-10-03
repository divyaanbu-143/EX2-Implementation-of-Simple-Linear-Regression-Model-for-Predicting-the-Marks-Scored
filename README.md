# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: A Divya
RegisterNumber:  2305002007
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
x_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train,lr.predict(x_train),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, pred)
print(f'Mean Squared Error (MSE): {mse}')

*/
```

## Output:
![Screenshot 2024-10-03 081501](https://github.com/user-attachments/assets/a5d82592-2404-4315-af60-b55e21d22604)
![Screenshot 2024-10-03 081542](https://github.com/user-attachments/assets/e5a92e39-fe09-4d79-8414-d3b522a910c3)
![Screenshot 2024-10-03 081612](https://github.com/user-attachments/assets/cf28793c-754a-4556-85ca-d59fe1fd7d56)
![Screenshot 2024-10-03 081640](https://github.com/user-attachments/assets/ae449a85-8fc3-41fe-ae26-bf17ecf5760c)
![Screenshot 2024-10-03 081704](https://github.com/user-attachments/assets/1fc37d85-900e-4321-a93f-494f3c218da6)
![Screenshot 2024-10-03 081734](https://github.com/user-attachments/assets/00be262a-90a9-480a-afae-b147a24a3a46)

![Screenshot 2024-10-03 081753](https://github.com/user-attachments/assets/85651897-fc0d-4211-8bc2-2f7b8e169d01)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
