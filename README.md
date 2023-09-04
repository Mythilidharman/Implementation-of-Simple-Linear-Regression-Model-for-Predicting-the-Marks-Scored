# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:mythili.D 
RegisterNumber: 212222040104



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

y_test
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scrores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scrores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```


## Output:

# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/08ba02f4-5e6c-4da8-841a-890495183518)
# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/eafaff50-6bba-47cc-9335-c755eac01fa7)
# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/c37c39b7-c844-454b-a06a-fab88ff2c4ea)
# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/8eeb2f26-867d-430c-85fa-ab5206a09a4b)
# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/26f77809-7230-48ce-832d-e3c65837874c)
# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/fcf9f1f1-5fc3-4ed4-a34b-60998c102665)
# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/c14400bc-b51d-489e-a4d3-8fc9662aa7fe)
# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/862089e6-ecf8-472e-b29f-5aab7f8c0702)
# ![image](https://github.com/Mythilidharman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104110/8df5c557-31ac-4dd4-9c28-bf258c531ba1)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
