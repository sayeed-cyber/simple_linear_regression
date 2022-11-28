# importing the all required modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


saldata= pd.read_csv('Salary_Data.csv') # reading or opening the csv file
x= saldata.iloc[:,:-1].values # reading first column and converting it to 2D array
y=saldata.iloc[:,1].values #


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=1/3,random_state=0)# splitting the data into train and test data randomly

lr= LinearRegression() # creating the object
lr.fit(xtrain,ytrain)# sending the train data to fit function  and completing the training
pred=lr.predict(xtest) # predicting the test salary with test experience
print("predicted ytest data",pred) 
print("given ytest value",ytest)

#mean squared error " it is a mean value of squered difference between actual and pridected value"
sum=0
diff=(ytest-pred)**2
for item in diff:
    sum=sum+item
error=sum/len(ytest) 
print("error",error)



plt.scatter(xtest,ytest,color="green",marker="*")# scatter plot for given test data
plt.scatter(xtrain,ytrain,color="blue",marker="*")# scatter plot for given train data
plt.scatter(xtest,lr.predict(xtest),color="black",marker="+")# scatter plot for predicted test data 
plt.plot(xtest,lr.predict(xtest),color="red")# predicted line
plt.xlabel('''years of experience   
green * = given splited test data 25%
blue* =given splited train data 75%
black + =pridected salary 
red  line = linear regration line ''') # label for x axis
plt.ylabel("salary")# label for y axis
plt.show() # showing the graph



'''
by sayeed khan 
contact me on instagram :jefreonsyed 
'''
