# Poverty Prediction during Covid19
# Using Technique : Machine Learning
# Algorithm : Random Forest Classification

# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from config import *

# %matplotlib inline
# Reading data
data = pd.read_csv(DATA_FILE)
test_data = pd.read_csv(TEST_FILE)

# Checking Null Values and dealing it
data.isnull().sum()
test_data.isnull().sum()
data=data.fillna(value=0)
test_data = test_data.fillna(value=0)

# Plotting Intial data
fig, ax = plt.subplots(figsize = (20,10))
sns.barplot(x=data["Multidimensional Poverty Index\n (MPI = H*A)"],
            y=data["Country"],
            data=data,ci=None,ax=ax)
plt.xlabel("MPI")
plt.ylabel("Country")
plt.show()


fig, ax = plt.subplots(figsize = (40,10))
sns.lineplot(x=data['GDP'],
             y=data['Multidimensional Poverty Index\n (MPI = H*A)'],
             data=data,sort=True,ax=ax)
plt.xlabel("GDP")
plt.ylabel("MPI")
plt.show()

fig, ax = plt.subplots(figsize = (20,10))
sns.barplot(x=data["Child Mortality"],
            y=data["Multidimensional Poverty Index\n (MPI = H*A)"],
            data=data,ax=ax)
plt.xlabel("Child Mortality")
plt.ylabel("MPI")
plt.show()

fig, ax = plt.subplots(figsize = (20,10))
sns.scatterplot(x=data["World Region"],
                y=data["Country"],
                hue=data['Poverty'],data=data,ax=ax)
plt.xlabel("World Region")
plt.ylabel("Country")
plt.show()

# Label Encoding train data
Country_encoder=LabelEncoder()
data["Country"]=Country_encoder.fit_transform(data["Country"])
Region_encoder=LabelEncoder()
data["World Region"]=Region_encoder.fit_transform(data["World Region"])
Poverty_encoder=LabelEncoder()
data["Poverty"]=Poverty_encoder.fit_transform(data["Poverty"])
data.head()
data=data.drop(columns='S NO')

# Defining x and y
x= data.drop(columns='Poverty')
y= data['Poverty']
x=np.array(x)
y=np.array(y)
x.shape,y.shape

# Scaling the data
scalerx=StandardScaler()
x=scalerx.fit_transform(x)

# Splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,
      test_size=0.40,random_state=6)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

# Model creation and Prediction
model = RandomForestClassifier(random_state=6,min_samples_split=4)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

# Training Score
train_score=model.score(x_train,train_pred)
train_score = round(train_score, 4)*100
print('Training Score = {} %'.format(train_score))

# Testing Score
test_score=model.score(x_test,test_pred)
test_score = round(test_score, 4)*100
print('Testing Score = {} %'.format(test_score))

# Printing the classification report
print(classification_report(y_train,train_pred))

Country = test_data["Country"]
Region = test_data["World Region"]
# Country.to_csv("E:\project\Country.csv",index=False)
# Region.to_csv("E:\poverty_data\Region.csv",index=False)

# Label Encoding test data
Country1_encoder=LabelEncoder()
test_data["Country"]=Country1_encoder.fit_transform(test_data["Country"])
Region1_encoder=LabelEncoder()
test_data["World Region"]=Region1_encoder.fit_transform(test_data["World Region"])

# Result Management
test_data=test_data.drop(columns='S NO')
test_data.shape
scalerx=StandardScaler()
test_data=scalerx.fit_transform(test_data)
test_prediction=model.predict(test_data)
results=pd.DataFrame(data=test_prediction,columns=['Poverty']).round(2)

# Printing the prediction classification report
print(classification_report(results,test_prediction))
results.replace(to_replace={0:'Not Poor',1:'Poor'},inplace=True)
Result = pd.concat([Country,Region,results], axis=1)
print(Result)
# Result.to_csv("E:\poject\poverty_predictions.csv",index=False)

# Plotting the result
fig, ax = plt.subplots(figsize = (20,10))
sns.scatterplot(x=Result["World Region"],
                y=Result["Country"],hue=Result['Poverty'],
                data=Result,ax=ax)
plt.xlabel("World Region")
plt.ylabel("Country")
plt.show()
