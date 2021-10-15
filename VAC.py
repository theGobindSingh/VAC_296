import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

data_set=pd.read_csv('http://bit.ly/w-data')
data_set.head()
data_set.isnull()

corrmat = data_set.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20)) #for heat map
g=sns.heatmap(data_set[top_corr_features].corr(),annot=True,cmap="RdYlGn")

data_set.plot(x='Hours', y='Scores',style='o' )  
plt.show()

independent=data_set.iloc[:,:-1].values
dependent=data_set.iloc[:,1].values

independent_train, independent_test, dependent_train, dependent_test = train_test_split(independent, dependent,test_size=0.2, random_state=0)
regressor=LinearRegression()
regressor.fit(independent_train,dependent_train)

line=regressor.coef_*independent+regressor.intercept_
plt.scatter(independent,dependent)
plt.plot(independent,line)
plt.show()

print(independent_test)
print()

dependent_pred=regressor.predict(independent_test)

model=pd.DataFrame({'Actual':dependent_test,'Predicted':dependent_pred})

print(model)
print()

hours=[[9.25]]
own_pred=regressor.predict(hours)
print()

print("Number of hours={}".format(hours))
if own_pred[0]>100:
    print("prediction score=100")
else:
    print("Prediction Score = {}".format(own_pred[0]))
print()

print('Mean Absolute Error:', metrics.mean_absolute_error(dependent_test, dependent_pred))
print()

print('Variance score :%2f'% regressor.score(independent_test,dependent_test))
print()