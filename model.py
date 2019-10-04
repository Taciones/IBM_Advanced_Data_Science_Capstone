# Import dependencies
import pandas as pd
import numpy as np

# Load the dataset in a dataframe object and include only four features as mentioned

with open('sao-paulo-properties-april-2019.csv','r') as File:
    data = File.read()
df = pd.read_csv('sao-paulo-properties-april-2019.csv',names = None)
df1 = df[df.Negotiation_Type != 'sale']

df2 = df1[['Price','Condo','Size','Rooms','Toilets','Suites','Parking','Elevator','Furnished','Swimming Pool']]

labels = df2.Price
features = df2.drop(['Price'],axis=1)


# Linear Regression classifier
import sklearn
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
reg = LinearRegression()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    features,labels,test_size = 0.25,random_state=2, shuffle = True)

reg.fit(x_train,y_train)




# Save your model
from sklearn.externals import joblib
joblib.dump(reg, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")