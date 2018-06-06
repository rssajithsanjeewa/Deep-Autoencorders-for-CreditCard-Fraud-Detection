### Ref Materials
## Data source : https://www.kaggle.com/mlg-ulb/creditcardfraud/data
## This contain data set contain 492 frauds and 284,315 Normal transactions
## Data highly imbalance due to that we are using Deep Autoencorder for creating model
## Ref webpages & materials :
## 1) https://shiring.github.io/machine_learning/2017/05/01/fraud
## 2) https://blog.keras.io/building-autoencoders-in-keras.html
## 3) https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798
## 4) https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd
## 5) https://elitedatascience.com/keras-tutorial-deep-learning-in-python
## 6) http://thesai.org/Downloads/Volume9No1/Paper_3-Credit_Card_Fraud_Detection_Using_Deep_Learning.pdf
## 7) https://en.wikipedia.org/wiki/Autoencoder
## 8) https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/
## 9) https://github.com/otenim/AnomalyDetectionUsingAutoencoder
## 10) http://mail.tku.edu.tw/myday/teaching/1042/SCBDA/1042SCBDA09_Social_Computing_and_Big_Data_Analytics.pdf
## 11) https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
## 12) https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/


## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler

#Importing csv file
df = pd.read_csv("./creditcard.csv")
print('Null values :',df.isnull().values.any()) ## Checking whether data set contains any null values
print(df.describe()) ## Describe the data set
print('Import dataframe shape',df.shape)  ## datframe size
print(df.head(2))  ##first 2 rows of the data frmae
print('***********************')

# creating two days and tag them
df['day'] = np.where(df['Time']<86401, 'day1', 'day2')
df['Status'] = np.where(df['Class']==0, 'N', 'Y') # Adding new column and decode class
print('***********************')
print(df.head(2))  ##first 2 rows of the data frmae

# Creting graphs to check fraud trnasactions & normal transactions
plt.show(sns.lmplot(x='Time',y='Amount',data=df[df.day=='day1'],hue='Status')) # Day 1 Normal Vs Fraud transations
plt.show(sns.lmplot(x='Time',y='Amount',data=df[df.day=='day2'],hue='Status')) # Day 2 Normal Vs Fraud transation
#plt.show(sns.pairplot(df[df.day=='day1'], kind="scatter",hue='Status'))
#plt.show(sns.pairplot(df[df.day=='day2'], kind="scatter",hue='Status'))

## showing the anomoly
plt.show(df.pivot_table(values=["Class"],index=["Status"],aggfunc='count').plot(kind='bar')) # Fraud Vs Normal
plt.show(df.pivot_table(values=["Class"],index=["day","Status"],aggfunc='count').plot(kind='bar')) # Fraud Vs Normal daily

print('***********************')
print(df.pivot_table(values=["Class"],index=["Status"],aggfunc='count')) # Fraud Vs Normal - pivot
print(df.pivot_table(values=["Class"],index=["day","Status"],aggfunc='count')) # Fraud Vs Normal daily - pivot

## Applying auto encorders
print('***********************')
data = df.drop(['Time','day','Status'], axis=1) # Removing unwanted columns
print(data.shape)

## Scaling data
print('***********************')
scaler = MinMaxScaler(feature_range=(0, 1))
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
print(data.Amount.describe())
print('*********************')

# Data partition to test & train data set 70% totrain & 30% to test
X_train, X_test = train_test_split(data, test_size=0.3, random_state=100)
print('x train',X_train.shape) ## train data set
print('x test',X_test.shape) ## test data set

X_train = X_train[X_train.Class == 0]  ## selecting only Normal transactions for inserting to autoencorder
X_train = X_train.drop(['Class'], axis=1) ## removing class variable
y_test = X_test['Class'] ## Test data
X_test = X_test.drop(['Class'], axis=1) # Remove CLass variable to test the data
X_train = X_train.values
X_test = X_test.values
print('x train',X_train.shape)
print('x train',X_train.shape[1])

## Deep Autoencorder
input_l=Input(shape=(29,))
## Encode
encoded=Dense(25,activation='relu')(input_l)
encoded=Dense(20,activation='relu')(encoded)
encoded=Dense(10,activation='relu')(encoded)
encoded=Dense(5,activation='relu')(encoded)
## Decode
decoded=Dense(10,activation='relu')(encoded)
decoded=Dense(20,activation='relu')(decoded)
decoded=Dense(25,activation='relu')(decoded)
decoded=Dense(29,activation='relu')(decoded)

autoencorder=Model(inputs=input_l,outputs=decoded)
autoencorder.compile(optimizer='adam', loss='binary_crossentropy')

autoencorder.fit(X_train, X_train,
                epochs=200,
                batch_size=10000,
                shuffle=True,
                validation_data=(X_test, X_test))

## Predict the result for test data set
predictions=autoencorder.predict(X_test)

## Error
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test})

## Confusion Matrix
## get 2.6 as the thresor hold
print(error_df.head(2))
print('************************')
error_df['y_pred'] = np.where(error_df['reconstruction_error']>2.6, '1', '0')
print(pd.crosstab(error_df['true_class'], error_df['y_pred']))
