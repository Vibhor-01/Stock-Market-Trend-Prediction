#To run the Webapp Simply the copy the Below command in cmd in the same directory where the model and notebook is saved
# python -m streamlit run app.py 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
from keras.models import load_model
import streamlit as st

yf.pdr_override()
end=dt.datetime.now()
start=dt.datetime(2010,1,1)

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker Name', 'GOOG')
dataframe=pdr.get_data_yahoo(user_input, start=start, end=end)

#DescribingData
st.subheader('Data from 2010-2023')
st.write(dataframe.describe())
#visualizations
st.subheader('Open Price Vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(dataframe.Open)
st.pyplot(fig)

st.subheader('Opening Price Vs Time Chart With 100MA')
MA100=dataframe.Open.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(MA100)
plt.plot(dataframe.Open)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Opening Price Vs Time Chart With 100MA and 200MA')
MA100=dataframe.Open.rolling(100).mean()
MA200=dataframe.Open.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(MA100,'r')
plt.plot(MA200,'b')
plt.plot(dataframe.Open,'g')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig)

#Splitting Data into training and test Set

data_train=pd.DataFrame(dataframe['Open'][0:int(len(dataframe)*0.70)])
#To take only the Open Column in our dataframe we start from index o of Open column and then take the first 70% of the values

data_test=pd.DataFrame(dataframe['Open'][int(len(dataframe)*0.70): int(len(dataframe))])
#The Test Dataframe will take the remaining 30% values

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
data_train_array=sc.fit_transform(data_train)

#Loading Model
regressor=load_model('stock_pred.h5')

#Testing Part
past_100_days=data_train.tail(100)
final_df=pd.concat((past_100_days, data_test),ignore_index=True)
input_data=sc.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=regressor.predict(x_test)
scaler=sc.scale_

scalor_factor=1/scaler[0]
y_predicted=y_predicted*scalor_factor
y_test=y_test*scalor_factor

#Final Graph
st.subheader('Predictions Vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)