#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# # LOADING DATASET

# In[2]:


df = pd.read_csv (r'C:\Users\adity\Downloads\Reliance.csv')
print (df)


# In[3]:


df.head(6)


# In[4]:


df.tail()


# In[5]:


df.columns


# In[6]:


df.duplicated().sum()


# # CHECKING NULL VALUES

# In[7]:


df.isnull()


# In[8]:


df.isnull().sum()


# In[9]:


df.dropna(inplace=True)


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.nunique()


# In[13]:


df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace=True)


# In[14]:


df.head()


# In[15]:


df['date'] = pd.to_datetime(df.date)


# In[16]:


df.sort_values(by='date', inplace=True)


# # DURATION OF DATASET

# In[17]:


print("Starting date: ", df.iloc[0][0])
print("Ending date: ", df.iloc[-1][0])
print("Duration: ", df.iloc[-1][0]-df.iloc[0][0])


# In[18]:


monthvise= df.groupby(df['date'].dt.strftime('%B'))[['open','close']].mean().sort_values(by='close')


# In[19]:


monthvise.head()


# # IMPORTING OTHER LIBRARIES

# In[20]:


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime


# # MONTHWISE COMPARISON BETWEEN STOCK ACTUAL, OPEN AND CLOSE PRICE

# In[128]:


fig=go.Figure()

fig.add_trace(go.Bar(x=monthvise.index, y=monthvise['open'], name='Stock Open Price', marker_color='black'))

fig.add_trace(go.Bar(x=monthvise.index, y=monthvise['close'], name='Stock Close Price', marker_color='red'))

fig.update_layout(barmode='group', xaxis_tickangle=-45,title='Monthwise comparison between Stock actual, open and close price')

fig.show()


# In[21]:


df.groupby(df['date'].dt.strftime('%B'))['low'].min()


# In[22]:


monthvise_high=df.groupby(df['date'].dt.strftime('%B'))['high'].max()
monthvise_low=df.groupby(df['date'].dt.strftime('%B'))['low'].min()


# In[23]:


fig=go.Figure()
fig.add_trace(go.Bar(x=monthvise_high.index, y=monthvise_high, name='Stock High Price', marker_color='rgb(0,153,204)'))
fig.add_trace(go.Bar(x=monthvise_low.index, y=monthvise_low, name='Stock Low Price', marker_color='rgb(255,128,0)'))
fig.update_layout(barmode='group', title='Monthwise High and low Stock Price')

fig.show()


# In[24]:


from itertools import cycle


# # TREND COMPARISON BETWEEN STOCK PRICE, OPEN PRICE, CLOSE PRICE, HIGH PRICE AND LOW PRICE

# In[25]:


names=cycle(['Stock Open price','Stock Closed Price','Stock High Price','Stock Low Price'])
fig = px.line(df, x=df.date, y=[df['open'], df['close'], df['high'], df['low']], labels={'date':'Date','value':'Stock value'})
fig.update_layout(title_text='Stock Analysis Chart', font_size=15, font_color='red')
fig.for_each_trace(lambda t:t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()                                                                                         
                                                                                        
                                                                                         


# # MAKING SEPERATE DATAFRAME WITH CLOSE PRICE

# In[26]:


closedf=df[['date','close']]
print("Shape of close dataframe:", closedf.shape)


# # PLOTTING STOCK CLOSE PRICE CHART

# In[27]:


fig=px.line(closedf, x=closedf.date, y=closedf.close, labels={'date':'Date', 'close':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.6)
fig.update_layout(title_text='Stock Close Price Chart', plot_bgcolor='cyan', font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# # IMPORTING OTHER IMPORTANT LIBRARIES

# In[28]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler


# In[29]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU


# # NORMALISING VALUES

# In[30]:


close_stock=closedf.copy()
del closedf['date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)


# # SPLITTING DATA FOR TRAINING AND TESTING

# In[31]:


training_size=int(len(closedf)*0.65)
test_size=len(closedf)-training_size
train_data, test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data:",train_data.shape)
print("test_data:",test_data.shape)


# # CREATING NEW DATASET ACCORDING TO REQUIREMENT OF TIME-SERIES PREDICTION 

# In[32]:


def create_dataset(dataset, time_steps=1):
    dataX, dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[33]:


time_step=15
X_train, y_train=create_dataset(train_data,time_step)
X_test, y_test=create_dataset(test_data,time_step)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# # USING SVR MODEL FROM SVM(SUPPORT VECTOR MACHINE)

# In[34]:


from sklearn.svm import SVR
svr_rbf=SVR(kernel='rbf', C=1e2, gamma=0.1)
svr_rbf.fit(X_train, y_train)


# In[35]:


train_predict=svr_rbf.predict(X_train)
test_predict=svr_rbf.predict(X_test)

print(train_predict.shape)
print(test_predict.shape)

train_predict=train_predict.reshape(-1,1)
test_predict=test_predict.reshape(-1,1)

print("Train data prediction:", train_predict.shape)
print("Test data prediction:", test_predict.shape)


# In[36]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

print(train_predict.shape)
print(test_predict.shape)

original_ytrain=scaler.inverse_transform(y_train.reshape(-1,1))
original_ytest=scaler.inverse_transform(y_test.reshape(-1,1))

print(original_ytrain.shape)
print(original_ytest.shape)


# In[37]:


import math


# # EVALUATING METRICES

# In[38]:


print("Train data RMSE:", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE:", mean_squared_error(original_ytrain,train_predict))
print("Test data MAE:", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------")
print("Test data RMSE:", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE:", mean_squared_error(original_ytest,test_predict))
print("Test data MAE:", mean_absolute_error(original_ytest,test_predict))


# In[39]:


print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))


# In[40]:


print("Train data R2 score:",r2_score(original_ytrain, train_predict))
print("Test data R2 score:",r2_score(original_ytest, test_predict))


# In[41]:


print("Train data MGD:", mean_gamma_deviance(original_ytrain,train_predict))
print("Test data MGD:", mean_gamma_deviance(original_ytest,test_predict))
print("-----------------------------")
print("Train data MPD:", mean_poisson_deviance(original_ytrain,train_predict))
print("Test data MPD:", mean_poisson_deviance(original_ytest,test_predict))


# # COMPARISON BETWEEN ORIGINAL STOCK CLOSE PRICE VS PREDICTED PRICE

# In[42]:


look_back=time_step
trainPredictPlot=np.empty_like(closedf)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data:", trainPredictPlot.shape)

testPredictPlot=np.empty_like(closedf)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data:", testPredictPlot.shape)


# In[43]:


names=cycle(['Original close price','Train predicted close price','Test predicted close price'])
plotdf=pd.DataFrame({'date':close_stock['date'],'original_close':close_stock['close'],
                     'train_predicted_close':trainPredictPlot.reshape(1,-1)[0].tolist(),
                     'test_predicted_close':testPredictPlot.reshape(1,-1)[0].tolist()})
fig=px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'], plotdf['test_predicted_close']]
           , labels={'value':'Stock Price','date':'Date'})
fig.update_layout(title_text='Comparison between original close price vs predicted closed price',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# # PREDICTING NEXT 10 DAYS PRICE

# In[44]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days=10
while(i<pred_days):
    if(len(temp_input)>time_step):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        
        yhat=svr_rbf.predict(x_input)
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]
        
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        yhat=svr_rbf.predict(x_input)
        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())
        
        i=i+1
        
print("output of predicted next days:", len(lst_output))


    


# # PLOTTING LAST 15 DAYS VS NEXT 10 PREDICTED DAYS

# In[45]:


last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1, time_step+pred_days+1)
print(last_days)
print(day_pred)


# In[46]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

names = cycle(['Last 15 days close price','Predicted next 10 days close price'])

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[47]:


new_pred_plot.tail(5)


# # PLOTTING WHOLE CLOSING STOCK PRICE WITH PREDICTION

# In[48]:


rfdf=closedf.tolist()
rfdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
rfdf=scaler.inverse_transform(rfdf).reshape(1,-1).tolist()[0]

names = cycle(['Close price'])

fig = px.line(rfdf,labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[49]:


list1=np.array([2123,1234,4567,2122,2111,2221,2098,2112,1223,2000,4256,1489,1563,3246,1326]).reshape(1,-1)
svr_rbf.predict(list1)


# In[ ]:




