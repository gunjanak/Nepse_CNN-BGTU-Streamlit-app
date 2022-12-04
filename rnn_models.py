import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional, GRU, Conv1D,Conv2D,MaxPooling1D



def Model_CNN_BGRU(X,Y,Look_back_period):

  model = Sequential()
  model.add(Conv1D(100, (3), activation='relu', padding='same', input_shape=(Look_back_period,1)))
  model.add(MaxPooling1D(pool_size=2,strides=1, padding='valid'))
  model.add(Bidirectional(GRU(50)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error',optimizer='adam',metrics = ['accuracy'])
  print(X.shape)
  model.fit(X,Y,epochs=100,batch_size=64,verbose=0)
  model_cnn_bgru = model

  return model_cnn_bgru


def model_prediction(data,model,company_df,Look_back_period,future):
  L = Look_back_period
  f = future
  a = data
  mean_std = a[0]
  mean = mean_std[0]
  std = mean_std[1]
  # 5 is no of features
  ran = company_df.shape[0] - (L+f)
  company_df = company_df.reset_index()

  #Dataframe to record
  column_names = ["Date", "Actual","Prediction"]
  record = pd.DataFrame(columns = column_names)
  #print(record)

  for i in range(ran):
    count = i+L
    tail =company_df[i:count]
    #print(tail)

    # setting date as index
    tail = tail.set_index('Date')


    numpy_array = tail.values
    predict_input = np.zeros(shape=(1,L,1))

    for i in range(L):
      predict_input[:,i] = numpy_array[i]

    predict_scaled = (predict_input-mean)/(std)
    #print("Shape of predict scaled")
    #print(predict_scaled.shape)
    #load model

    prediction = model.predict(predict_scaled)
    predict_final = (prediction*(std)) + mean
    
    count = count + f
    date = company_df['Date'][count]
    actual = company_df['Close'][count]
    list_predict = [date,actual,predict_final[0][0]]
    series_predict = pd.Series(list_predict, index = record.columns)
    record = record.append(series_predict, ignore_index=True)
    #converting to datatime format
    record['Date'] =  pd.to_datetime(record['Date'], infer_datetime_format=True)
    #print(type(record['Date'][0]))
  
  return record

def MAPE(record):
  num = record.shape[0]
  record['error'] = abs((record['Actual']-record['Prediction'])/record['Actual'])
  sum2 = record['error'].sum()
  MAPE = sum2/num
  return MAPE


def final_prediction(company_df,lookback,data,model):
    mean_std = data[0]
    mean = mean_std[0]
    std = mean_std[1]

    last_data = company_df.tail(lookback)
    last_data = last_data.values
    predict_scaled = (last_data-mean)/(std)
    predict_scaled = predict_scaled.reshape((1,lookback,1))
    prediction = model.predict(predict_scaled)
    predict_final = (prediction*(std)) + mean
  
    return predict_final