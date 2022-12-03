import numpy as np
import pandas as pd
from sklearn import preprocessing




def make_frame(input_frame, scaled_frame,rows,columns):
  
  for j in range(rows):
    count = 0
    for i in range(0+j,columns+j):
      input_frame[j][count] = scaled_frame[i][0]
      count = count + 1
  return input_frame

def data_for_model(company_df,Look_back_period,future):
  company_df = company_df.fillna(method='ffill')
  
  numpy_array = company_df.values
  
  close_array = numpy_array
  entries_total = close_array.shape[0]
  
  mean_std_array = np.array([close_array.mean(),close_array.std()])
  mean_std_scaler = preprocessing.StandardScaler()
  close_scaled = mean_std_scaler.fit_transform(close_array)
  rows = Look_back_period
  columns = entries_total-(rows+future)
  company_input = np.zeros(shape=(rows,columns))

  company_input = make_frame(company_input,close_scaled,rows,columns)

  company_input = company_input.T
  company_output = np.zeros(shape=(columns,1))
  for i in range(rows,(columns+rows)):
    company_output[i-rows][0] = close_scaled[i+future][0]
  
  #combining all arrays
  features = 1
  company_input_3d = np.zeros(shape=(columns,rows,features))

  company_input_3d[:,:,0] = company_input

  return_list = []

  return_list.append(mean_std_array)
  return_list.append(company_input_3d)
  return_list.append(company_output)

  return return_list