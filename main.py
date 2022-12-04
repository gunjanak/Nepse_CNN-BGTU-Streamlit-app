
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model


#Importing my own libraries
from scrapper import price_history
from process import data_for_model
from rnn_models import Model_CNN_BGRU
from rnn_models import model_prediction
from rnn_models import final_prediction
from rnn_models import MAPE
from interpolate import find_lookback

def main():

    #setting starting date on 1st jan 2018
    Start_date = "2018-10-01"

    #setting default value of lookback period and future
    lookbackperiod = 5
    future = 0

    #Any empty dataframe
    company_df = pd.DataFrame()
   
    #The title
    st.title('CNN-BGRU method of Stock Price Prediction')

    #Asking using to enter the symbol listed in NEPSE
    company_name = st.text_input('Enter Company name')


    if(company_name != ""):
        
        #Asking user to enter the future
        future = st.text_input('Enter future')
        
        if(type(future) == str and future != ""):
            future = int(future)

        #Setting the lookback period for different type of futures
        if(future == 0):
            lookbackperiod = 5
        elif(future == 15):
            lookbackperiod = 50
          
        elif (future == 30):
            lookbackperiod = 100

        #using cubic spline interpolation to calculate lookback period
        elif (type(future) == int):
            lookbackperiod = find_lookback(future)
        else:
            future = 0


        try:
            #calling function to get price history
            company_df = price_history(company_name,Start_date)
        except:
            print("An exception occurred")

        #Printing the basics    
        st.subheader('Head of Data')
        st.dataframe(company_df.head())
        st.subheader('Tail of Data')
        st.dataframe(company_df.tail())

        st.write("Length of data")
        st.write(len(company_df))

        
        st.write("Length of lookback period")
        st.write(lookbackperiod)

        #converting the dataframe to data ready for tensorflow model
        data = data_for_model(company_df,lookbackperiod,future)
        
        X = data[1]
        Y = data[2]

        #Press this button to train the model
        #Along with training, we also compare the predicted and actual price
        #save the model
        model_name = 'model_cnn_bgru_'+company_name+"_"+str(future)+".h5"
        if st.button('Train and save Model'):
            st.write('Training')
            model_CNN_BGRU = Model_CNN_BGRU(X,Y,lookbackperiod)
            st.write('Saving')
            
            model_CNN_BGRU.save(model_name)
            st.write('Predicting ')
            record = model_prediction(data,model_CNN_BGRU,company_df,lookbackperiod,future)
            record = record.set_index('Date')
            st.write(record)
            st.line_chart(record)
            error = MAPE(record)
            st.write('Calculating MAPE')
            error = error*100
            st.write(error)
            

        #When this button is pressed our model will finally predict the stock price in required future
        if st.button("Predict for future"):
            #model_name = 'model_cnn_bgru_'+company_name+"_"+str(future)+".h5"
            model = load_model(model_name)
            prediction = final_prediction(company_df,lookbackperiod,data,model)
            prediction = "This model predicts that the stock price of " + company_name + " after "+str(future+1)+" days "+"is "+str(prediction[0][0])
            st.write(prediction)



if __name__ == '__main__':
    main()