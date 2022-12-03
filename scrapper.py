from bs4 import BeautifulSoup
from requests_html import HTMLSession
#using regex to format string
import re 
import pandas as pd
from datetime import datetime
import numpy as np

def price_history(company_name,Start_date):
  # initialize an HTTP session
  session = HTMLSession()
  url = "https://www.nepalipaisa.com/Modules/CompanyProfile/webservices/CompanyService.asmx/GetCompanyPriceHistory"
  max_limit = 5000
#  Start_date = "2011-12-15"
  #End_date = "2020-12-15"
  End_date = datetime.today().strftime('%Y-%m-%d')

  data = {"StockSymbol": company_name, "FromDate": Start_date, "ToDate": End_date, "Offset": 1, "Limit": max_limit}
  res = session.post(url, data=data)

  result = res.text
  

  #removing comma
  result = result.replace(',', '')

  # removing <anything> between < and >
  a = re.sub("[\<].*?[\>]"," ", result)

  # this will return float and int in string
  d = re.findall("(\d+(?:\.\d+)?)", a)
  print(d[-20])
  real_max = int(d[-20])

  close_price = []

  start = 5
  for i in range(real_max):
    close_price.append(float(d[start]))
    start = start + 20
  

  dates = []
  start = 14
  for i in range(real_max):
    temp = d[start]+ "-"+d[start+1]+"-"+d[start+2]
    dates.append(temp)
    start = start + 20

  #Puting scrapped closed price with date in dataframe
  lst_col = ["Date","Close"]

  df = pd.DataFrame(columns = lst_col) 
  df['Date'] = dates
  df['Close'] = close_price
  #Putting oldest data at the start of dataframe
  df = df.iloc[::-1]

  #setting date as index
  df = df.set_index('Date')

  df = df.dropna()

  #replacing zero with previous data
  #df = df['Close'].replace(to_replace=0, method='ffill')
  df = df['Close'].mask(df['Close'] == 0).ffill(downcast='infer')
  #print(type(df))
  df = df.to_frame()
  #print(type(df))

  #df.plot(figsize=(16, 9))

  return df



