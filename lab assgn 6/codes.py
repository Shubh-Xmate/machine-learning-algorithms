"""
IC272 DSIII
lab Assignment 6

Shubham Shukla
B20168
mob : 8317012277
"""

# importing important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats.stats import pearsonr
from statsmodels.tsa.ar_model import AutoReg as AR
import statsmodels.api as statMod
from sklearn.metrics import mean_squared_error

"""Question 1"""
#reading the csv data
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/lab assgn 6/daily_covid_cases.csv")


'''part(a)'''
# making list for xticks
xticks_points = [21]; # for 20th feb

# now will iterate by two months till oct
for i in range(10):
    xticks_points.append(xticks_points[i] + 60);
    
# now creating labels for these points
xticks_labels = ['Feb-20', 'Apr-20', 'Jun-20', 'Aug-20', 'Oct-20', 'Dec-20', 'Feb-21', 'Apr-21', 'Jun-21', 'Aug-21', 'Oct-21']

# storing the new cases data in a variable
y_vals = df['new_cases']

# now let's plot
fig = plt.figure(figsize = (20, 10));
plt.plot(y_vals)
plt.xticks(xticks_points, xticks_labels, fontsize = 12)
plt.xlabel('Month-Year', fontsize = 14)
plt.ylabel('New Confirmed Cases', fontsize = 14)
plt.title('New Confirmed Cases vs Month-Year', fontsize = 16)
plt.show()

def create_lag(data, lag):
    return df.iloc[:(data.shape[0] - lag), :];

'''part(b)'''
# getting the number of rows
num_rows = df.shape[0]

# storing the lag series 0 and 1
lag_series_0 = df.iloc[1:num_rows, :];
lag_series_1 = create_lag(df, 1)

# calculating the correlation between them and showing the result
print("The Pearson  correlation  (autocorrelation)  coefficient  between  \nthe  generated  one-day  lag  time \
sequence and the given time sequence : ", format(lag_series_0['new_cases'].corr(lag_series_1['new_cases']),'.3f'))

'''part(c)'''
# since we have all the required data so plotting the things accordingly''''''
plt.scatter(lag_series_0['new_cases'], lag_series_1['new_cases'], s = 5, c = 'g')
plt.title("no lag series vs one-day lagged series", fontsize = 14)
plt.xlabel("no lag series", fontsize = 12)
plt.ylabel("one-day lagged series", fontsize = 12)
plt.show()

'''part(d)'''

# creating two arrays, one will contain lag series and other correspoding original series 
# as we have to slice the data according to lag coeff that's why necessary
lag_series = []; corr_original_series = []

# storing the lag coefficients
lag_val = (1,2,3,4,5,6)
for i in lag_val:
    lag_series.append(create_lag(df,i));
    corr_original_series.append(df.iloc[i:, :])

# creating an array to store the correspoding correlation wrt lag val
corr_correlation = [];
for i in lag_val:
    corr_correlation.append(lag_series[i - 1]['new_cases'].corr(corr_original_series[i-1]['new_cases']))
    
# plotting lag coefficients vs corresponding correlation using scatter plot
plt.scatter(lag_val, corr_correlation, s = 20, c = 'g')
plt.xlabel('Lag number')
plt.ylabel('Corresponding correlation')
plt.title('Correlation coefficient vs lag value')
plt.show()

'''part(e)'''
# plotting corelogram using plot_acf function
statMod.graphics.tsa.plot_acf(df['new_cases'],lags=lag_val)
plt.xlabel('days lagged')
plt.ylabel('Corresponding Correlation coefficient')
plt.show()


"""Question 2"""
#reading the csv data again
series = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/lab assgn 6/daily_covid_cases.csv', parse_dates=['Date'], index_col=['Date'], sep=',') 

test_size = 0.35 # 35% for testing 
X = series.values 
tst_sz = math.ceil(len(X)*test_size) 
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

'''Part(a)'''
p = 5

# making model for lag 5 series
ar_model = AR(train, lags = p)
AR_model = ar_model.fit()
coef = AR_model.params # storing coefficient for lag 5 data
print('The coefficient for lag 5 series are : \n', coef)

'''part(b)'''

history = train[len(train)-p:] 
history = [history[i] for i in range(len(history))] 
predictions = list() # List to hold the predictions, 1 step at a time 
for t in range(len(test)): 
  length = len(history) 
  lag = [history[i] for i in range(length-p,length)] 
  yhat = coef[0]  # Initialize to w0 
  for d in range(p): 
   yhat += coef[d+1] * lag[p-d-1]  # Add other values 
  obs = test[t] 
  predictions.append(yhat)  #Append  predictions  to  compute  RMSE later 
  history.append(obs)  # Append actual test value to history, to be used in next step.

'''part(b(i))'''
# scatter plot in b/w actual and predicted values
plt.scatter(predictions, test, s = 5, c = 'g')
plt.xlabel('Predicted Values', fontsize = 12)
plt.ylabel('Actual Values', fontsize = 12)
plt.show()

'''part(b(ii))'''
plt.plot(test, label = ['Actual data'])
plt.plot(predictions, label = ['Predicted data'])
plt.legend()
plt.xlabel('Time points')
plt.ylabel('New cases')
plt.show()

'''part(b(iii))'''
# calculating rmse
rmse_b_iii =( math.sqrt(mean_squared_error(test, predictions))/np.mean(test) )* 100
print("RMSE % is : ", format(rmse_b_iii, '.3f'))

'''part(b(iv))'''
# calculating mape
mape_b_iv = np.mean( np.abs( (test - predictions)/test) ) * 100
print("Mape is : ", format(mape_b_iv, '.3f'))


"""Question 3"""

# creating tuple and list to store lag values, rmse and mape values
lag_val2 = (1, 5, 10, 15, 25)
rmse_3 = list()
mape_3 = list()

# let's loop through each lag days and get the corresponding result by creating the model
for p in lag_val2:

    # creating the model
  ar_model = AR(train, lags=p)
  AR_model = ar_model.fit()
  coef = AR_model.params
  history = train[len(train)-p:] 
  history = [history[i] for i in range(len(history))] 

  predictions = list() # List to hold the predictions, 1 step at a time 

  for t in range(len(test)): 
    length = len(history) 
    lag = [history[i] for i in range(length-p,length)] 
    yhat = coef[0]  # Initialize to w0 
    for d in range(p): 
      yhat += coef[d+1] * lag[p-d-1]  # Add other values 
    obs = test[t] 
    predictions.append(yhat)  #Append  predictions  to  compute  RMSE later 
    history.append(obs)  # Append actual test value to history, to be used in next step.

  # RMSE calculation and storing in rmse_3
  rmse_calculated =( math.sqrt(mean_squared_error(test, predictions))/np.mean(test) )* 100
  rmse_3.append(rmse_calculated)

  # mape calculation and storing in mape_3
  mape_calculated = np.mean( np.abs( (test - predictions)/test) ) * 100
  mape_3.append(mape_calculated)

# Creating bar chart b/w rmse values and lag values
plt.bar(lag_val2, rmse_3)
plt.xlabel('Lag values')
plt.ylabel('Corresponding RMSE(%)')
plt.show()

# Creating bar chart b/w mape values and lag values
plt.bar(lag_val2, mape_3, color = 'g')
plt.xlabel('Lag values')
plt.ylabel('Corresponding Mape values')
plt.show()

"""Question 4"""

# calculating optimum value of lag
p = 1
train_data = list()
for i in range(train.shape[0]):
  train_data.append(train[i][0])

train_data = np.array(train_data)
while(p < len(train_data)):
  corr_value = np.corrcoef(train_data[:len(train_data) - p], train_data[p:])
  p = p + 1;
  if(abs(corr_value[0,1]) <= 2/math.sqrt(train_data.shape[0] - p)):
    print("The optimum value for lag is : ", p)
    break;


# Creating AR model with the above calculated optimum value of p
ar_model = AR(train, lags=p)
AR_model = ar_model.fit()
coef = AR_model.params
history = train[len(train)-p:] 
history = [history[i] for i in range(len(history))] 
predictions = list() # List to hold the predictions, 1 step at a time 

for t in range(len(test)): 
  length = len(history) 
  lag = [history[i] for i in range(length-p,length)] 
  yhat = coef[0]  # Initialize to w0 
  for d in range(p): 
    yhat += coef[d+1] * lag[p-d-1]  # Add other values 
  obs = test[t] 
  predictions.append(yhat)  #Append  predictions  to  compute  RMSE later 
  history.append(obs)  # Append actual test value to history, to be used in next step.

# RMSE calculation
rmse_calculated =( math.sqrt(mean_squared_error(test, predictions))/np.mean(test) )* 100
print('RMSE(%) is : ', format(rmse_calculated, '.3f'))

# MAPE calculation
mape_calculated = np.mean( np.abs( (test - predictions)/test) ) * 100
print('MAPE is : ', format(mape_calculated, '.3f'))