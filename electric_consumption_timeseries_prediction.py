# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import warning library
import warnings
warnings.filterwarnings('ignore')

#Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Define visualization style (Given style is highly useful for visualizing time series data)
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7

#Load the dataset
df = pd.read_csv("Electric_Consumption.csv")

#Define column names and drop nulls
df.columns = ['Date', 'Consumption']
df=df.dropna()

#Convert date to date and time and make date column as index column
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True) 

#Visualizing the time series data
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.title("Consumption graph")
plt.plot(df)

#Plot scatter plot
df.plot(style='k.')
plt.show()

#We may need to separate seasonality and trend from our series to make seies stationary
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df, model='multiplicative')
result.plot()
plt.show()

"""We will use Augmented Dickery-Fuller Test to check if the series is stationary or not. If we fail to reject null hypothesis,
we can say series is not stationary. If mean and standard deviation are flat, series is staionary."""

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    #Plotting rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    
    #Performing ADFT  
    print("Results of dickey fuller test")
    adft = adfuller(timeseries['Consumption'],autolag='AIC')
    
    #Output for dft will give us values without defining what they are, hence we manually write what values does it explains
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

#Call the function    
test_stationarity(df)

"""Here rolling mean and standard deviation is increasing, thus series is increasing. ALso, p-value > 0.005, so we can't reject
null hypothesis. Also, test stat is greater than critical value, so data series is not stationary"""

#To reduce the magnitudes, we will take log of series and try to reduce rising trends. We will find rolling average for that.
df_log = np.log(df)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.plot(df_log)
plt.plot(moving_avg, color="red")
plt.plot(std_dev, color ="black")
plt.show()

#We take difference of series and mean at every point, thus eliminate trends out of series
df_log_moving_avg_diff = df_log-moving_avg
df_log_moving_avg_diff.dropna(inplace=True)

#Now we will perform ADFT to validate the step
test_stationarity(df_log_moving_avg_diff)

#The result shows data is stationary. Now we will try to find weighted average to understand trend of data in time series.
weighted_average = df_log.ewm(halflife=12, min_periods=0,adjust=True).mean()

"""We will subtract df_log with weighted average now to get idea of Exponential Moving Average(EMA).
It gives recent price more weight than past price."""
logScale_weightedMean = df_log-weighted_average
from pylab import rcParams
rcParams['figure.figsize'] = 10,6
test_stationarity(logScale_weightedMean) 

#The result shows that data is attained stationary; aslo, test and critical values are relatively equal.

"""There can be case of high seasonality of data. Thus, just removing trend does not help much, we need to take care of seasonality.
Differencing is method used to remove series dependency on time. It helps in stabilizing the series and mean by removing changes in
level of time series, thus removing seasonality and trends. It is subtracting previous observation from current observation"""

df_log_diff = df_log - df_log.shift()
plt.title("Shifted timeseries")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.plot(df_log_diff)
#Let us test the stationarity of our resultant series
df_log_diff.dropna(inplace=True)

#Validate by ADFT
test_stationarity(df_log_diff)

#Now, we will perform decomposition which provide structured way of thinking about forecasting time series data. 
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df_log, model='additive', freq = 12)
result.plot()
plt.show()

#Dropping null values in trends
trend = result.trend
trend.dropna(inplace=True)

#Dropping null values in seasonality
seasonality = result.seasonal
seasonality.dropna(inplace=True)

#Dropping null values in Residual
residual = result.resid
residual.dropna(inplace=True)

#Perform ADFT to validate
test_stationarity(residual)

#The result shows that mean and std dev are almost flat line. We have stationary model and can find the best parameters of model

#We need Autocorrelation Function and Partial Autocorrelation Function plots to determine the optimal parameters.

"""As the correlation of the time series observations is calculated with values of the same series at previous times,
this is called an autocorrelation. The partial autocorrelation at lag k is the correlation that results after removing 
the effect of any correlations due to the terms at shorter lags. The autocorrelation for observation and observation at a 
prior time step is comprised of both the direct correlation and indirect correlations. It is these indirect correlations 
that the partial autocorrelation function seeks to remove."""

from statsmodels.tsa.stattools import acf,pacf
# we use d value here(data_log_shift)
acf = acf(df_log_diff, nlags=15)
pacf= pacf(df_log_diff, nlags=15,method='ols')


#plot ACF
plt.subplot(121)
plt.plot(acf) 
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.title('Auto corellation function')
plt.tight_layout()
#plot PACF
plt.subplot(122)
plt.plot(pacf) 
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.title('Partially auto corellation function')
plt.tight_layout()

#Now we will fit the model
"""In order to find the p and q values from the above graphs, we need to check, where the graph cuts off the origin or 
drops to zero for the first time from the above graphs the p and q values are merely close to 3"""

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_log, order=(3,1,3))
result_AR = model.fit(disp = 0)
plt.plot(df_log_diff)
plt.plot(result_AR.fittedvalues, color='red')
plt.title("sum of squares of residuals")
print('RSS : %f' %sum((result_AR.fittedvalues-df_log_diff["Consumption"])**2)) 
plt.close()
#Lesser the RSS value, more effective the model
try:
    forecast = int(input("Enter the year upto which you want to forecast: "))
except ValueError:
    print(" ")

num = int((forecast - int(1985)) * 12)
#Predict forecast for next 5 years
result_AR.plot_predict(1,num)
x=result_AR.forecast(steps=200)
plt.savefig('output.png')
#Thus we have calculated the future prediction till 2024 and confidence grey interval is area under which prediction will lie.