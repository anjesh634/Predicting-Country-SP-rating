# Predicting country's S&P rating

#### <B>Note</B>

> Before Runing the code please make sure you have these libraries in your local environment. There are some codes which provides huge number of outputs. Hence it is recommended to run on local environment not on google colab  

- pandas 
- numpy
- matplotlib
- seaborn
- csv
- time
- plotly
- sklearn
- statsmodels

# Table of Content 
1. Discription
2. Data Understanding 
3. Problem faced and solution approch
4. Web Scraping
5. Data Preprocessing 
6. Model Building
    - Bayes Classification
    - Time series AR model
7. Result 
8. Interpretation section
9. Refrences

## 1. Discription

### What is S&P Rating 
- S&P Global Ratings is an American credit rating agency and a division of S&P Global that publishes financial research and analysis on stocks, bonds, and commodities. S&P is considered the largest of the Big Three credit-rating agencies, which also include Moody's Investors Service and Fitch Ratings


## 2. Data Understanding

#### Attribute 1 - Rating 

| Rating | Type | - |
| --- | --- | --- |
| AAA| Investment | Extremely strong    |
| AA+, AA, AA-  | Investment | Very Strong  |
| BBB+, BBB, BBB-  | Investment | Strong  |
| BB+, BB, BB-  | Speculative | Adequate  |
| B+, B, B-  | Speculative | Faces major future uncertainties  |
| CCC | Speculative | Currently vulnerable  |
| CC | Speculative | Currently highly vulnerable   |
| C | Speculative | Has filed bankruptcy petition   |
| D | Speculative | In defaulf  |





| Rating | Numeric Rating | Type |
| ---- | ---- | ---- |
| AAA | 22 | Top Notch for investement  |
| AA+ | 21 | Invest UNDER OBSERVATION  |
| AA | 20 | Invest UNDER OBSERVATION  |
| AA- | 19 | Invest UNDER OBSERVATION  |
| A+ | 18 | Invest UNDER OBSERVATION  |
| A | 17 | Invest UNDER OBSERVATION  |
| A- | 16 | Invest UNDER OBSERVATION  |
| BBB+ | 15 | Invest UNDER OBSERVATION  |
| BBB | 14 | Invest UNDER OBSERVATION  |
| BBB- | 13 | Invest UNDER OBSERVATION  |
| BB+ | 12 | Bad for investment  |
| BB | 11 | Bad for investment  |
| BB- | 10 | Bad for investment  |
| B+ | 9 | Bad for investment  |
| B | 8 | Bad for investment  |
| B- | 7 | Bad for investment  |
| CCC+ | 6 | Bad for investment  |
| CCC | 5 | Bad for investment  |
| CCC- | 4 | Bad for investment  |
| CC | 3 | Bad for investment  |
| C | 2 | Bad for investment |
| D | 1 | Bad for investment  |


#### Attribute 2 - Outlook

| Outlook type | Interpretation | 
| ----- | ----- |
| Positive | rating may be raised next month  | 
| Negative | rating may be lowered next month | 
| Stable | rating is not likely to change  | 


#### Attribute 3 - Date
- This attribute specify the date at which rating is catlulated



#### Example 

-  For any country with rating BBB in 2009 means, that particular country will repay the debt has a chance of 0.55% 

| Year |   AAA  |    AA  |    A |     BBB |     BB |     B  |    CCC/C |
| --- | --- | --- | --- | --- | --- | --- | --- | 
|2009  |    0.00  |  0.00  |  0.22  |  0.55  |  0.75  |  11.01  |  49.46 |
|2010  |   0.00   | 0.00  |  0.00  |  0.00  |  0.58  |  0.87   | 22.73 |
|2011  |  0.00   | 0.00  |  0.00  |  0.07  |  0.00  |  1.68   | 16.42 |
|2012  |  0.00   | 0.00  |  0.00  |  0.00  |  0.30  |  1.58   | 27.52 |
|2013  |  0.00   | 0.00  |  0.00  |  0.00  |  0.10  |  1.65   | 24.67 |
|2014  |  0.00   | 0.00  |  0.00  |  0.00  |  0.00  |  0.78   | 17.51 |
|2015  |  0.00   | 0.00  |  0.00  |  0.00  |  0.16  |  2.41   | 26.67 |
|2016  |  0.00   | 0.00  |  0.00  |  0.06  |  0.47  |  3.75   | 33.33 |


## 3. Problem faced and solution approch

 - Very less number of data were available for applying any kind of machine learning model. So we used web scraping techinque to get more data from offical website of S&P Global rating 
 
 Example - India 
 
 https://tradingeconomics.com/india/rating
 
 
 # Time Series Forcasting Model

### Diffrent Types of time series model

1. Autoregression Models (AR Models) 
2. Moving averages Models (MA Models)
3. Autoregressive Moving Averages (ARMA Models)
4. Autoregressive Integrated Moving Averages (ARIMA Models)

### What is time series and why to use?

- A time series is a set of observations taken at a specified time interval


### Checking Stationarity
- Constant Mean 
- Constant Variance 
- No Seasonality

   If any time series data set satisfy above mentioned pionts then it is stationart and time series model can be used on it

### Time Series Analysis can be used in:

- Economic Forecasting
- Sales Forecasting
- Budgetary Analysis
- Stock Market Analysis
- Yield Projections
- Inventory assessments
- Workload projections
- Demographics projections
- Weather patterns and forecasts


### AutoRegression Model
AR Model of order 1
$$  
         \Upsilon_{t} = \beta_{0}+ \beta_{1}*\Upsilon _{t-1}
$$

AR Model of order 2

$$
        \Upsilon_{t} = \beta_{0}+ \beta_{1}*\Upsilon _{t-1} + \beta_{2}*\Upsilon _{t-2}
$$
AR Model of order p

$$
        \Upsilon_{t} = \beta_{0}+ \beta_{1}*\Upsilon _{t-1} + \beta_{2}*\Upsilon _{t-2}  . . . . + \beta_{p}*\Upsilon _{t-p}
$$


- <B>Stationarity of the time-series data</B>: The stationarity of the data can be found using adfuller class of statsmodels.tsa.stattools module. The value of p-value is used to determine whether there is stationarity. If the value is less than 0.05, the stationarity exists.
- <B>Order of AR model to be trained</B>: The order of AR model is determined by checking the partial autocorrelation plot. The plot_pacf method of statsmodels.graphics.tsaplots is used to plot.

 



