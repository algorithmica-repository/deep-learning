import math 
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
from statsmodels.tsa import ar_model
print(statsmodels.__version__)

def grid_search_best_model_timeseries_ar(df, grid, cv):
    best_param = None
    best_score = np.infty
    tsp = TimeSeriesSplit(n_splits=cv)
            
    for param in grid.get('lags'):
        scores = []
        for train_ind, test_ind in tsp.split(df):
            train_data = df.iloc[train_ind]
            test_data = df.iloc[test_ind]
            try:
                #print(train_data, test_data)
                estimator = ar_model.AutoReg(train_data, lags=param)
                res = estimator.fit() 
                #print(res.params)
                #get out of sample predictions with test data start and end
                pred = estimator.predict(res.params, test_data.index[0], test_data.index[-1])
                #print(pred)
                y_pred = pred.values.reshape(-1)
                y_test = test_data.values.reshape(-1)
                score = math.sqrt(metrics.mean_squared_error(y_test,  y_pred))
                scores.append(score)
            except:
                pass
        #print(scores)
        if len(scores) > 0  and np.mean(scores) < best_score :
            best_score = np.mean(scores)
            best_param = param
        
    if best_param is not None:
        estimator = ar_model.AutoReg(df, lags=best_param)
        res = estimator.fit()
        print("best parameters:" + str(best_param))
        print("validation rmse:" +  str(best_score))
        #get insample predictions with start and end indices
        predictions = estimator.predict(res.params, start=0, end=df.shape[0]-1 )
        y_pred = predictions.values.reshape(-1)
        y_train = df.values.reshape(-1)[best_param:]
        train_rmse = math.sqrt(metrics.mean_squared_error(y_train,  y_pred))
        print("train rmse:" + str(train_rmse))
        return estimator, res
    else:
        return None, None
 
path = 'F:/'
data = pd.read_excel(os.path.join(path, 'energy.xlsx'))
data.info()
data.head()

data['timestamp'] = data['Date'].add(pd.to_timedelta(data.Hour - 1, unit='h'))
data.head()

data = data[['timestamp', 'load', 'T']]
data.head()

data = data.rename(columns={'T':'temp'})
data.head()

data = data[data.timestamp >= '2006-01-01']
data.head()

data.index = data['timestamp']
data.head()
data = data.drop('timestamp', axis=1)
data.head()

#univariate time series data
energy = data[['load']]
energy.head()

energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()

#hold out data fro validationa and test data
valid_start_dt = '2014-09-01 00:00:00'
test_start_dt = '2014-11-01 00:00:00'

energy_train = energy[energy.index < valid_start_dt]
energy_validation = energy[(energy.index >= valid_start_dt) & (energy.index < test_start_dt)]
energy_test = energy[energy.index >= test_start_dt]

scaler = MinMaxScaler()
energy_train['load_scaled'] = scaler.fit_transform(energy_train['load'])
energy_train.head(10)

#plot both
sns.distplot(energy_train['load'])
sns.distplot(energy_train['load_scaled'])

energy_train1 = energy_train.copy()
energy_train1 = energy_train1.drop('load', axis=1)
energy_train1.index.freq = 'H'

#build model
estimator = ar_model.AutoReg(energy_train1, lags=5)
res = estimator.fit()
print(res.params)

energy_validation['load_scaled'] = scaler.fit_transform(energy_validation)
energy_validation1 = energy_validation.copy()
energy_validation1 = energy_validation1.drop('load', axis=1)
energy_validation1.index.freq = 'H'

#valdiation error
pred = estimator.predict(res.params, energy_validation1.index[0], energy_validation1.index[-1])
print(pred)
y_pred = pred.values.reshape(-1)
y_test = energy_validation1.values.reshape(-1)
score = math.sqrt(metrics.mean_squared_error(y_test,  y_pred))
print(score)

#train error
pred = estimator.predict(res.params, 0, energy_train1.shape[0]-1)
print(pred)
y_pred = pred.values.reshape(-1)
y_test = energy_train1.values.reshape(-1)[5:]
score = math.sqrt(metrics.mean_squared_error(y_test,  y_pred))
print(score)

index = pd.date_range('1-9-2014', '12-12-2014', freq='H')
pred = estimator.predict(res.params, index[0], index[-1])
print(pred)

plt.figure()
plt.plot(pred)
pred = pred.values.reshape(-1,1)
scaler.inverse_transform(pred)
