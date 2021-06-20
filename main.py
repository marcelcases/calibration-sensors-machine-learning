"""
Calibration of sensors in uncontrolled environments in
Air Pollution Sensor Monitoring Networks
TOML - Project 2
Marcel Cases
June 2021
"""

#%%
# General dependencies
import pandas as pd # for data handling
import matplotlib.pyplot as plt # for linear plot
import seaborn as sns # for scatter plot
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#%%
# Read sensor data
sensor = pd.read_csv("data.csv", sep=';', index_col=0, parse_dates=False)

# Build dataset
df = pd.DataFrame({'RefSt': sensor["RefSt"], 'Sensor_O3': sensor["Sensor_O3"], 'Temp': sensor["Temp"], 'RelHum': sensor["RelHum"]})

#%%
# Intro to Pandas
# Print first top lines from data
print(sensor.head(5))

#%%
# Print all data types
print(sensor.dtypes)

#%%
# Show data info summary
print(sensor.info())

#%%
# Select and print specific columns
Temp_Sensor_O3 = sensor[["Temp", "Sensor_O3"]]
print(Temp_Sensor_O3.head(5))

#%%
# Simple plot
df.plot()
plt.xticks(rotation=20)

#%%
# Data observation
# Plot the ozone (KOhms) and ozone reference data (μgr/m^3) as function of time
df[["Sensor_O3", "RefSt"]].plot()
plt.xticks(rotation=20)

# %%
# Plot the ozone (KOhms) and ozone reference data (μgr/m^3) as function of time - factor
Sensor_O3_RefSt_factor = df[["Sensor_O3", "RefSt"]]
Sensor_O3_RefSt_factor["RefSt"] = 4*Sensor_O3_RefSt_factor["RefSt"]
Sensor_O3_RefSt_factor.plot()
plt.xticks(rotation=20)

# %%
# Raw scatter plot
sns.lmplot(x = 'Sensor_O3', y = 'RefSt', data = df, fit_reg=True, line_kws={'color': 'orange'}) 

# %%
# Normalise sensor data
def normalize(col):
    μ = col.mean()
    σ = col.std()
    return (col - μ)/σ

df["normRefSt"] = normalize(df["RefSt"])
df["normSensor_O3"] = normalize(df["Sensor_O3"])
df["normTemp"] = normalize(df["Temp"])
df["normRelHum"] = normalize(df["RelHum"])

print(df)

# %%
# Normalised scatter plot
sns.lmplot(x = 'normSensor_O3', y = 'normRefSt', data = df, fit_reg=True, line_kws={'color': 'orange'}) 

# %%
# Temp with respect to Sensor_O3
sns.lmplot(x = 'Sensor_O3', y = 'Temp', data = df, fit_reg=True, line_kws={'color': 'orange'}) 

# %%
# Temp with respect to RefSt
sns.lmplot(x = 'RefSt', y = 'Temp', data = df, fit_reg=True, line_kws={'color': 'orange'}) 

# %%
# RelHum with respect to Sensor_O3
sns.lmplot(x = 'Sensor_O3', y = 'RelHum', data = df, fit_reg=True, line_kws={'color': 'orange'}) 

# %%
# RelHum with respect to RefSt
sns.lmplot(x = 'RefSt', y = 'RelHum', data = df, fit_reg=True, line_kws={'color': 'orange'}) 

# %%
# Data calibration
# Multiple Linear Regression
from sklearn import linear_model

X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

df["MLR_Pred"] = -34.0316709 + 0.15929287*df["Sensor_O3"] + 2.49694134*df["Temp"] - 0.02949471*df["RelHum"]
df[["RefSt", "MLR_Pred"]].plot()
plt.xticks(rotation=20)

#%%
sns.lmplot(x = 'RefSt', y = 'MLR_Pred', data = df, fit_reg=True, line_kws={'color': 'orange'}) 

#%%
# Loss functions definition
def loss_functions(y_true, y_pred):
    print("Loss functions:")
    print("* R-squared =", r2_score(y_true, y_pred))
    print("* RMSE =", mean_squared_error(y_true, y_pred))
    print("* MAE =", mean_absolute_error(y_true, y_pred))

#%%
# MLR loss
loss_functions(y_true=df["RefSt"], y_pred=df["MLR_Pred"])

# %%
# K-Nearest Neighbor
from sklearn.neighbors import KNeighborsRegressor

X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']

# fit
neigh = KNeighborsRegressor(n_neighbors=31)
neigh.fit(X, Y)

# predict
df["KNN_Pred"] = neigh.predict(df[['Sensor_O3', 'Temp', 'RelHum']])
print(df)

# plot linear
df[["RefSt", "KNN_Pred"]].plot()
plt.xticks(rotation=20)

# plot regression
sns.lmplot(x = 'RefSt', y = 'KNN_Pred', data= df, fit_reg=True, line_kws={'color': 'orange'}) 

# KNN loss
loss_functions(y_true=df["RefSt"], y_pred=df["KNN_Pred"])


# %%
# Random Forest
from sklearn.ensemble import RandomForestRegressor

X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']

# fit
rf=RandomForestRegressor(n_estimators=30,random_state=0)
rf.fit(X, Y)

# predict
df["RF_Pred"] = rf.predict(X)
print(df)

# plot linear
df[["RefSt", "RF_Pred"]].plot()
plt.xticks(rotation=20)

# plot regression
sns.lmplot(x = 'RefSt', y = 'RF_Pred', data= df, fit_reg=True, line_kws={'color': 'orange'}) 

# RF loss
loss_functions(y_true=df["RefSt"], y_pred=df["RF_Pred"])
print('Feature importances:\n', list(zip(X.columns, rf.feature_importances_)))

# %%
# Random Forest stats vs. hyperparameters
import datetime

def rf_stats():
    X = df[['Sensor_O3', 'Temp', 'RelHum']]
    Y = df['RefSt']

    rf_aux = pd.DataFrame({'RefSt': sensor["RefSt"]})

    n_estimators = [*range(1, 101, 1)]
    r_squared = []
    rmse = []
    mae = []
    rf_time_ms = []

    for i in n_estimators:
        rf=RandomForestRegressor(n_estimators=i,random_state=0)

        # fit
        start_time = datetime.datetime.now()
        rf.fit(X, Y)
        end_time = datetime.datetime.now()

        # predict
        rf_aux["RF_Pred"] = rf.predict(X)

        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000
    
        # RF loss
        r_squared.append(r2_score(rf_aux["RefSt"], rf_aux["RF_Pred"]))
        rmse.append(mean_squared_error(rf_aux["RefSt"], rf_aux["RF_Pred"]))
        mae.append(mean_absolute_error(rf_aux["RefSt"], rf_aux["RF_Pred"]))
        rf_time_ms.append(execution_time)

    rf_stats = pd.DataFrame({'n_estimators': n_estimators, 'r_squared': r_squared, 'rmse': rmse, 'mae': mae, 'rf_time_ms': rf_time_ms})
    rf_stats = rf_stats.set_index('n_estimators') # index column (X axis for the plots)
    print(rf_stats)

    # plot
    rf_stats[["r_squared"]].plot()
    rf_stats[["rmse"]].plot()
    rf_stats[["mae"]].plot()
    rf_stats[["rf_time_ms"]].plot()

# rf_stats()

# %%
# Gaussian Process
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=rbf, alpha=0)

X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']

# fit
gp.fit(X, Y)

# predict
df["GP_Pred"] = gp.predict(X)

# Obtain optimized kernel parameters
l = gp.kernel_.k2.get_params()['length_scale']
sigma_f = np.sqrt(gp.kernel_.k1.get_params()['constant_value'])

# plot linear
df[["RefSt", "GP_Pred"]].plot()
plt.xticks(rotation=20)

# plot regression
sns.lmplot(x = 'RefSt', y = 'GP_Pred', data= df, fit_reg=True, line_kws={'color': 'orange'}) 

# GP loss
loss_functions(y_true=df["RefSt"], y_pred=df["GP_Pred"])


# %%
# Neural Network
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, InputLayer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)
X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.08, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
model = Sequential()

model.add(InputLayer(input_shape=(3))) # Input layer
model.add(Dense(units = 64, activation = 'relu')) # 1st hidden layer
model.add(Dense(units = 64, activation = 'relu')) # 2nd hidden layer
model.add(Dense(units = 64, activation = 'relu')) # 3rd hidden layer
model.add(Dense(units = 64, activation = 'relu')) # 4th hidden layer
model.add(Dense(units = 64, activation = 'relu')) # 5th hidden layer

model.add(Dense(units = 1)) # Output layer

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, Y_train, batch_size = 10, epochs = 200)

y_pred = model.predict(df[['normSensor_O3', 'normTemp', 'normRelHum']])
df["NN_Pred"] = y_pred
print(df)

#%%
# plot linear
df[["RefSt", "NN_Pred"]].plot()
plt.xticks(rotation=20)

#%%
# plot regression
sns.lmplot(x = 'RefSt', y = 'NN_Pred', data = df, fit_reg=True, line_kws={'color': 'orange'}) 

#%%
# NN loss
loss_functions(y_true=df["RefSt"], y_pred=df["NN_Pred"])



# %%
