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
from sklearn.model_selection import train_test_split
import datetime


#%%
# Read sensor data
sensor = pd.read_csv("data.csv", sep=';', index_col=0, parse_dates=False)

# Build main dataset
df = pd.DataFrame({'RefSt': sensor["RefSt"], 'Sensor_O3': sensor["Sensor_O3"], 'Temp': sensor["Temp"], 'RelHum': sensor["RelHum"]})

# Split main dataset and build train and test datasets
X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1, shuffle=False)

df_train = pd.DataFrame({'RefSt': Y_train, 'Sensor_O3': X_train["Sensor_O3"], 'Temp': X_train["Temp"], 'RelHum': X_train["RelHum"]})
df_test = pd.DataFrame({'RefSt': Y_test, 'Sensor_O3': X_test["Sensor_O3"], 'Temp': X_test["Temp"], 'RelHum': X_test["RelHum"]})


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


#%%
# Loss functions definition
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def loss_functions(y_true, y_pred):
    print("Loss functions:")
    print("* R-squared =", r2_score(y_true, y_pred))
    print("* RMSE =", mean_squared_error(y_true, y_pred))
    print("* MAE =", mean_absolute_error(y_true, y_pred))


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
from sklearn.linear_model import LinearRegression

# Model
lr = LinearRegression()

# Fit
lr.fit(X_train, Y_train)

# Get MLR coefficients
print('Intercept: \n', lr.intercept_)
print('Coefficients: \n', lr.coef_)

# Predict
df_test["MLR_Pred"] = lr.intercept_ + lr.coef_[0]*df_test["Sensor_O3"] + lr.coef_[1]*df_test["Temp"] - lr.coef_[2]*df_test["RelHum"]

# Plot linear
df_test[["RefSt", "MLR_Pred"]].plot()
plt.xticks(rotation=20)

# Plot regression
sns.lmplot(x = 'RefSt', y = 'MLR_Pred', data = df_test, fit_reg=True, line_kws={'color': 'orange'}) 

# MLR loss
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["MLR_Pred"])


# %%
# Multiple Linear Regression with Batch Gradient Descent



# %%
# Multiple Linear Regression with Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Model
sgdr = SGDRegressor()

# Normalize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit
sgdr.fit(X_train, Y_train)

# Get MLR coefficients
print('Intercept: \n', sgdr.intercept_)
print('Coefficients: \n', sgdr.coef_)

# Predict
# df_test["MLR_SGDR_Pred"] = sgdr.intercept_ + sgdr.coef_[0]*X_test[0] + sgdr.coef_[1]*X_test[1] - sgdr.coef_[2]*X_test[2]
df_test["MLR_SGDR_Pred"] = sgdr.predict(X_test)

# Plot linear
df_test[["RefSt", "MLR_SGDR_Pred"]].plot()
plt.xticks(rotation=20)

# Plot regression
sns.lmplot(x = 'RefSt', y = 'MLR_SGDR_Pred', data = df_test, fit_reg=True, line_kws={'color': 'orange'}) 

# MLR loss
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["MLR_SGDR_Pred"])


# %%
# K-Nearest Neighbor
from sklearn.neighbors import KNeighborsRegressor

# fit
knn = KNeighborsRegressor(n_neighbors=19)
knn.fit(X_train, Y_train)

# predict
df_test["KNN_Pred"] = knn.predict(X_test)
print(df_test)

# plot linear
df_test[["RefSt", "KNN_Pred"]].plot()
plt.xticks(rotation=20)

# plot regression
sns.lmplot(x = 'RefSt', y = 'KNN_Pred', data= df_test, fit_reg=True, line_kws={'color': 'orange'}) 

# KNN loss
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["KNN_Pred"])


# %%
# K-Nearest Neighbor stats vs. hyperparameters
def knn_stats():
    knn_aux = pd.DataFrame({'RefSt': Y_test})

    n_neighbors = [*range(1, 151, 1)]
    r_squared = []
    rmse = []
    mae = []
    time_ms = []

    for i in n_neighbors:
        knn = KNeighborsRegressor(n_neighbors=i)

        # fit
        start_time = float(datetime.datetime.now().strftime('%S.%f'))
        knn.fit(X_train, Y_train)
        end_time = float(datetime.datetime.now().strftime('%S.%f'))
        execution_time = (end_time - start_time) * 1000

        # predict
        knn_aux["KNN_Pred"] = knn.predict(X_test)

            # KNN loss
        r_squared.append(r2_score(knn_aux["RefSt"], knn_aux["KNN_Pred"]))
        rmse.append(mean_squared_error(knn_aux["RefSt"], knn_aux["KNN_Pred"]))
        mae.append(mean_absolute_error(knn_aux["RefSt"], knn_aux["KNN_Pred"]))
        time_ms.append(execution_time)

    knn_stats = pd.DataFrame({'n_neighbors': n_neighbors, 'r_squared': r_squared, 'rmse': rmse, 'mae': mae, 'time_ms': time_ms})
    knn_stats = knn_stats.set_index('n_neighbors') # index column (X axis for the plots)
    print(knn_stats)

    # plot
    knn_stats[["r_squared"]].plot()
    knn_stats[["rmse"]].plot()
    knn_stats[["mae"]].plot()
    knn_stats[["time_ms"]].plot()

knn_stats()


# %%
# Random Forest
from sklearn.ensemble import RandomForestRegressor

# fit
rf=RandomForestRegressor(n_estimators=20,random_state=0)
rf.fit(X_train, Y_train)

# predict
df_test["RF_Pred"] = rf.predict(X_test)
print(df_test)

# plot linear
df_test[["RefSt", "RF_Pred"]].plot()
plt.xticks(rotation=20)

# plot regression
sns.lmplot(x = 'RefSt', y = 'RF_Pred', data= df_test, fit_reg=True, line_kws={'color': 'orange'}) 

# RF loss
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["RF_Pred"])

# RF feature importances
print('Feature importances:\n', list(zip(X.columns, rf.feature_importances_)))


# %%
# Random Forest stats vs. hyperparameters
def rf_stats():
    rf_aux = pd.DataFrame({'RefSt': Y_test})

    n_estimators = [*range(1, 101, 1)]
    r_squared = []
    rmse = []
    mae = []
    time_ms = []

    for i in n_estimators:
        rf=RandomForestRegressor(n_estimators=i,random_state=0)

        # fit
        start_time = float(datetime.datetime.now().strftime('%S.%f'))
        rf.fit(X_train, Y_train)
        end_time = float(datetime.datetime.now().strftime('%S.%f'))
        execution_time = (end_time - start_time) * 1000

        # predict
        rf_aux["RF_Pred"] = rf.predict(X_test)

        # RF loss
        r_squared.append(r2_score(rf_aux["RefSt"], rf_aux["RF_Pred"]))
        rmse.append(mean_squared_error(rf_aux["RefSt"], rf_aux["RF_Pred"]))
        mae.append(mean_absolute_error(rf_aux["RefSt"], rf_aux["RF_Pred"]))
        time_ms.append(execution_time)

    rf_stats = pd.DataFrame({'n_estimators': n_estimators, 'r_squared': r_squared, 'rmse': rmse, 'mae': mae, 'time_ms': time_ms})
    rf_stats = rf_stats.set_index('n_estimators') # index column (X axis for the plots)
    print(rf_stats)

    # plot
    rf_stats[["r_squared"]].plot()
    rf_stats[["rmse"]].plot()
    rf_stats[["mae"]].plot()
    rf_stats[["time_ms"]].plot()

rf_stats()

# %%
# Gaussian Process
# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
# gp = GaussianProcessRegressor(kernel=rbf, alpha=0)

# X = df[['Sensor_O3', 'Temp', 'RelHum']]
# Y = df['RefSt']

# # fit
# gp.fit(X, Y)

# # predict
# df["GP_Pred"] = gp.predict(X)

# # Obtain optimized kernel parameters
# l = gp.kernel_.k2.get_params()['length_scale']
# sigma_f = np.sqrt(gp.kernel_.k1.get_params()['constant_value'])

# # plot linear
# df[["RefSt", "GP_Pred"]].plot()
# plt.xticks(rotation=20)

# # plot regression
# sns.lmplot(x = 'RefSt', y = 'GP_Pred', data= df, fit_reg=True, line_kws={'color': 'orange'}) 

# # GP loss
# loss_functions(y_true=df["RefSt"], y_pred=df["GP_Pred"])


# %%
# Neural Network
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, InputLayer
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler

print(tf.__version__)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model
nn = Sequential()

# Model - Layers
nn.add(InputLayer(input_shape=(3))) # Input layer
nn.add(Dense(units = 64, activation = 'relu')) # 1st hidden layer
nn.add(Dense(units = 64, activation = 'relu')) # 2nd hidden layer
nn.add(Dense(units = 64, activation = 'relu')) # 3rd hidden layer
nn.add(Dense(units = 64, activation = 'relu')) # 4th hidden layer
nn.add(Dense(units = 64, activation = 'relu')) # 5th hidden layer
nn.add(Dense(units = 1)) # Output layer

nn.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit
history = nn.fit(X_train, Y_train, batch_size = 10, epochs = 200)

# Plot loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])

# Predict
df_test["NN_Pred"] = nn.predict(X_test)
print(df_test)

# Plot linear
df_test[["RefSt", "NN_Pred"]].plot()
plt.xticks(rotation=20)

# Plot regression
sns.lmplot(x = 'RefSt', y = 'NN_Pred', data = df_test, fit_reg=True, line_kws={'color': 'orange'}) 

# NN loss
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["NN_Pred"])


# %%
