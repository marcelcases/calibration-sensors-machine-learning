"""
Calibration of sensors in uncontrolled environments in
Air Pollution Sensor Monitoring Networks
TOML - Project 2
Marcel Cases
June 2021
"""

#%%
# Dependencies
import pandas as pd # for data handling
import matplotlib.pyplot as plt # for linear plot
import seaborn as sns # for scatter plot
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

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






# %%
# Neural Network
import tensorflow as tf
from tensorflow import keras
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
