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
Sensor_O3_RefSt = df[["Sensor_O3", "RefSt"]]
Sensor_O3_RefSt.plot()
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
RefSt_MLR_Pred = df[["RefSt", "MLR_Pred"]]
RefSt_MLR_Pred.plot()
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
loss_functions(df["RefSt"], df["MLR_Pred"])

# %%
# K-nearest Neighbor
X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']

# fit
neigh = KNeighborsRegressor(n_neighbors=31)
neigh.fit(X, Y)

# predict
df["KNN_Pred"] = neigh.predict(df[['Sensor_O3', 'Temp', 'RelHum']])
print(df)

#plot linear
RefSt_KNN_Pred = df[["RefSt", "KNN_Pred"]]
RefSt_KNN_Pred.plot()
plt.xticks(rotation=20)

# plot regression
sns.lmplot(x = 'RefSt', y = 'KNN_Pred', data= df, fit_reg=True, line_kws={'color': 'orange'}) 

# KNN loss
loss_functions(df["RefSt"], df["KNN_Pred"])

# %%
