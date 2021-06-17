"""
Calibration of sensors in uncontrolled environments in
Air Pollution Sensor Monitoring Networks
TOML - Project 2
Marcel Cases
June 2021
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
sensor = pd.read_csv("data.csv", sep=';', index_col=0, parse_dates=False)


# Intro to Pandas

#%%
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
Temp_Sensor_O3.plot()


# Data observation

#%%
# Plot the ozone (KOhms) and ozone reference data (μgr/m^3) as function of time
Sensor_O3_RefSt = sensor[["Sensor_O3", "RefSt"]]
Sensor_O3_RefSt.plot()
plt.xticks(rotation=20)

# %%
# Plot the ozone (KOhms) and ozone reference data (μgr/m^3) as function of time - normalised
Sensor_O3_RefSt_norm = sensor[["Sensor_O3", "RefSt"]]
Sensor_O3_RefSt_norm["RefSt"] = 4*Sensor_O3_RefSt_norm["RefSt"]
Sensor_O3_RefSt_norm.plot()

# %%
# Raw scatter plot
df = pd.DataFrame({'RefSt': sensor["RefSt"], 'Sensor_O3': sensor["Sensor_O3"]})
df.plot.scatter(x = 'Sensor_O3', y = 'RefSt')


# %%
# Normalised scatter plot
μSensor_O3 = sensor["Sensor_O3"].mean()
σSensor_O3 = sensor["Sensor_O3"].std()
normSensor_O3 = (sensor["Sensor_O3"] - μSensor_O3)/σSensor_O3

print(μSensor_O3, σSensor_O3, normSensor_O3)

μRefSt = sensor["RefSt"].mean()
σRefSt = sensor["RefSt"].std()
normRefSt = (sensor["RefSt"] - μRefSt)/σRefSt

print(μRefSt, σRefSt, normRefSt)

df = pd.DataFrame({'normRefSt': normRefSt, 'normSensor_O3': normSensor_O3})
df.plot.scatter(x = 'normSensor_O3', y = 'normRefSt')

# %%
# Temp with respect to Sensor_O3
df = pd.DataFrame({'Temp': sensor["Temp"], 'Sensor_O3': sensor["Sensor_O3"]})
df.plot.scatter(x = 'Sensor_O3', y = 'Temp')

# %%
# Temp with respect to RefSt
df = pd.DataFrame({'Temp': sensor["Temp"], 'RefSt': sensor["RefSt"]})
df.plot.scatter(x = 'RefSt', y = 'Temp')

# %%
# RelHum with respect to Sensor_O3
df = pd.DataFrame({'RelHum': sensor["RelHum"], 'Sensor_O3': sensor["Sensor_O3"]})
df.plot.scatter(x = 'Sensor_O3', y = 'RelHum')

# %%
# RelHum with respect to RefSt
df = pd.DataFrame({'RelHum': sensor["RelHum"], 'RefSt': sensor["RefSt"]})
df.plot.scatter(x = 'RefSt', y = 'RelHum')

# %%
# Multiple Linear Regression
from sklearn import linear_model
df = pd.DataFrame({'Sensor_O3': sensor["Sensor_O3"], 'RefSt': sensor["RefSt"], 'Temp': sensor["Temp"], 'RelHum': sensor["RelHum"] })
X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# Pred = lambda Sensor_O3, Temp, RelHum: -34.0316709 + 0.15929287*Sensor_O3 + 2.49694134*Temp - 0.02949471*RelHum
MLR = sensor[["Sensor_O3", "RefSt", "Temp", "RelHum"]]
MLR["MLR_Pred"] = -34.0316709 + 0.15929287*sensor["Sensor_O3"] + 2.49694134*sensor["Temp"] - 0.02949471*sensor["RelHum"]
print(MLR)
MLR_plot = MLR[["RefSt", "MLR_Pred"]]
MLR_plot.plot()
plt.xticks(rotation=20)

#%%
# To plot the regression line over the scatter plot
import seaborn as sns
df = pd.DataFrame({'RefSt': MLR["RefSt"], 'MLR_Pred': MLR["MLR_Pred"]})
sns.lmplot(x = 'RefSt', y = 'MLR_Pred', data= df, fit_reg=True, line_kws={'color': 'orange'}) 

#%%
# Loss functions
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
def loss_functions(y_true, y_pred):
    print("Loss functions:")
    print("* R-squared =", r2_score(y_true, y_pred))
    print("* RMSE =", mean_squared_error(y_true, y_pred))
    print("* MAE =", mean_absolute_error(y_true, y_pred))

loss_functions(MLR["RefSt"], MLR["MLR_Pred"])
# %%
