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

# %%
# Plot the ozone (KOhms) and ozone reference data (μgr/m^3) as function of time - normalised
Sensor_O3_RefSt_norm = sensor[["Sensor_O3", "RefSt"]]
Sensor_O3_RefSt_norm["RefSt"] = 4*Sensor_O3_RefSt_norm["RefSt"]
Sensor_O3_RefSt_norm.plot()

# %%
# Raw scatter plot
scplt = pd.DataFrame({'RefSt': sensor["RefSt"], 'Sensor_O3': sensor["Sensor_O3"]})
scplt.plot.scatter(x = 'Sensor_O3', y = 'RefSt')


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

scplt = pd.DataFrame({'normRefSt': normRefSt, 'normSensor_O3': normSensor_O3})
scplt.plot.scatter(x = 'normSensor_O3', y = 'normRefSt')

# %%
# Sensor_O3 with respect to Temp
scplt = pd.DataFrame({'Temp': sensor["Temp"], 'Sensor_O3': sensor["Sensor_O3"]})
scplt.plot.scatter(x = 'Sensor_O3', y = 'Temp')

# %%
# Sensor_O3 with respect to RelHum
scplt = pd.DataFrame({'RelHum': sensor["RelHum"], 'Sensor_O3': sensor["Sensor_O3"]})
scplt.plot.scatter(x = 'Sensor_O3', y = 'RelHum')

# %%
# RefSt with respect to Temp
scplt = pd.DataFrame({'Temp': sensor["Temp"], 'RefSt': sensor["RefSt"]})
scplt.plot.scatter(x = 'RefSt', y = 'Temp')

# %%
# RefSt with respect to RelHum
scplt = pd.DataFrame({'RelHum': sensor["RelHum"], 'RefSt': sensor["RefSt"]})
scplt.plot.scatter(x = 'RefSt', y = 'RelHum')

# %%
