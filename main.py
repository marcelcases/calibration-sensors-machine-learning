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


#%%
# Loss functions definition
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def loss_functions(y_true, y_pred):
    print("Loss functions:")
    print("* R-squared =", r2_score(y_true, y_pred))
    print("* RMSE =", mean_squared_error(y_true, y_pred))
    print("* MAE =", mean_absolute_error(y_true, y_pred))


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
# sgdr = SGDRegressor(loss='squared_loss', alpha=.001, tol=1e-5)
sgdr = SGDRegressor(loss='squared_loss', max_iter=5)

# Normalize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit
sgdr.fit(X_train, Y_train)

# Get MLR coefficients
print('Intercept: \n', sgdr.intercept_)
print('Coefficients: \n', sgdr.coef_)
print('Iters: \n', sgdr.n_iter_)
print(sgdr.get_params())


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
# Kernel Regression
# from sklearn_extensions.kernel_regression import KernelRegression
from sklearn.kernel_ridge import KernelRidge

# Model
kr_rbf = KernelRidge(kernel="rbf")
kr_poly = KernelRidge(kernel="poly", degree=4)

# Fit
kr_rbf.fit(X_train, Y_train)
kr_poly.fit(X_train, Y_train)

# Predict
df_test["KR_RBF_Pred"] = kr_rbf.predict(X_test)
df_test["KR_Poly_Pred"] = kr_poly.predict(X_test)

# Plot linear
df_test[["RefSt", "KR_RBF_Pred", "KR_Poly_Pred"]].plot()
plt.xticks(rotation=20)

# Plot regression
sns.lmplot(x = 'RefSt', y = 'KR_RBF_Pred', data = df_test, fit_reg=True, line_kws={'color': 'orange'}) 
sns.lmplot(x = 'RefSt', y = 'KR_Poly_Pred', data = df_test, fit_reg=True, line_kws={'color': 'orange'}) 

# MLR loss
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["KR_RBF_Pred"])
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["KR_Poly_Pred"])


# %%
# Polynomial Kernel Regression stats vs. hyperparameters
def kr_stats():
    kr_aux = pd.DataFrame({'RefSt': Y_test})

    degree = [*range(1, 26, 1)]
    r_squared = []
    rmse = []
    mae = []
    time_ms = []

    for i in degree:
        kr = KernelRidge(kernel="poly", degree=i)

        # fit
        start_time = float(datetime.datetime.now().strftime('%S.%f'))
        kr.fit(X_train, Y_train)
        end_time = float(datetime.datetime.now().strftime('%S.%f'))
        execution_time = (end_time - start_time) * 1000

        # predict
        kr_aux["KR_Pred"] = kr.predict(X_test)

        # RF loss
        r_squared.append(r2_score(kr_aux["RefSt"], kr_aux["KR_Pred"]))
        rmse.append(mean_squared_error(kr_aux["RefSt"], kr_aux["KR_Pred"]))
        mae.append(mean_absolute_error(kr_aux["RefSt"], kr_aux["KR_Pred"]))
        time_ms.append(execution_time)

    kr_stats = pd.DataFrame({'degree': degree, 'r_squared': r_squared, 'rmse': rmse, 'mae': mae, 'time_ms': time_ms})
    kr_stats = kr_stats.set_index('degree') # index column (X axis for the plots)
    print(kr_stats)

    # plot
    kr_stats[["r_squared"]].plot()
    kr_stats[["rmse"]].plot()
    kr_stats[["mae"]].plot()
    kr_stats[["time_ms"]].plot()

kr_stats()


# %%
# Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct, WhiteKernel

# rbf = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-10, 1e10)) * RBF(length_scale=1.0, length_scale_bounds=(1e-10, 1e10))
rbf = ConstantKernel() * RBF()
dpwh = DotProduct() + WhiteKernel()
gp_rbf = GaussianProcessRegressor(kernel=rbf, alpha=150, random_state=0)
gp_dpwh = GaussianProcessRegressor(kernel=dpwh, alpha=150, random_state=0)

# fit
gp_rbf.fit(X_train, Y_train)
gp_dpwh.fit(X_train, Y_train)

# predict
df_test["GP_RBF_Pred"] = gp_rbf.predict(X_test)
df_test["GP_DPWK_Pred"] = gp_dpwh.predict(X_test)

# Obtain optimized kernel parameters
# l = gp.kernel_.k2.get_params()['length_scale']
# sigma_f = np.sqrt(gp.kernel_.k1.get_params()['constant_value'])

# print("Kernel params k1", gp.kernel_.k1.get_params())
# print("Kernel params k2", gp.kernel_.k2.get_params())

# plot linear
df_test[["RefSt", "GP_RBF_Pred", "GP_DPWK_Pred"]].plot()
plt.xticks(rotation=20)

# plot regression
sns.lmplot(x = 'RefSt', y = 'GP_RBF_Pred', data= df_test, fit_reg=True, line_kws={'color': 'orange'}) 
sns.lmplot(x = 'RefSt', y = 'GP_DPWK_Pred', data= df_test, fit_reg=True, line_kws={'color': 'orange'}) 

# GP loss
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["GP_RBF_Pred"])
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["GP_DPWK_Pred"])


# %%
# Gaussian Process stats vs. hyperparameters
def gp_stats():
    gp_aux = pd.DataFrame({'RefSt': Y_test})

    alpha = [*range(5, 55, 5)]
    # alpha = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10,50,100,150,200]
    r_squared = []
    rmse = []
    mae = []
    time_ms = []

    for i in alpha:
        rbf = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-10, 1e10)) * RBF(length_scale=1.0, length_scale_bounds=(1e-10, 1e10))
        gp = GaussianProcessRegressor(kernel=rbf, alpha=i, random_state=0)

        # fit
        start_time = float(datetime.datetime.now().strftime('%S.%f'))
        gp.fit(X_train, Y_train)
        end_time = float(datetime.datetime.now().strftime('%S.%f'))
        execution_time = (end_time - start_time) * 1000

        # predict
        gp_aux["GP_Pred"] = gp.predict(X_test)

        # RF loss
        r_squared.append(r2_score(gp_aux["RefSt"], gp_aux["GP_Pred"]))
        rmse.append(mean_squared_error(gp_aux["RefSt"], gp_aux["GP_Pred"]))
        mae.append(mean_absolute_error(gp_aux["RefSt"], gp_aux["GP_Pred"]))
        time_ms.append(execution_time)

    gp_stats = pd.DataFrame({'alpha': alpha, 'r_squared': r_squared, 'rmse': rmse, 'mae': mae, 'time_ms': time_ms})
    gp_stats = gp_stats.set_index('alpha') # index column (X axis for the plots)
    print(gp_stats)

    # plot
    gp_stats[["r_squared"]].plot()
    gp_stats[["rmse"]].plot()
    gp_stats[["mae"]].plot()
    gp_stats[["time_ms"]].plot()

gp_stats()


# %%
# Support Vector Regression
from sklearn.svm import SVR

# Models
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=3)

# Fit
svr_rbf.fit(X_train, Y_train)
svr_lin.fit(X_train, Y_train)
svr_poly.fit(X_train, Y_train)

# Predict
df_test["SVR_RBF_Pred"] = svr_rbf.predict(X_test)
df_test["SVR_Line_Pred"] = svr_lin.predict(X_test)
df_test["SVR_Poly_Pred"] = svr_poly.predict(X_test)

# Plot linear
df_test[["RefSt", "SVR_RBF_Pred", "SVR_Line_Pred", "SVR_Poly_Pred"]].plot()
plt.xticks(rotation=20)

# Plot regression
sns.lmplot(x = 'RefSt', y = 'SVR_RBF_Pred', data = df_test, fit_reg=True, line_kws={'color': 'orange'}) 
sns.lmplot(x = 'RefSt', y = 'SVR_Line_Pred', data = df_test, fit_reg=True, line_kws={'color': 'orange'}) 
sns.lmplot(x = 'RefSt', y = 'SVR_Poly_Pred', data = df_test, fit_reg=True, line_kws={'color': 'orange'}) 

# NN loss
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["SVR_RBF_Pred"])
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["SVR_Line_Pred"])
loss_functions(y_true=df_test["RefSt"], y_pred=df_test["SVR_Poly_Pred"])



# %%
# Neural Network
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, InputLayer
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler

print(tf.__version__)

# Normalise data
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
history = nn.fit(X_train, Y_train, batch_size = 10, epochs = 2000)

# Plot loss
plt.plot(history.history['loss'][5:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
plt.show()

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
