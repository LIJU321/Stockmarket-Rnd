#!/usr/bin/env python
# coding: utf-8



# all imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# from sklearn import neural_network

import pandas as pd
import matplotlib.pyplot as plt 
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, LSTM
from sklearn import svm
from sklearn.preprocessing import StandardScaler




df1 = pd.read_csv("RND_APP\ADANIPORTS.csv")
#dfx = pd.read_csv(r"ADANIPORTS5.csv")  #  CSV file from - 2021-04-30  (APRIL 30 2021)till todays date.
dfx = pd.read_csv(r"RND_APP\ADANI-X.csv")
df2 = df1[["Date","Open","Close","Low","High"]] 

dfx2 = dfx.dropna()

dfx3 = dfx2.drop(0)  #! IMPORTANT TO DROP FIRST ROW 

dfx4 = dfx3[["Date","Open","Close","Low","High"]]


df = pd.concat([df2, dfx4], ignore_index=True)  # CONCATINATE DATA TO THE END OF CURRENT DATAFRAM ,, CONCATINATE DFX4 to THE df2 DATAFRAME 
# MEANING 2ND DATAFRAME IS CONCATINATED TO THE 1ST ONE


Low_= df["Low"].mean()

open = df["Open"].mean()

x =df[["Open"]]


y = df[["Low"]]



# # (OPEN ANDE LOW PRICE ANALYSIS)


# 
# (OPEN AND PREVIOUS CLOSED PRICE ANALYSIS ....)
# plot shows we have a positive (correlation) relation between open and prev close



coefx = df["Open"]
coefy =df["Low"]



s = np.corrcoef(coefx,coefy)


correlation_coefficeint = np.corrcoef(coefx,coefy)[1,0]



# 0.9979343764297565  almost Close to 1 and  & above  0 so its considered as a positive correlation  -1 considered to be a neagtive correlation, 0 considered no correlation. 


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)





YY =Y_train.head(997)




X_test2 = X_test.to_numpy()



X_train2 = X_train.to_numpy()





Y_train2 = Y_train.to_numpy()
Y_train2 = Y_train2.ravel()





#########################################################################################################################################################


# # UPDATED KNN MODEL ON LOW PRICE EVALUATION AND FIT



# Split the data into training and testing sets  FOR  HIGH PRICE





### UPDATED KNN CLOSE #####
""" Relaible i think"""
# Split the data into features (A, B, C) and target variable (X)
#features = df[['Open', 'Low', 'High','Volume']].fillna(0) #volume will decrease accuracy of model
features = df[['Open']]
features2 = features.to_numpy()
target = df['Low']

# Split the data into training and testing sets
X_trainF, X_testF, Y_trainF, Y_testF = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and fit the KNN model
knn = KNeighborsRegressor()
knn.fit(X_trainF, Y_trainF)

# Predict on the test set
y_pred = knn.predict(X_testF)

# Evaluate the model (e.g., using mean squared error)
mse = mean_squared_error(Y_testF, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_testF, y_pred)
mae = mean_absolute_error(Y_testF, y_pred)

print("Mean Squared Error (KNN2):", mse)
print("Root Mean Squared Error (KNN2):", rmse)
print("Mean Absolute Error (KNN2):", mae)
print("R-squared (R2) Score (KNN2):", r2)
# y_pred


# # LINEAR REGRESSION

# # LinearRegression on LOW


# # LinearRegression on LOW MODEL


RegressorLow = LinearRegression()

features = df[['Open']]
features2 = features.to_numpy()
target = df['Low']

X_train4, X_test4, Y_train4, Y_test4 = train_test_split(features, target, test_size=0.2, random_state=42)

RegressorLow.fit(X_train4,Y_train4)
# Split the data into training and testing sets

y_predictedLow =RegressorLow.predict(X_test4)
mseHigh = metrics.mean_squared_error(Y_test4,y_predictedLow)
rmseHigh = np.sqrt(metrics.mean_squared_error(Y_test4,y_predictedLow))
maeHigh =  metrics.mean_absolute_error(Y_test4,y_predictedLow)
r2High = r2_score(Y_test4,y_predictedLow)

print("MSE:", mseHigh)
print("RMSE:",rmseHigh)
print("R-squared:",r2High)
print("MAE:", maeHigh)
X_test4_ = np.asarray(X_test4)


################################################################ SCALER VALUE LOW PRICE  #############################################################



##################################################################      ###########################################################




# # Validating models quality using metrics Linear regression model evaluation

# # MSE Lower is better
# # RMSE Lower is better
# # R^2 Higher is better
# # MAE	Lower is better

# # DEcicsionTree Regressor model fitting and evaluation 




# # DEcicsionTree Regressor model on LOW price


"""the model performs quite well."""
DecisionTReeLow = DecisionTreeRegressor()
DecisionTReeLow.fit(X_train4,Y_train4)
y_TreePredictHigh = DecisionTReeLow.predict(X_test4)
#y_TreePredictHigh

""" DecisionTree  model evaluation """
mseLowTree = metrics.mean_squared_error(Y_test4,y_predictedLow)
rmseLowTree = np.sqrt(metrics.mean_squared_error(Y_test4,y_predictedLow))
maeLowTree =  metrics.mean_absolute_error(Y_test4,y_predictedLow)
r2LowTree = r2_score(Y_test4,y_predictedLow)

print("MSE:", mseLowTree)
print("RMSE:",rmseLowTree)
print("R-squared:",r2LowTree)
print("MAE:", maeLowTree)


# #  Fitting SVM model to X_train and X_test



####### TEST ######
X_train2 = X_train.to_numpy()
X_train2
Y =  np.asarray(Y_train)



scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)



scaler.fit(Y)
input1 = [[722.95]]
Y_train_scaled = scaler.transform(Y)



scaler = StandardScaler()
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)



Y_train_scaled2 = Y_train_scaled.reshape(-1)



# # SVM ON LOW PRICE


"""the model is performing very well. The low MSE and RMSE values,
along with the high R-squared value, indicate that the model's predictions are close to the actual values.
"""
svmclassifierLow = svm.SVR(kernel="linear")
##### TEST #####
svmclassifierLow.fit(X_train4, Y_train4)

# Use the trained SVM model to make predictions on the scaled test data
Y_predLowsvm = svmclassifierLow.predict(X_test4)
# Y_predHighsvm

# SVM MODEL evaluation ....using metrics

mseLowsvm = metrics.mean_squared_error(Y_test4,Y_predLowsvm)
rmseLowsvm = np.sqrt(metrics.mean_squared_error(Y_test4,Y_predLowsvm))
maeLowsvm =  metrics.mean_absolute_error(Y_test4,Y_predLowsvm)
r2Lowsvm = r2_score(Y_test4,Y_predLowsvm)


print("MSE:", mseLowsvm)
print("RMSE:",rmseLowsvm)
print("R-squared:",r2Lowsvm)
print("MAE:",maeLowsvm)



################################################################################################################################


#######################################################################################################################################


# # KNN MODEL AND knn evaluation using metricS

# #  KNN ON CLOSE PRICE


### KNN ON CLOSE PRICE 
""" have higher prediction errors """
# Create the KNN regressor
knn = KNeighborsRegressor()
# Fit the regressor to the data
knn.fit(X_trainF, Y_trainF)

# Predict the value of y for a new input x
KNNPRED = knn.predict(X_testF)
#print(KNNPRED)

mse = metrics.mean_squared_error(Y_testF,KNNPRED)
# Calculate the RMSE
rmse = np.sqrt(mse)

# Calculate the R-squared
r2 = r2_score(Y_testF,KNNPRED)
mae =  metrics.mean_absolute_error(Y_testF,KNNPRED)

print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
print("MAE:", mae)



X_train4_  = X_train4.to_numpy()

"""model appears to perform well, with a relatively low MSE and RMSE,
a high R-squared value indicating good explanatory power, and a reasonable MAE
"""
Lowknn = KNeighborsRegressor()
# Fit the regressor to the data
Lowknn.fit(X_train4_,Y_train4)

# Predict the value of y for a new input x
KNNPREDLOW = Lowknn.predict(X_test4)
#print(KNNPREDHigh)


mseknnLow = metrics.mean_squared_error(Y_test4,KNNPREDLOW)

# Calculate the RMSE
rmseknnLow = np.sqrt(mse)

# Calculate the R-squared
r2KnnLow = r2_score(Y_test4,KNNPREDLOW)
maeknnLow =  metrics.mean_absolute_error(Y_test4,KNNPREDLOW)

print("MSE:", mseknnLow)
print("RMSE:",rmseknnLow)
print("R-squared:",r2KnnLow)
print("MAE:", maeknnLow)


# # MSE	Lower is better
# # RMSE	Lower is better
# # R^2	Higher is better
# # MAE	Lower is better



# # Analysis of better model found here is SVM and Linear Regressor decisionTree is flop here. 
# # Linear regressor mse = 177.80298234987615
# # linear regressor rmse = 13.313181177910515
# # linear regressor mae = 9.260471493846618
# # linear regressor r2 = 0.9954780246991811 
# 

# # we tested and predicted with only the data from 2007-11-27 to 2021-04-30

# # To get the corresponding  stock price details Type = Adani Ports and Special Economic Zone Ld Stock price today

# # neural_network and neural network model evaluation  on Open


# Create the neural network model


# # Neural network on Low PRICE


"""THIS IS MODEL IS  PERFORMING WELL  MODEL BASED ON THE METRICS"""

# Create the neural network model
Lowmodel_neural_network = Sequential()
Lowmodel_neural_network.add(Dense(64, activation='relu', input_shape=(X_train4.shape[1],)))
Lowmodel_neural_network.add(Dense(64, activation='relu'))
Lowmodel_neural_network.add(Dense(1, activation='linear'))

# Compile the model
Lowmodel_neural_network.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
Lowmodel_neural_network.fit(X_train4,Y_train4, epochs=10, batch_size=32)

# Predict the values for the test data
predicted_valuesLow = Lowmodel_neural_network.predict(X_test4)

# Print the predicted values
#print(predicted_values)

# Predict the values for the test data Low PRICE>>>>>

""" neural network model evaluation """

# Calculate MSE
msecnn = mean_squared_error(Y_test4, predicted_valuesLow)
print("Mean Squared Error (MSE):", msecnn)

# Calculate RMSE
rmsecnn = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmsecnn)

# Calculate MAE
maecnn = mean_absolute_error(Y_test4, predicted_valuesLow)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared
r3cnn = r2_score(Y_test4, predicted_valuesLow)
print("R-squared:", r3cnn)



###################################################### all models input is open price ################################################################

# #  MODEL 6 - DNN Low price



# Create the neural network model>>>>>>>>
"""THIS IS MODEL IS  PERFORMING WELL  MODEL BASED ON THE METRICS"""
Dnnmodel = Sequential()
Dnnmodel.add(Dense(64, activation='relu', input_shape=(X_train4.shape[1],)))
Dnnmodel.add(Dense(64, activation='relu'))
Dnnmodel.add(Dense(1, activation='linear'))

# Compile the model
Dnnmodel.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
Dnnmodel.fit(X_train4, Y_train4, epochs=10, batch_size=32)

# Predict the values for the test data
predicted_values = Dnnmodel.predict(X_test4)

# Evaluate the model on the test data using metrics


mse = mean_squared_error(Y_test4, predicted_values)
mae = mean_absolute_error(Y_test4, predicted_values)
r2 = r2_score(Y_test4, predicted_values)
rmse =  np.sqrt(mse)

# Print the predicted values and evaluation results
#print("Predicted values:")
#print(predicted_values)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared (R2) Score:", r2)

Dnnpred = Dnnmodel.predict(X_test4)

LOW = Dnnmodel.predict([[737.65]]) # INput is OPEN, LOW, CLOSE for output LOW PRICE
print("PREDICTED_VALUE:", LOW)


# #  MODEL 7 - RNN  Low price




"""These metrics indicate that your model performs well, as the errors are relatively low, 
and the model can explain a significant portion of the variance in the data."""

# Create the neural network model
Rnnmodel = Sequential()
Rnnmodel.add(LSTM(64, activation='relu', input_shape=(X_train4_.shape[1], 1)))
Rnnmodel.add(Dense(64, activation='relu'))
Rnnmodel.add(Dense(1, activation='linear'))

# Compile the model
Rnnmodel.compile(loss='mean_squared_error', optimizer='adam')

# Reshape the input data for LSTM
X_train_reshaped = X_train4_.reshape(X_train4_.shape[0], X_train4_.shape[1], 1)
X_test_reshaped = X_test4_.reshape(X_test4_.shape[0], X_test4_.shape[1], 1)

# Fit the model to the training data
Rnnmodel.fit(X_train_reshaped,Y_train4, epochs=10, batch_size=32)


# Predict the value for a single data point
# input_data = np.array([735.00, 700, 740]).reshape(1, 3, 1)  #INPUT OPEN PRICE
# input_data_reshaped = np.array(X_test_reshaped).reshape(1, 1, 1)
#predicted_value = Rnnmodel.predict(input_data_reshaped)


predicted_value = Rnnmodel.predict(X_test_reshaped)

# Evaluate the model on the test data using metrics
predicted_values = Rnnmodel.predict(X_test_reshaped)
mse = mean_squared_error(Y_test4, predicted_values)
rmse =  np.sqrt(mse)
mae = mean_absolute_error(Y_test4, predicted_values)
r2 = r2_score(Y_test4, predicted_values)

# Print the predicted value and evaluation results
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared (R2) Score:", r2)
print()
print()



RnnLow = Rnnmodel.predict(np.array([730]).reshape(1, 1, 1)) #<-- <-- # INput is OPEN, LOW, CLOSE for output HIGH PRICE
print("PREDICTED_VALUE:",RnnLow)





RnnLow = Rnnmodel.predict(np.array([730]).reshape(1, 1, 1)) #<-- <-- # INput is OPEN, LOW, CLOSE for output HIGH PRICE
print("PREDICTED_VALUE:",RnnLow)


# # MODEL 8 - CNN MODEL Low


# Create the neural network model
cnn_model = Sequential()
cnn_model.add(Reshape((X_train4.shape[1],), input_shape=(X_train4.shape[1], 1)))
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(1, activation='linear'))

# Compile the model
cnn_model.compile(loss='mean_squared_error', optimizer='adam')

# Reshape the input data for CNN
X_train_reshaped = X_train4_.reshape(X_train4_.shape[0], X_train4_.shape[1])
X_test_reshaped = X_test4_.reshape(X_test4_.shape[0], X_test4_.shape[1])

# Fit the model to the training data
cnn_model.fit(X_train_reshaped, Y_train4, epochs=10, batch_size=32)

# Predict the value for a single data point

#input_data = [[735.55]]  # INPUT OPEN PRICE

predicted_value = cnn_model.predict(X_test_reshaped)

# Evaluate the model on the test data using metrics
predicted_values = cnn_model.predict(X_test_reshaped)
mse = mean_squared_error(Y_test4, predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test4, predicted_values)
r2 = r2_score(Y_test4, predicted_values)

# Print the predicted value and evaluation results
print()
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared (R2) Score:", r2)





cnn_Low = cnn_model.predict([[737.65]]) #<-- <-- # INput is OPEN, LOW, CLOSE for output HIGH PRICE
print("PREDICTED_VALUE:",cnn_Low)




cnn_Low = cnn_model.predict([[724.50]]) #<-- <-- # INput is OPEN, LOW, CLOSE for output HIGH PRICE
print("PREDICTED_VALUE:",cnn_Low)


# # Feedforward Neural Network on Low price (Multi-Layer Perceptron):



""" These results indicate that the FFNN model has achieved relatively low mean squared error, root mean squared error,
and mean absolute error values, suggesting that it is performing well in predicting the target variable. The high R-squared (R2) 
score of 0.9971320639799347 further confirms the model's
strong performance in explaining the variance in the target variable."""

# Create the FFNN model
ffnn_model_low = Sequential()
ffnn_model_low.add(Dense(64, activation='relu', input_shape=(X_train4.shape[1],)))
ffnn_model_low.add(Dense(64, activation='relu'))
ffnn_model_low.add(Dense(1, activation='linear'))

# Compile the model
ffnn_model_low.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
ffnn_model_low.fit(X_train4, Y_train4, epochs=10, batch_size=32)

# Predict using the fitted FFNN model
Y_pred_ffnn = ffnn_model_low.predict(X_test4)

# Evaluate the FFNN model on the test data using metrics
mse_ffnn = mean_squared_error(Y_test4, Y_pred_ffnn)
rmse_ffnn = np.sqrt(mse_ffnn)
mae_ffnn = mean_absolute_error(Y_test4, Y_pred_ffnn)
r2_ffnn = r2_score(Y_test4, Y_pred_ffnn)

# Print the predicted values and evaluation results
print("Predicted Values (FFNN):")
#print(Y_pred_ffnn)
print()
print("Mean Squared Error (FFNN):", mse_ffnn)
print("Root Mean Squared Error (FFNN):", rmse_ffnn)
print("Mean Absolute Error (FFNN):", mae_ffnn)

print("R-squared (R2) Score (FFNN):", r2_ffnn)




PREDICTEDLow = ffnn_model_low.predict([[737.65]])  #<-- <-- # INput is OPEN, LOW, CLOSE for output HIGH PRICE
print("PREDICTED_VALUE:",PREDICTEDLow)

print()
print("CODE EXECUTED")






