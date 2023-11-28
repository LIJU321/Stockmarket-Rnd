
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# from sklearn import neural_network

import pandas as pd
import matplotlib.pyplot as plt 
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, LSTM


# In[231]:


df1 = pd.read_csv("RND_APP\ADANIPORTS.csv")
#dfx = pd.read_csv(r"ADANIPORTS5.csv")  #  CSV file from - 2021-04-30  (APRIL 30 2021)till todays date.
dfx = pd.read_csv("RND_APP\ADANIPORTSNS8.csv")
df2 = df1[["Date","Open","Close","Low","High"]] 





df1 # 3322 rows × 15 columns





dfx   





dfx2 = dfx.dropna()
dfx2




df2



dfx3 = dfx2.drop(0)  #! IMPORTANT TO DROP FIRST ROW 
dfx3





dfx4 = dfx3[["Date","Open","Close","Low","High"]]
dfx4





df = pd.concat([df2, dfx4], ignore_index=True)  # CONCATINATE DATA TO THE END OF CURRENT DATAFRAM ,, CONCATINATE DFX4 to THE df2 DATAFRAME





df







# **Analysis of of whole dataframe**


















# **Analysis of close price,previous close and open ** **bold text**



close_= df["Close"].mean()






open = df["Open"].mean()




x = df[["Open"]]





y = df[["Close"]]








# # (OPEN AND CLOSE PRICE ANALYSIS)






sns.scatterplot(df,x="Open",y="Close")




sns.scatterplot(df,x="Open",y="High")




sns.scatterplot(df,x="Open",y="Low")



# 
# (OPEN AND PREVIOUS CLOSED PRICE ANALYSIS ....)
# plot shows we have a positive (correlation) relation between open and prev close




Y = df["Open"]


# In[269]:


X2 = df[["Open"]]
Y2 = df[["High"]]


# "Close" is our depended variable(Target) 



sns.scatterplot(df)


# In[279]:


histx = plt.hist(x,bins=100) , plt.xlabel("X _features"), plt.ylabel("Y or frequency "), plt.title("predictors / Features histogram plot")


# In[280]:


hsity = plt.hist(y,bins=100),plt.xlabel("X _features"), plt.ylabel("Y or frequency "), plt.title("response/ target histogram plot")


# In[281]:


def plotter(dataframe,feauters,target):
   return sns.scatterplot(dataframe,x=feauters,y=target)
plotter(df,"Open","Close")


# # (OPEN AND HIGH PRICE ANALYSIS)



#### sns.regplot(x,y) 
sns.regplot(df,x="Open",y="High")


# In[283]:


sns.scatterplot(df,x="Open",y="High")


# # To manually calculate the correlation coefficient between two variables, you can use the following formula:/
# # r = (n * sum(x*y) - sum(x) * sum(y)) / sqrt((n * sum(x**2) - sum(x)**2) * (n * sum(y**2) - sum(y)**2))
# 
# In a square matrix, the diagonal elements are the elements that are on the diagonal from the upper-left corner to the lower-right corner. The off-diagonal elements are all the other elements in the matrix that are not on the diagonal.
# 
# In the context of numpy.corrcoef(), the off-diagonal element of the returned matrix is the correlation coefficient between the two variables that you passed as input. The diagonal elements are always equal to 1, because the correlation coefficient between a variable and itself is always 1.
# 
# To manually calculate the correlation coefficient between two variables, you can use the following formula:
# 
# r = (n * sum(x*y) - sum(x) * sum(y)) / sqrt((n * sum(x**2) - sum(x)**2) * (n * sum(y**2) - sum(y)**2))
# where:
# 
# r is the correlation coefficient
# n is the number of data points
# x and y are the two arrays of data
# 

# **check the correlation coefficient using numpy function correlation coefficent are given as a matrix and the coffecient are in off daigonal **

# In[284]:


coefx = df["Open"]
coefy =df["Close"]




s = np.corrcoef(coefx,coefy)




correlation_coefficeint = np.corrcoef(coefx,coefy)[1,0]



# 0.9979343764297565  almost Close to 1 and  & above  0 so its considered as a positive correlation  -1 considered to be a neagtive correlation, 0 considered no correlation. 



# CLoSE
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[289]:


YY = Y_train.head(997)



X_test2 = X_test.to_numpy()


X_train2 = X_train.to_numpy()



Y_train2 = Y_train.to_numpy()
Y_train2 = Y_train2.ravel()


#########################################################################################################################################################



### UPDATED KNN CLOSE #####
""" Relaible i think"""
# Split the data into features (A, B, C) and target variable (X)
#features = df[['Open', 'Low', 'High','Volume']].fillna(0) #volume will decrease accuracy of model
features = df[['Open', 'Low', 'High']]
features2 = features.to_numpy()
target = df['Close']

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





# # LINEAR REGRESSION

# # LinearRegression on CLOSE


Regressor = LinearRegression()
Regressor.fit(X_train,Y_train)
y_predicted = Regressor.predict(X_test)
#####
mse = metrics.mean_squared_error(Y_test,y_predicted)
rmse = np.sqrt(metrics.mean_squared_error(Y_test,y_predicted))
mae =  metrics.mean_absolute_error(Y_test,y_predicted)
r2 = r2_score(Y_test,y_predicted)

print("MSE:", mse)
print("RMSE:",rmse)
print("R-squared:",r2)
print("MAE:", mae)


# # LinearRegression on HIGH MODEL




RegressorHigh = LinearRegression()

# features = df[['Open', 'Low', 'Close']]
features = df[['Open']]
features2 = features.to_numpy()
target = df['High']

X_train4, X_test4, Y_train4, Y_test4 = train_test_split(features, target, test_size=0.2, random_state=42)

RegressorHigh.fit(X_train4,Y_train4)
# Split the data into training and testing sets

y_predictedHigh = RegressorHigh.predict(X_test4)
mseHigh = metrics.mean_squared_error(Y_test4,y_predictedHigh)
rmseHigh = np.sqrt(metrics.mean_squared_error(Y_test4,y_predictedHigh))
maeHigh =  metrics.mean_absolute_error(Y_test4,y_predictedHigh)
r2High = r2_score(Y_test4,y_predictedHigh)

print("MSE:", mseHigh)
print("RMSE:",rmseHigh)
print("R-squared:",r2High)
print("MAE:", maeHigh)



X_test4_ = np.asarray(X_test4)




################################################################ SCALER VALUE CLOSE PRICE  #############################################################



# mypredtest1 = Regressor.predict(np.array([784]).reshape(1, -1))  # <-- # INput is OPEN for output CLOSE PRICE




# mypredtes2 = Regressor.predict([[780]])  # <-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE



##################################################################      ###########################################################




plotter(df,"Open","Close")




# **Visualizations**


hist3 = plt.hist(y_predicted,bins=100),plt.xlabel("the value of predicted varible and its occurance")


# # Validating models quality using metrics Linear regression model evaluation

# # MSE Lower is better
# # RMSE Lower is better
# # R^2 Higher is better
# # MAE	Lower is better

# # DEcicsionTree Regressor model on close fitting and evaluation 

# In[310]:


DecisionTRee = DecisionTreeRegressor()
DecisionTRee.fit(X_train, Y_train)
y_TreePredict = DecisionTRee.predict(X_test)
#y_TreePredict
""" DecisionTree  model evaluation"""
msetree = metrics.mean_squared_error(Y_test,y_TreePredict)
rmsetree = np.sqrt(metrics.mean_squared_error(Y_test,y_TreePredict))
maetree =  metrics.mean_absolute_error(Y_test,y_TreePredict)
r2tree = r2_score(Y_test,y_TreePredict)

print("MSE:", msetree)
print("RMSE:",rmsetree)
print("R-squared:",r2tree)
print("MAE:", maetree)


# # DEcicsionTree Regressor model on HIGH price



"""the model performs quite well."""
DecisionTReeHigh = DecisionTreeRegressor()
DecisionTReeHigh.fit(X_train4,Y_train4)
y_TreePredictHigh = DecisionTReeHigh.predict(X_test4)
#y_TreePredictHigh

""" DecisionTree  model evaluation """
mseHighTree = metrics.mean_squared_error(Y_test4,y_predictedHigh)
rmseHighTree = np.sqrt(metrics.mean_squared_error(Y_test4,y_predictedHigh))
maeHighTree =  metrics.mean_absolute_error(Y_test4,y_predictedHigh)
r2HighTree = r2_score(Y_test4,y_predictedHigh)

print("MSE:", mseHighTree )
print("RMSE:",rmseHighTree)
print("R-squared:",r2HighTree)
print("MAE:", maeHighTree)




myTreePredict1 = DecisionTRee.predict(np.array([780]).reshape(1,-1)) # <-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE




myTreePredict2 = DecisionTRee.predict([[780]]) #<-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE


# #  Fitting SVM model to X_train and X_test



from sklearn import svm
from sklearn.preprocessing import StandardScaler
####### TEST ######
X_train2 = X_train.to_numpy()

Y =  np.asarray(Y_train)
#Y



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



# flatten()# to change 2D array to 1D array


X_train_scaled = X_train_scaled.reshape(-1, 1)

# # SVM ON CLOSE PRICE


""" he model performs reasonably well"""
# Train the SVM model on the scaled training data
svmclassifier = svm.SVR(kernel="linear")
svmclassifier.fit(X_train,Y_train)

# Use the trained SVM model to make predictions on the scaled test data
Y_pred = svmclassifier.predict(X_test)

#Train the SVM model on the scaled training data
svmclassifier.fit(X_train_scaled,Y_train_scaled2)

# Use the trained SVM model to make predictions on the scaled test data
Y_pred = svmclassifier.predict(X_test2)
#  Y_pred
    
# SVM MODEL evaluation ....using metrics

msesvm = metrics.mean_squared_error(Y_test,Y_pred)
rmsesvm = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
maesvm =  metrics.mean_absolute_error(Y_test,Y_pred)
r2svm = r2_score(Y_test,Y_pred)

print("SVM_MSE:", msesvm)
print("SVM_RMSE:",rmsesvm)
print("SVM_R-squared:",r2svm)
print("SVM_MAE:", maesvm)




# X_test4 = X_test4.to_numpy()
# X_test4


# # SVM ON HIGH PRICE



"""the model is performing very well. The low MSE and RMSE values,
along with the high R-squared value, indicate that the model's predictions are close to the actual values.
"""
# svmclassifierHigh = svm.SVR(kernel="linear")
# ##### TEST #####
# svmclassifierHigh.fit(X_train4, Y_train4)

# # Use the trained SVM model to make predictions on the scaled test data
# Y_predHighsvm = svmclassifierHigh.predict(X_test4)
# # Y_predHighsvm

# # SVM MODEL evaluation ....using metrics

# mseHighsvm = metrics.mean_squared_error(Y_test4,Y_predHighsvm)
# rmseHighsvm = np.sqrt(metrics.mean_squared_error(Y_test4,Y_predHighsvm))
# maeHighsvm =  metrics.mean_absolute_error(Y_test4,Y_predHighsvm)
# r2Highsvm = r2_score(Y_test4,Y_predHighsvm)


# print("MSE:", mseHighsvm)
# print("RMSE:",rmseHighsvm)
# print("R-squared:",r2Highsvm)
# print("MAE:", maeHighsvm)




################################################################################################################################




# mySVMpred1 = svmclassifier.predict(np.array([780]).reshape(1,-1))# <-- # INput is OPEN  for output CLOSE PRICE



# mySVMpred2 = svmclassifier.predict([[780]]) #<-- # INput is OPEN for output CLOSE PRICE


#######################################################################################################################################


# # KNN MODEL AND knn evaluation using metricS

# #  KNN ON CLOSE PRICE




### KNN ON CLOSE PRICE 
""" have higher prediction errors """
# Create the KNN regressor
knn = KNeighborsRegressor()
# Fit the regressor to the data
knn.fit(X_train, Y_train)

# Predict the value of y for a new input x
KNNPRED = knn.predict(X_test)
#print(KNNPRED)

mse = metrics.mean_squared_error(Y_test,KNNPRED)
# Calculate the RMSE
rmse = np.sqrt(mse)

# Calculate the R-squared
r2 = r2_score(Y_test,KNNPRED)
mae =  metrics.mean_absolute_error(Y_test,KNNPRED)

print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
print("MAE:", mae)


# #  KNN ON HIGH PRICE



X_train4_  = X_train4.to_numpy()



"""model appears to perform well, with a relatively low MSE and RMSE,
a high R-squared value indicating good explanatory power, and a reasonable MAE
"""
# Highknn = KNeighborsRegressor()
# # Fit the regressor to the data
# Highknn.fit(X_train4_,Y_train4)

# # Predict the value of y for a new input x
# KNNPREDHigh = Highknn.predict(X_test4)
# #print(KNNPREDHigh)


# mseknnHigh = metrics.mean_squared_error(Y_test4,KNNPREDHigh)

# # Calculate the RMSE
# rmseknnHigh = np.sqrt(mse)

# # Calculate the R-squared
# r2KnnHigh = r2_score(Y_test4,KNNPREDHigh)
# maeknnHigh =  metrics.mean_absolute_error(Y_test4,KNNPREDHigh)

# print("MSE:", mseknnHigh)
# print("RMSE:",rmseknnHigh)
# print("R-squared:",r2KnnHigh)
# print("MAE:", maeknnHigh)


# # MSE	Lower is better
# # RMSE	Lower is better
# # R^2	Higher is better
# # MAE	Lower is better

# In[335]:


# MyKNNPred1 = knn.predict(np.array([780]).reshape(1,-1)) # <-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE



# MyKNNPred2 = knn.predict([[780]]) #<-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE


# # Analysis of better model found here is SVM and Linear Regressor decisionTree is flop here. 
# # Linear regressor mse = 177.80298234987615
# # linear regressor rmse = 13.313181177910515
# # linear regressor mae = 9.260471493846618
# # linear regressor r2 = 0.9954780246991811 
# 

# # we tested and predicted with only the data from 2007-11-27 to 2021-04-30

# # To get the corresponding  stock price details Type = Adani Ports and Special Economic Zone Ld Stock price today

# # neural_network and neural network model evaluation  on CLOSE



"""the model's performance is reasonably good"""
# Create the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# Predict the values for the test data
predicted_values = model.predict(X_test)

# Print the predicted values
#print(predicted_values)

# Predict the values for the test data
# predicted_values = model.predict(X_test2)

""" neural network model evaluation """
# Calculate MSE
mse = mean_squared_error(Y_test, predicted_values)
print("Mean Squared Error (MSE):", mse)

# Calculate RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate MAE
mae = mean_absolute_error(Y_test, predicted_values)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared
r3 = r2_score(Y_test, predicted_values)
print("R-squared:", r3)


# # DNN ON CLOSE PRICE



# Create the neural network model  ON CLOSE PRICE >>>>>>>>
""" DNN ON CLOSE PRICE """
"""THIS IS MODEL IS  PERFORMING WELL  MODEL BASED ON THE METRICS"""
Dnnmodel = Sequential()
Dnnmodel.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
Dnnmodel.add(Dense(64, activation='relu'))
Dnnmodel.add(Dense(1, activation='linear'))

# Compile the model
Dnnmodel.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
Dnnmodel.fit(X_train, Y_train, epochs=10, batch_size=32)

# Predict the values for the test data
predicted_values = Dnnmodel.predict(X_test)

# Evaluate the model on the test data using metrics


mse = mean_squared_error(Y_test, predicted_values)
mae = mean_absolute_error(Y_test, predicted_values)
r2 = r2_score(Y_test, predicted_values)
rmse =  np.sqrt(mse)

# Print the predicted values and evaluation results
#print("Predicted values:")
#print(predicted_values)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared (R2) Score:", r2)

Dnnpred = Dnnmodel.predict(X_test)

CLOSE = Dnnmodel.predict([[780]]) # INput is OPEN, LOW, CLOSE for output HIGH PRICE
print("PREDICTED_VALUE:", CLOSE)







# # RNN  MODEL ON CLOSE PRICE




X_trainp = np.asarray(X_train)
X_testp = np.asarray(X_test)




"""These metrics indicate that your model performs well, as the errors are relatively low, 
and the model can explain a significant portion of the variance in the data."""

# Create the neural network model
# Rnnmodel = Sequential()
# Rnnmodel.add(LSTM(64, activation='relu', input_shape=(X_trainp.shape[1], 1)))
# Rnnmodel.add(Dense(64, activation='relu'))
# Rnnmodel.add(Dense(1, activation='linear'))

# # Compile the model
# Rnnmodel.compile(loss='mean_squared_error', optimizer='adam')

# # Reshape the input data for LSTM
# X_train_reshaped = X_trainp.reshape(X_trainp.shape[0], X_trainp.shape[1], 1)
# X_test_reshaped = X_testp.reshape(X_testp.shape[0], X_testp.shape[1], 1)

# # Fit the model to the training data
# Rnnmodel.fit(X_train_reshaped,Y_train, epochs=10, batch_size=32)


# # Predict the value for a single data point
# # input_data = np.array([735.00, 700, 740]).reshape(1, 3, 1)  #INPUT OPEN PRICE
# # input_data_reshaped = np.array(X_test_reshaped).reshape(1, 1, 1)
# #predicted_value = Rnnmodel.predict(input_data_reshaped)


# predicted_value = Rnnmodel.predict(X_test_reshaped)

# # Evaluate the model on the test data using metrics
# predicted_values = Rnnmodel.predict(X_test_reshaped)
# mse = mean_squared_error(Y_test, predicted_values)
# rmse =  np.sqrt(mse)
# mae = mean_absolute_error(Y_test, predicted_values)
# r2 = r2_score(Y_test, predicted_values)

# # Print the predicted value and evaluation results
# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# print("Mean Absolute Error:", mae)
# print("R-squared (R2) Score:", r2)
# print()
# #print("PREDICTED_VALUE:",predicted_value)
# print()


# # CNN ON CLOSE PRICE



# Create the neural network model
# cnn_model = Sequential()
# cnn_model.add(Reshape((X_trainp.shape[1],), input_shape=(X_trainp.shape[1], 1)))
# cnn_model.add(Dense(64, activation='relu'))
# cnn_model.add(Dense(64, activation='relu'))
# cnn_model.add(Dense(1, activation='linear'))

# # Compile the model
# cnn_model.compile(loss='mean_squared_error', optimizer='adam')

# # Reshape the input data for CNN
# X_train_reshaped = X_trainp.reshape(X_trainp.shape[0], X_trainp.shape[1])
# X_test_reshaped = X_testp.reshape(X_testp.shape[0], X_testp.shape[1])

# # Fit the model to the training data
# cnn_model.fit(X_train_reshaped, Y_train, epochs=10, batch_size=32)

# # Predict the value for a single data point

# #input_data = [[735.55]]  # INPUT OPEN PRICE

# predicted_value = cnn_model.predict(X_test_reshaped)

# # Evaluate the model on the test data using metrics
# predicted_values = cnn_model.predict(X_test_reshaped)
# mse = mean_squared_error(Y_test, predicted_values)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(Y_test, predicted_values)
# r2 = r2_score(Y_test, predicted_values)

# # Print the predicted value and evaluation results
# print()
# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# print("Mean Absolute Error:", mae)
# print("R-squared (R2) Score:", r2)

#print("PREDICTED_VALUE:", predicted_value)

# # Feedforward Neural Network on CLOSE price (Multi-Layer Perceptron):




""" These results indicate that the FFNN model has achieved relatively low mean squared error, root mean squared error,
and mean absolute error values, suggesting that it is performing well in predicting the target variable. The high R-squared (R2) 
score of 0.9971320639799347 further confirms the model's
strong performance in explaining the variance in the target variable."""

# Create the FFNN model
# ffnn_model = Sequential()
# ffnn_model.add(Dense(64, activation='relu', input_shape=(X_trainp.shape[1],)))
# ffnn_model.add(Dense(64, activation='relu'))
# ffnn_model.add(Dense(1, activation='linear'))

# # Compile the model
# ffnn_model.compile(loss='mean_squared_error', optimizer='adam')

# # Fit the model to the training data
# ffnn_model.fit(X_trainp, Y_train, epochs=10, batch_size=32)

# # Predict using the fitted FFNN model
# Y_pred_ffnn = ffnn_model.predict(X_testp)

# # Evaluate the FFNN model on the test data using metrics
# mse_ffnn = mean_squared_error(Y_test, Y_pred_ffnn)
# rmse_ffnn = np.sqrt(mse_ffnn)
# mae_ffnn = mean_absolute_error(Y_test, Y_pred_ffnn)
# r2_ffnn = r2_score(Y_test, Y_pred_ffnn)

# # Print the predicted values and evaluation results
# print("Predicted Values (FFNN):")
# #print(Y_pred_ffnn)
# print()
# print("Mean Squared Error (FFNN):", mse_ffnn)
# print("Root Mean Squared Error (FFNN):", rmse_ffnn)
# print("Mean Absolute Error (FFNN):", mae_ffnn)

# print("R-squared (R2) Score (FFNN):", r2_ffnn)



# # Neural network on HIGH PRICE




"""THIS IS MODEL IS  PERFORMING WELL  MODEL BASED ON THE METRICS"""

# Create the neural network model
# Highmodel = Sequential()
# Highmodel.add(Dense(64, activation='relu', input_shape=(X_train4.shape[1],)))
# Highmodel.add(Dense(64, activation='relu'))
# Highmodel.add(Dense(1, activation='linear'))

# # Compile the model
# Highmodel.compile(loss='mean_squared_error', optimizer='adam')

# # Fit the model to the training data
# Highmodel.fit(X_train4,Y_train4, epochs=10, batch_size=32)

# # Predict the values for the test data
# predicted_valuesHigh = Highmodel.predict(X_test4)

# # Print the predicted values
# #print(predicted_values)

# # Predict the values for the test data HIGH PRICE>>>>>

# """ neural network model evaluation """

# # Calculate MSE
# msecnn = mean_squared_error(Y_test4, predicted_valuesHigh)
# print("Mean Squared Error (MSE):", msecnn)

# # Calculate RMSE
# rmsecnn = np.sqrt(mse)
# print("Root Mean Squared Error (RMSE):", rmsecnn)

# # Calculate MAE
# maecnn = mean_absolute_error(Y_test4, predicted_valuesHigh)
# print("Mean Absolute Error (MAE):", mae)

# # Calculate R-squared
# r3cnn = r2_score(Y_test4, predicted_valuesHigh)
# print("R-squared:", r3cnn)



###################################################### all models input is open price ################################################################


# #  DEFFERENT MODELS ON CLOSE PRICES 

# # MODEL NUMBER 1 - LINEAR REGRESSION ON CLOSE PRICE



# mypredtest1 = Regressor.predict(np.array([[807]]).reshape(1,-1)) # <-- input is open price  // INput is OPEN, LOW, HIGH for output CLOSE PRICE


# mypredtes2 = Regressor.predict([[780]])  # <-- <-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE

# # MODEL NUMBER 2 - DECISIONTREE ON CLOSE



# myTreePredict1 = DecisionTRee.predict(np.array([780]).reshape(1,-1)) #<-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE





# myTreePredict2 = DecisionTRee.predict([[780]]) #<-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE


 # MODEL NUMBER 3 - SUPPORT VECTOR MACHINE ON CLOSE 


# mySVMpred1 = svmclassifier.predict(np.array([780]).reshape(1,-1)) #<-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE





# mySVMpred2 = svmclassifier.predict([[780]]) #<-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE



# # # MODEL 4 - K NEAREST NEIGHBOURS KNN supervised learning ON CLOSE PRICE



# MyKNNPred2 = knn.predict([[780]]) #<-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE






# MyKNNPred1 = knn.predict(np.array([780]).reshape(1,-1)) #<-- # INput is OPEN, LOW, HIGH for output CLOSE PRICE


# # # MODEL 5 - Neural network model unsupervised Learning ON CLOSE PRICE
# #  


# Neural_networkPred1 =  model.predict([[780]]) #<-- # INput is OPEN  for output CLOSE PRICE


# Neural_networkPred2 =  model.predict(np.array([780]).reshape(1,-1)) #<-- # INput is OPEN CLOSE PRICE



# #################################################################################################################


# # # Dnnmodel ON CLOSE



# DnnmodelPred =  Dnnmodel.predict([[780]]) #<-- <-- # INput is OPEN  for output CLOSE PRICE




# DnnmodelPred =  Dnnmodel.predict([[780]]) #<-- <-- # INput is OPEN  for output CLOSE PRICE



# #######################################################################################################################


# # # RNN  MODEL ON CLOSE PRICE




# RnnmodelPred =  Rnnmodel.predict([[780]]) #<-- <-- # INput is OPEN  for output CLOSE PRICE

# RnnmodelPred =  Rnnmodel.predict([[780]]) #<-- <-- # INput is OPEN  for output CLOSE PRICE



# ####################################################################################################################################


# # # CNN ON CLOSE PRICE



# cnn_modelPred =  cnn_model.predict([[780]]) #<-- <-- # INput is OPEN  for output CLOSE PRICE



# cnn_modelPred =  cnn_model.predict([[780]]) #<-- <-- # INput is OPEN  for output CLOSE PRICE


# ##################################################################################################################################


# # # Feedforward Neural Network on CLOSE price (Multi-Layer Perceptron):



# ffnn_modelPred =  ffnn_model.predict([[780]]) #<-- <-- # INput is OPEN  for output CLOSE PRICE


# ffnn_modelPred =  ffnn_model.predict([[780]]) #<-- <-- # INput is OPEN  for output CLOSE PRICE


# ####################################################################################################################################


# # 
# # # DEFFERENT MODELS ON HIGH PRICES 


# X_train4 ### THESE ARE THE FEATURES OR INPUTS TO GET THE HIGH PRICE 


# # # MODEL NUMBER 1 - LINEAR REGRESSION High price



# mypredHigh1 = RegressorHigh.predict(np.array([807]).reshape(1,-1)) #<-- <-- # INput is OPEN for output HIGH PRICE


# mypredHigh2 = RegressorHigh.predict([[807]]) #<-- <-- # INput is OPEN, for output HIGH PRICE


# # # MODEL NUMBER 2 - DECISIONTREE High price 


# mypredHighTree1= DecisionTReeHigh.predict(np.array([807]).reshape(1,-1)) #<-- <-- # INput is OPEN, for output HIGH PRICE



# mypredHighTree2 = DecisionTReeHigh.predict([[807]])  #<-- <-- # INput is OPEN, for output HIGH PRICE

# # #  MODEL NUMBER 3 - SUPPORT VECTOR MACHINE HIgh Price
# # 




# Y_predHighsvm1 = svmclassifierHigh.predict([[807]]) #<-- <-- # INput is OPEN, for output HIGH PRICE



# Y_predHighsvm2 = svmclassifierHigh.predict(np.array([807]).reshape(1,-1)) #<-- <-- # INput is OPEN, for output HIGH PRICE

# # # MODEL 4 - K NEAREST NEIGHBOURS KNN supervised learning HIgh Price



# KNNPREDHigh1 = Highknn.predict(np.array([807]).reshape(1,-1))  #<-- <-- # INput is OPEN, for output HIGH PRICE



# KNNPREDHigh2 = Highknn.predict([[807]])  #<-- <-- # INput is OPEN,  for output HIGH PRICE


# # #  MODEL 5 - Neural network model unsupervised Learning HIgh Price



# Neural_networkPred1 =  Highmodel.predict([[807]]) #<-- <-- # INput is OPEN,  for output HIGH PRICE

# Neural_networkPred1 =  Highmodel.predict(np.array([807]).reshape(1,-1)) #<-- <-- # INput is OPEN,  for output HIGH PRICE
# ############################################################          ##########################################################



# ############################################################           ####################################################
# # #  MODEL 6 - DNN HIGH price

# # Create the neural network model>>>>>>>>
# """THIS IS MODEL IS  PERFORMING WELL  MODEL BASED ON THE METRICS"""
# Dnnmodel = Sequential()
# Dnnmodel.add(Dense(64, activation='relu', input_shape=(X_train4.shape[1],)))
# Dnnmodel.add(Dense(64, activation='relu'))
# Dnnmodel.add(Dense(1, activation='linear'))

# # Compile the model
# Dnnmodel.compile(loss='mean_squared_error', optimizer='adam')

# # Fit the model to the training data
# Dnnmodel.fit(X_train4, Y_train4, epochs=10, batch_size=32)

# # Predict the values for the test data
# predicted_values = Dnnmodel.predict(X_test4)

# # Evaluate the model on the test data using metrics


# mse = mean_squared_error(Y_test4, predicted_values)
# mae = mean_absolute_error(Y_test4, predicted_values)
# r2 = r2_score(Y_test4, predicted_values)
# rmse =  np.sqrt(mse)

# # Print the predicted values and evaluation results
# #print("Predicted values:")
# #print(predicted_values)
# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# print("Mean Absolute Error:", mae)
# print("R-squared (R2) Score:", r2)

# Dnnpred = Dnnmodel.predict(X_test4)

# HIGH = Dnnmodel.predict([[807]]) # INput is OPEN, for output HIGH PRICE
# print("PREDICTED_VALUE:", HIGH)

# #############################################################                 #######################################################




# #####################################################          ###############################################################
# # #  MODEL 7 - RNN  HIGH price

# X_train4 = np.asarray(X_train4)
# X_test4 = np.asarray(X_test4)

# """These metrics indicate that your model performs well, as the errors are relatively low, 
# and the model can explain a significant portion of the variance in the data."""

# # Create the neural network model
# Rnnmodel = Sequential()
# Rnnmodel.add(LSTM(64, activation='relu', input_shape=(X_train4.shape[1], 1)))
# Rnnmodel.add(Dense(64, activation='relu'))
# Rnnmodel.add(Dense(1, activation='linear'))

# # Compile the model
# Rnnmodel.compile(loss='mean_squared_error', optimizer='adam')

# # Reshape the input data for LSTM
# X_train_reshaped = X_train4.reshape(X_train4.shape[0], X_train4.shape[1], 1)
# X_test_reshaped = X_test4.reshape(X_test4.shape[0], X_test4.shape[1], 1)

# # Fit the model to the training data
# Rnnmodel.fit(X_train_reshaped,Y_train4, epochs=10, batch_size=32)


# # Predict the value for a single data point
# # input_data = np.array([735.00, 700, 740]).reshape(1, 3, 1)  #INPUT OPEN PRICE
# # input_data_reshaped = np.array(X_test_reshaped).reshape(1, 1, 1)
# #predicted_value = Rnnmodel.predict(input_data_reshaped)


# predicted_value = Rnnmodel.predict(X_test_reshaped)

# # Evaluate the model on the test data using metrics
# predicted_values = Rnnmodel.predict(X_test_reshaped)
# mse = mean_squared_error(Y_test4, predicted_values)
# rmse =  np.sqrt(mse)
# mae = mean_absolute_error(Y_test4, predicted_values)
# r2 = r2_score(Y_test4, predicted_values)

# # Print the predicted value and evaluation results
# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# print("Mean Absolute Error:", mae)
# print("R-squared (R2) Score:", r2)
# print()
# #print("PREDICTED_VALUE:",predicted_value)
# print()

# RnnHigh = Rnnmodel.predict(np.array([807]).reshape(1, 1)) #<-- <-- # INput is OPEN for output HIGH PRICE
# print("PREDICTED_VALUE:",RnnHigh)

# RnnHigh = Rnnmodel.predict(np.array([807]).reshape(1,1)) #<-- <-- # INput is OPEN,  for output HIGH PRICE
# print("PREDICTED_VALUE:",RnnHigh)
# #####################################################              #######################################################






# ###################################################         ##########################################################
# # # MODEL 8 - CNN MODEL HIGH 

# # Create the neural network model
# cnn_model = Sequential()
# cnn_model.add(Reshape((X_train4.shape[1],), input_shape=(X_train4.shape[1], 1)))
# cnn_model.add(Dense(64, activation='relu'))
# cnn_model.add(Dense(64, activation='relu'))
# cnn_model.add(Dense(1, activation='linear'))

# # Compile the model
# cnn_model.compile(loss='mean_squared_error', optimizer='adam')

# # Reshape the input data for CNN
# X_train_reshaped = X_train4.reshape(X_train4.shape[0], X_train4.shape[1])
# X_test_reshaped = X_test4.reshape(X_test4.shape[0], X_test4.shape[1])

# # Fit the model to the training data
# cnn_model.fit(X_train_reshaped, Y_train4, epochs=10, batch_size=32)

# # Predict the value for a single data point

# #input_data = [[735.55]]  # INPUT OPEN PRICE

# predicted_value = cnn_model.predict(X_test_reshaped)

# # Evaluate the model on the test data using metrics
# predicted_values = cnn_model.predict(X_test_reshaped)
# mse = mean_squared_error(Y_test4, predicted_values)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(Y_test4, predicted_values)
# r2 = r2_score(Y_test4, predicted_values)

# # Print the predicted value and evaluation results
# print()
# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# print("Mean Absolute Error:", mae)
# print("R-squared (R2) Score:", r2)

# #print("PREDICTED_VALUE:", predicted_value)


# cnn_High = cnn_model.predict([[807]]) #<-- <-- # INput is OPEN, for output HIGH PRICE
# print("PREDICTED_VALUE:",cnn_High)


# cnn_High = cnn_model.predict([[807]]) #<-- <-- # INput is OPEN, for output HIGH PRICE
# print("PREDICTED_VALUE:",cnn_High)


# #####################################################            ##################################################################




# ############################################################           ############################################################3
# # # Feedforward Neural Network on HIGH price (Multi-Layer Perceptron):

# """ These results indicate that the FFNN model has achieved relatively low mean squared error, root mean squared error,
# and mean absolute error values, suggesting that it is performing well in predicting the target variable. The high R-squared (R2) 
# score of 0.9971320639799347 further confirms the model's
# strong performance in explaining the variance in the target variable."""

# # Create the FFNN model
# ffnn_model = Sequential()
# ffnn_model.add(Dense(64, activation='relu', input_shape=(X_train4.shape[1],)))
# ffnn_model.add(Dense(64, activation='relu'))
# ffnn_model.add(Dense(1, activation='linear'))

# # Compile the model
# ffnn_model.compile(loss='mean_squared_error', optimizer='adam')

# # Fit the model to the training data
# ffnn_model.fit(X_train4, Y_train4, epochs=10, batch_size=32)

# # Predict using the fitted FFNN model
# Y_pred_ffnn = ffnn_model.predict(X_test4)

# # Evaluate the FFNN model on the test data using metrics
# mse_ffnn = mean_squared_error(Y_test4, Y_pred_ffnn)
# rmse_ffnn = np.sqrt(mse_ffnn)
# mae_ffnn = mean_absolute_error(Y_test4, Y_pred_ffnn)
# r2_ffnn = r2_score(Y_test4, Y_pred_ffnn)

# # Print the predicted values and evaluation results
# print("Predicted Values (FFNN):")
# #print(Y_pred_ffnn)
# print()
# print("Mean Squared Error (FFNN):", mse_ffnn)
# print("Root Mean Squared Error (FFNN):", rmse_ffnn)
# print("Mean Absolute Error (FFNN):", mae_ffnn)

# print("R-squared (R2) Score (FFNN):", r2_ffnn)

# PREDICTEDHIGH = ffnn_model.predict([[807]])  #<-- <-- # INput is OPEN for output HIGH PRICE
# print("PREDICTED_VALUE:",PREDICTEDHIGH)

# #######################################################          ########################################################


# print("CODE EXECUTED")
