import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import seaborn as sns
import numpy as np


df = pd.read_csv('sent_price_file.csv', sep = ',')


# Data Cleaning so there are no null values

df['Open'].replace(0, np.nan, inplace = True)
df['Open'].fillna(method ='ffill', inplace = True)

"""df['polarity'].replace(0, np.nan, inplace = True)
df['polarity'].fillna(method ='ffil', inplace = True)
"""



# Scaling the dataset for the neural network


# Create scaling object

sc = MinMaxScaler(feature_range = (0,1))


price = df['Open'].values.reshape(-1, 1).astype('float32')
sent = df['polarity'].values.reshape(-1, 1).astype('float32')



price_sc = sc.fit_transform(price)

# Split data
# Splitting Training Sheets to Test Sheets
from sklearn.model_selection import train_test_split 
train, test = train_test_split(price_sc, test_size = 0.25, random_state = 0) 


# Create a function for making this dataset have lookback

def look_back_df(df, look_back, sent):
    x, y = [], []
    for i in range(len(df) - look_back):
        a = df[i:i + look_back, 0]
        np.append(a, sent[i])
        x.append(a)
        y.append(df[i + look_back, 0])
    print(len(y))
    return np.array(x), np.array(y)


# Look back dataset
    
x_train, y_train = look_back_df(train, 1, sent[0:len(train)])
x_test, y_test = look_back_df(test, 1, sent[len(train) : len(price_sc)])


# Reshape train data

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


# Initiate neural network

model = Sequential()
model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(x_train, y_train, epochs=100, batch_size=50, validation_data=(x_test, y_test), verbose=0, shuffle=False)


# Plot the loss by epoch

y_pred = model.predict(x_test)
_ = plt.plot(history.history['loss'], label='train')
_ = plt.plot(history.history['val_loss'], label='test')
_ = plt.legend()
plt.show()


# Revert back to normal value from the scaled value

y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_inv = sc.inverse_transform(y_test.reshape(-1,1))

# check the root mean squared error

rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))



# Plot y pred vs y actual

_ = plt.plot(y_pred, label = 'Predicted Value')
_ = plt.plot(y_test_inv, label = 'Actual')
_ = plt.legend()
plt.show()