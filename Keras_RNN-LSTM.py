import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import seaborn as sns
import numpy as np
from collections import deque
import random
from sklearn.preprocessing import RobustScaler


df = pd.read_csv('sent_price_file.csv', sep = ',')

df.head()
df.info()
df.describe()


df = df.drop(columns = 'Unnamed: 0') # Remove index column
df = df.drop(columns = 'Volume')


df = df.set_index('Date')

df = df.dropna()


# Plotting values

_ = plt.plot(df.Close, color = 'blue')
_ = plt.plot(df.polarity, color = 'red')
_ = plt.legend()
plt.show()

# Checking for correlation between values


corr = df.corr()
f, ax = plt.subplots(figsize=(6, 3))
plt.title("Variable Correlation Plot")
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# Scaling the dataset for the neural network



  
    def finalize_df(df, test=True):
        df_edited = df.copy()
        steps = 60
    
        sc = MinMaxScaler(feature_range = (0,1))
        df = sc.fit_transform(df)
    
        seq_data = []  # this is a list that will CONTAIN the sequences
        prev_days = deque(maxlen=steps)  # deque removes old values as new ones come in
    
        for i in df: 
            prev_days.append([n for n in i[:-1]])  # do not include y
    
            # check for 60 days of previous data
            if len(prev_days) == steps:
                # seq data = [prev_days_data, target variable]
                seq_data.append([np.array(prev_days), i[-1]])  
        X = []
        y = []
    
        for seq, target in seq_data:  # iterating though the data
            X.append(seq)  # store the seqs for features
            y.append(target)  # y is the target
        
        if test == True:
            return np.array(X), y 
        else:
            return np.array(X), y, df_edited, sc


date = sorted(df.index.values)
percentage = sorted(df.index.values)[-int(0.3*len(date))]



validation_main_df = df[(df.index >= percentage)]
main_df = df[(df.index < percentage)]



x_train, y_train = finalize_df(main_df)
x_test, y_test, df_final, sc_0 = finalize_df(validation_main_df, test=False)

# Initiate neural network

model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1:]), dropout = 0.1, return_sequences = True))
model.add(LSTM(50, dropout = 0.1))

model.add(Dense(1, activation = 'tanh'))
model.compile(loss='mse', optimizer='adam', metrics = ['mae', 'mse'])
print(model.summary())
history = model.fit(x_train, y_train, epochs=5, batch_size=100)

score_train = model.evaluate(x_train, y_train, verbose = 0)

score_test = model.evaluate(x_test, y_test, verbose = 0)


# Plot the loss by epoch

_ = plt.plot(history.history['loss'], label='train')
_ = plt.legend()
plt.show()

# Predict 

y_pred = model.predict(x_test)

# Revert back to normal value from the scaled value
padding = np.zeros((38, 1))
y_pred_result = pd.DataFrame(data=padding)
y_pred_result['9'] = y_pred
y_pred_inv = sc.inverse_transform(y_pred_result)

y_test_results = pd.DataFrame(data=padding)
y_test_results['9'] = y_test
y_test_inv = sc.inverse_transform(y_test_results)

# check the root mean squared error



# Plot y pred vs y actual

_ = plt.plot(y_test_inv[:,-1], label = 'Predicted Value')
_ = plt.plot(y_pred_inv[:, -1], label = 'Actual')
_ = plt.title('Predictions vs. Actual')
_ = plt.ylabel('Closing Price (Unscaled)')
_ = plt.legend()
plt.show()






model2 = Sequential()
model2.add(LSTM(50, input_shape=(x_train.shape[1:]), dropout = 0.1, return_sequences = True))
model2.add(LSTM(50, dropout = 0.1, return_sequences = True))
model.add(Dense(1))
model2.compile(loss='mse', optimizer='adam', metrics = ['mae', 'mse'])
print(model2.summary())
history2 = model2.fit(x_train, y_train, epochs=500, batch_size=50)
