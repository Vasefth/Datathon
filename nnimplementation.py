# imports
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


drive.mount('/content/drive')

# Διάβασμα δεδομένων από το excel
data = pd.read_csv('/content/drive/MyDrive/DataThonTeam/powerconsumption.csv')
data.head()

X = data
y = data['PowerConsumption_Zone1']

y = tf.keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()

model.add(Dense(8, input_shape=(4,), activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Normalize the dataset
# Specify the columns to normalize
columns_to_normalize = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']

# Extract the columns to be normalized
data_to_normalize = data[columns_to_normalize]

# Perform Min-Max scaling on the selected columns
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_to_normalize)

# Create a DataFrame from the normalized data
normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)

# Add the other columns back to the normalized DataFrame
for column in data.columns:
    if column not in columns_to_normalize:
        normalized_df[column] = data[column]


# Reorder the columns to match the original DataFrame's order, excluding 'Datetime'
data = data[[col for col in data.columns if col != 'Datetime']]

# Display the normalized DataFrame
data.head()

import pandas as pd
from sklearn.model_selection import train_test_split



# Splitting the data into features and targets
X = data.iloc[:, :-3].values  # All columns except the last three are features
y = data.iloc[:, -3:].values  # The last three columns are targets

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be 3D [samples, timesteps, features] for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# LSTM Network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))  # Output layer with units equal to the number of target variables

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", test_loss)

# Make predictions
predictions = model.predict(X_test)

# (Optional) Display some predictions
print(predictions[:5])

import matplotlib.pyplot as plt
import numpy as np

# Sample size for plotting
sample_size = 20  # Adjust this as needed
actual_sample = y_test[:sample_size]
predicted_sample = predictions[:sample_size]

# Setting up the bar chart
barWidth = 0.35  # Width of a bar

# Names of the last three columns
column_names = ['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']  # Replace with actual column names

# Plot for each of the last three columns
for i in range(3):
    plt.figure(figsize=(15, 6))

    # Set position of bar on X axis
    r1 = np.arange(len(actual_sample))
    r2 = [x + barWidth for x in r1]

    # Make the plot for each target
    plt.bar(r1, actual_sample[:, i], color='blue', width=barWidth, edgecolor='grey', label='Actual ' + column_names[i])
    plt.bar(r2, predicted_sample[:, i], color='red', width=barWidth, edgecolor='grey', label='Predicted ' + column_names[i])


    # Add labels, title, and legend
    plt.xlabel('Sample Number', fontweight='bold')
    plt.ylabel('Values')
    plt.title('Comparison of Actual and Predicted Values for ' + column_names[i])
    plt.xticks([r + barWidth / 2 for r in range(len(actual_sample))], range(sample_size))
    plt.legend()

    # Show the plot
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Ensure 'predictions' and 'y_test' are available and properly scaled back to original scale
# Let's take a sample of the data for plotting for clarity
sample_size = 50  # Adjust the size as needed
actual_sample = y_test[:sample_size]
predicted_sample = predictions[:sample_size]

# Setting up the bar chart
barWidth = 0.4  # Width of a bar

for zone in range(actual_sample.shape[1]):  # Iterate over each zone
    plt.figure(figsize=(12, 6))

    # Set position of bar on X axis
    r1 = np.arange(len(actual_sample))
    r2 = [x + barWidth for x in r1]

    # Make the plot for each zone
    plt.bar(r1, actual_sample[:, zone], color='blue', width=barWidth, edgecolor='grey', label='Actual Zone ' + str(zone + 1))
    plt.bar(r2, predicted_sample[:, zone], color='red', width=barWidth, edgecolor='grey', label='Predicted Zone ' + str(zone + 1))

    # Add labels, title, and legend
    plt.xlabel('Sample Number', fontweight='bold')
    plt.ylabel('Power Consumption')
    plt.title('Comparison of Actual and Predicted Values for Zone ' + str(zone + 1))
    plt.xticks([r + barWidth/2 for r in range(len(actual_sample))], range(sample_size))
    plt.legend()

    # Show the plot
    plt.show()

import matplotlib.pyplot as plt

# Assuming predictions and y_test are available and properly scaled back to original scale
# Let's take a sample of the data for plotting for clarity
sample_size = 20  # You can adjust this size
actual_sample = y_test[:sample_size]
predicted_sample = predictions[:sample_size]

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(actual_sample, label='Actual Values', color='blue', marker='o')
plt.plot(predicted_sample, label='Predicted Values', color='red', linestyle='dashed', marker='x')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Sample Number')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()
