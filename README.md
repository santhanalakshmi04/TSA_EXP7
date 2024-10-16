# NAME : Santhana Lakshmi k
# REGISTER NUMBER : 212222240091
# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Read the power consumption CSV file into a DataFrame
file_path = '/mnt/data/powerconsumption.csv'
data = pd.read_csv(file_path)

# Convert 'Datetime' to datetime format and set it as the index
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m/%d/%Y %H:%M')
data.set_index('Datetime', inplace=True)

# Use data for Zone 1 Power Consumption (you can change this to Zone 2 or Zone 3 as needed)
data = data[['PowerConsumption_Zone1']]

# Print the first few rows of the dataset as "Given Data"
print("GIVEN DATA:")
print(data.head())

# Perform Augmented Dickey-Fuller test to check for stationarity
result = adfuller(data['PowerConsumption_Zone1'])
print("\nAugmented Dickey-Fuller test:")
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Split the data into training and testing sets (80% train, 20% test)
train_data = data.iloc[:int(0.8 * len(data))]
test_data = data.iloc[int(0.8 * len(data)):]

# Fit an AutoRegressive (AR) model with 13 lags (or adjust based on your data size)
lag_order = 13
model = AutoReg(train_data['PowerConsumption_Zone1'], lags=lag_order)
model_fit = model.fit()

# Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plot_acf(data['PowerConsumption_Zone1'])
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['PowerConsumption_Zone1'])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Compare the predictions with the test data
mse = mean_squared_error(test_data['PowerConsumption_Zone1'], predictions)
print('Mean Squared Error (MSE):', mse)

# Plot the test data and predictions
plt.plot(test_data.index, test_data['PowerConsumption_Zone1'], label='Test Data', color='blue')
plt.plot(test_data.index, predictions, label='Predictions', color='orange')
plt.xlabel('Date')
plt.ylabel('Power Consumption (Zone 1)')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

### OUTPUT:

GIVEN DATA:

![image](https://github.com/user-attachments/assets/2bba1236-a117-4a83-90d8-b5cd5e16241b)

Augmented Dickey-Fuller test:

![image](https://github.com/user-attachments/assets/510dafeb-574a-4af6-9071-95ad1cc76d89)


PACF - ACF:

![image](https://github.com/user-attachments/assets/ba8f504f-5642-4a62-948b-134971b5c31b)

![image](https://github.com/user-attachments/assets/1c7880fd-1073-4f57-9635-ef28d0c3609e)

Mean Squared Error:

![image](https://github.com/user-attachments/assets/0f29e683-f0e9-40ae-8e92-c3582b6f59d1)

FINIAL PREDICTION:

![image](https://github.com/user-attachments/assets/7f7eb974-91df-4429-9d99-6a5104db77ab)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
