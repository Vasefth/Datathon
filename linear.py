import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import statsmodels.api as sm


drive.mount('/content/drive')

# Διάβασμα δεδομένων από το excel
data = pd.read_csv('/content/drive/MyDrive/DataThonTeam/powerconsumption.csv')
data.head()

data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
daily_data = data.resample('D').mean()

fig, ax1 = plt.subplots()

ax1.plot(daily_data.index, daily_data['PowerConsumption_Zone1'], 'b', label='Consumption')
ax1.set_xlabel('Date')
ax1.set_ylabel('kWh', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()

ax2.plot(daily_data.index, daily_data['Temperature'], 'r', label='Temperature')
ax2.set_ylabel('ºC', color='r')
ax2.tick_params('y', colors='r')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper left')

plt.show()

X = np.expand_dims(np.array(daily_data['Temperature']), axis=1)
y = np.array(daily_data['PowerConsumption_Zone1'])
reg = LinearRegression().fit(X,y)

print(f'Learned function: consumption = temperature*{reg.coef_[0]:.2f} + {reg.intercept_:.2f}')
print(f'R^2 score: {reg.score(X,y)}')

f = lambda x: x*reg.coef_[0] + reg.intercept_
plt.plot(daily_data['Temperature'], daily_data['Temperature'].apply(f), 'red', label='Learned function')
plt.scatter(daily_data['Temperature'], daily_data['PowerConsumption_Zone1'], marker='+', label='Data points')
plt.xlabel('Temperature (ºC)')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.show()

plt.plot(daily_data.index, daily_data['PowerConsumption_Zone1'], label='Actual')
plt.plot(daily_data.index, daily_data['Temperature'].apply(f), label='Predicted')
plt.xlabel('Date')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.show()

def poly_reg(x, t, deg):
    '''
    params:
        x - input data (2D array)
        t - target data (1D array)
        deg - polynomial degree
    '''
    coefs = np.polyfit(X.reshape(-1), y, deg=deg)[::-1]
    print('Learned Function: y =', ' + '.join([f'{coefs[i]:.2f}*(x**{i})' for i in range(len(coefs))]))
    func = lambda x: sum([coefs[i]*(x**i) for i in range(len(coefs))])
    return np.vectorize(func) #Return the vectorized regression function

g = poly_reg(X, y, 2)

print(f'R^2 score: {r2_score(y,g(X))}')

plt.plot(daily_data['Temperature'].sort_values(), daily_data['Temperature'].sort_values().apply(g), 'red', label='Learned function')
plt.scatter(daily_data['Temperature'], daily_data['PowerConsumption_Zone1'], marker='+', label='Data points')
plt.xlabel('Temperature (ºC)')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.show()

plt.plot(daily_data.index, daily_data['PowerConsumption_Zone1'], label='Actual')
plt.plot(daily_data.index, daily_data['Temperature'].apply(g), label='Predicted')
plt.xlabel('Date')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.show()

