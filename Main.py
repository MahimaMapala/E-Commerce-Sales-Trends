import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load and preprocess data
try:
    df = pd.read_csv('data.csv', encoding='ISO-8859-1')
except FileNotFoundError:
    print("Error: 'data.csv' not found. Download from https://www.kaggle.com/datasets/carrie1/ecommerce-data")
    exit()

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[df['Quantity'] > 0]  # Remove returns
daily_sales = df.groupby(df['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
daily_sales.columns = ['ds', 'y']

# Data validation
if len(daily_sales) < 14:  # Need at least 2 weeks for weekly decomposition
    print(f"Error: Only {len(daily_sales)} days of data. Need at least 14 days.")
    exit()
print(f"Dataset size: {len(daily_sales)} days")

# Feature engineering
daily_sales['day_of_week'] = daily_sales['ds'].apply(lambda x: x.weekday())
daily_sales['month'] = daily_sales['ds'].apply(lambda x: x.month)

# Decompose time series (weekly cycle: period=7)
decomp = seasonal_decompose(daily_sales['y'], model='additive', period=7)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
decomp.trend.plot(ax=ax1, title='Trend')
decomp.seasonal.plot(ax=ax2, title='Seasonality (Weekly)')
decomp.resid.plot(ax=ax3, title='Residuals')
plt.tight_layout()
plt.show()

# Split data for training/testing
train_size = int(len(daily_sales) * 0.8)
train, test = daily_sales[:train_size], daily_sales[train_size:]

# ML: Train Prophet model with regressors
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.add_regressor('day_of_week')
model.add_regressor('month')
model.fit(train)

# Predict and evaluate
future = model.make_future_dataframe(periods=len(test))
future['day_of_week'] = future['ds'].apply(lambda x: x.weekday())
future['month'] = future['ds'].apply(lambda x: x.month)
forecast = model.predict(future)
mae = mean_absolute_error(test['y'], forecast['yhat'][-len(test):])
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Visualization: Interactive plot with confidence intervals
fig = go.Figure()
fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], name='Train', mode='lines'))
fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], name='Test', mode='lines'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', mode='lines'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', 
                         mode='none', name='Upper CI', opacity=0.2))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', 
                         mode='none', name='Lower CI', opacity=0.2))
fig.update_layout(title='E-Commerce Sales Forecast (Weekly Analysis)',
                  xaxis_title='Date', yaxis_title='Sales Quantity')
fig.show()