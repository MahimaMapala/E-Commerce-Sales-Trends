# Install libraries: pip install pandas prophet plotly
import pandas as pd
from prophet import Prophet
import plotly.express as px

#download from https://www.kaggle.com/datasets/carrie1/ecommerce-data)
df = pd.read_csv('data.csv', encoding='ISO-8859-1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.groupby(df['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
df.columns = ['ds', 'y']

# ML: Predict sales with Prophet
model = Prophet(yearly_seasonality=True)
model.fit(df)
future = model.make_future_dataframe(periods=7)  # Predict next 7 days
forecast = model.predict(future)
print(forecast[['ds', 'yhat']].tail())  # Predicted sales

# Visualization: Plot sales trend
fig = px.line(forecast, x='ds', y='yhat', title='Weekly Sales Forecast')
fig.add_scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual')
fig.show()