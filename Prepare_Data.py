import pandas as pd
import numpy as np


## Main data (Price, onshore, load)
data = pd.read_csv("./time_series_30min_singleindex.csv")

columns = data.columns[data.columns.str.contains('IE_')].to_list()
columns.append('utc_timestamp')
columns.append('GB_GBN_load_actual_tso')
columns.append('GB_GBN_solar_generation_actual')
columns.append('GB_GBN_wind_generation_actual')
df_complete = data[columns]
df_complete = df_complete.set_index('utc_timestamp')
del df_complete.index.name
df_complete = df_complete.dropna()
df_complete.index = pd.to_datetime(df_complete.index)
df_complete.head()

column_names = {
    'IE_load_actual_entsoe_transparency':'load',
    'IE_load_forecast_entsoe_transparency':'load_forecast',
    'IE_wind_onshore_generation_actual':'onshore',
    'IE_sem_load_actual_entsoe_transparency':'load_sem',
    'IE_sem_load_forecast_entsoe_transparency': 'load_forecast_sem',
    'IE_sem_price_day_ahead':'price_da_sem',
    'IE_sem_wind_onshore_generation_actual':'onshore_sem',
    'GB_GBN_load_actual_tso': 'load_gb',
    'GB_GBN_solar_generation_actual': 'solar_gb',
    'GB_GBN_wind_generation_actual': 'wind_gb'
}

df_complete = df_complete.rename(columns=column_names)

# without _sem < with _sem
# ->without _sem must be only ROI

df_complete[['price_da_sem','onshore_sem', 'load_forecast_sem', 'load_sem', 'onshore', 'load_forecast', 'load',  'load_gb', 'wind_gb', 'solar_gb']]
df = df_complete[['price_da_sem','onshore_sem', 'load_forecast_sem', 'load_sem', 'load_gb', 'wind_gb', 'solar_gb']]

#df.plot(figsize=(16,9), colors=['black','lightskyblue', 'green', 'red'])


## Weather Data

weather_data = pd.read_csv("./weather_data.csv")

weather_data_columns = weather_data.columns[weather_data.columns.str.contains('IE')].to_list()
weather_data_columns.append('utc_timestamp')
df_weather_data = weather_data[weather_data_columns]
df_weather_data = df_weather_data.set_index('utc_timestamp')
del df_weather_data.index.name
df_weather_data = df_weather_data.dropna()
df_weather_data.index = pd.to_datetime(df_weather_data.index)

column_names_weather={
    'IE_temperature':'temp',
    'IE_radiation_direct_horizontal':'rad_direct',
    'IE_radiation_diffuse_horizontal':'rad_diffuse'
}

df_weather_data = df_weather_data.rename(columns=column_names_weather)

df = df.resample("60min").mean()
df.index.intersection(df_weather_data.index)
df[df.index.isin(df.index.intersection(df_weather_data.index))]
result = pd.concat([df[df.index.isin(df.index.intersection(df_weather_data.index))], df_weather_data[df_weather_data.index.isin(df.index.intersection(df_weather_data.index))]], axis=1, sort=False)

result.to_pickle("./data.pkl")
#pd.read_pickle("./data.pkl")

## Heat Demand
# -> not in time range 2015-2016

