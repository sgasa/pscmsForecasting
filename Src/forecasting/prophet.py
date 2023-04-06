from . import utils
import numpy as np
import pandas as pd
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.base import ForecastingHorizon

def fit(train_data, pred_steps, parameters):

  args = {}
  if parameters['seasonality'] == 'multiplicative':
    args['seasonality_mode'] = 'multiplicative'
    args['yearly_seasonality'] = True
  elif parameters['seasonality'] == 'additive':
    args['seasonality_mode'] = 'additive'
    args['yearly_seasonality'] = True
  elif parameters['seasonality'] == 'none':
    args['yearly_seasonality'] = False
  
  forecaster = Prophet(weekly_seasonality=False, daily_seasonality=False, **args)

  # Convert the data to dataframe, Prophet could not 
  # understand seasonality with no explicit date information
  fit_index = pd.date_range(
    start='2000-01-01', periods=len(train_data), freq='M'
  )
  train_data_df = pd.DataFrame(
    train_data,
    columns=['x'],
    index = fit_index
  )

  forecaster.fit(train_data_df)
  
  fit_horizon = ForecastingHorizon(fit_index, is_relative=False)
  fitted = forecaster.predict(fh=fit_horizon).values.flatten()
  
  forecast_index = pd.date_range(
    start=fit_index[-1], periods=pred_steps+1, freq='M'
  )[1:]
  forecasting_horizon = ForecastingHorizon(forecast_index, is_relative=False) 
  forecast = forecaster.predict(fh=forecasting_horizon).values.flatten()
  
  variance = forecaster.predict_var(fh=forecasting_horizon).values.flatten()
  forecast_error = np.sqrt(variance)

  metrics = utils.calculate_metrics(
    train_data, fitted
  )

  return fitted, forecast, forecast_error, metrics



  
  


  
