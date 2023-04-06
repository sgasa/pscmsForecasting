from . import utils
import numpy as np
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.base import ForecastingHorizon

def fit(train_data, pred_steps, parameters):

  args = {}
  if parameters['seasonality'] == 'true':
    args['sp'] = 12
  
  
  forecaster = AutoETS(auto=True, **args)
  forecaster.fit(train_data)
  
  fit_index = np.arange(len(train_data))
  fit_horizon = ForecastingHorizon(fit_index, is_relative=False)
  fitted = forecaster.predict(fh=fit_horizon).flatten()
  
  forecast_index = np.arange(len(train_data), len(train_data) + pred_steps)
  forecasting_horizon = ForecastingHorizon(forecast_index, is_relative=False) 
  forecast = forecaster.predict(fh=forecasting_horizon).flatten()

  variance = forecaster.predict_var(fh=forecasting_horizon).values.flatten()
  forecast_error = np.sqrt(variance)
  
  metrics = utils.calculate_metrics(
    train_data, fitted
  )

  return fitted, forecast, forecast_error, metrics

