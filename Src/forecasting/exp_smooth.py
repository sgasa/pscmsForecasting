import numpy as np
from statsmodels.tsa.api import (
  ExponentialSmoothing, SimpleExpSmoothing
)
from . import utils

def fit_naive(train_data, pred_steps):

  mean_value = np.mean(train_data)
  fitted = np.repeat(mean_value, len(train_data))
  forecast = np.repeat(mean_value, pred_steps)
  forecast_error = np.repeat(0, pred_steps)
  metrics = utils.calculate_metrics(
    train_data, fitted
  )

  return fitted, forecast, forecast_error, metrics


def fit_simple_exp_smooth(train_data, pred_steps):

  fit = SimpleExpSmoothing(train_data).fit()

  if fit.mle_retvals.success == False:
    raise RuntimeError(f'SimpleExpSmoothing did not converge: {fit.mle_retvals.message}')
    
  fitted = fit.fittedvalues
  forecast = fit.forecast(pred_steps)
  forecast_error = fit.simulate(nsimulations=pred_steps, repetitions=1000
    ).std(axis=1)

  metrics = utils.calculate_metrics(
    train_data, fitted
  )


  return fitted, forecast, forecast_error, metrics

def fit_exp_smooth(train_data, pred_steps, parameters):

  trend_map = {
    'add':'add','mul':'mul','none':None
  }
  damped_trend_map = {
    'true':True, 'false':False
  }
  seasonal_map = {
    'add':'add','mul':'mul','none':None
  }
  def use_boxcox_map(value):
    if value == 'true':
      return True
    elif value == 'false':
      return False
    elif value == 'log':
      return False
    else:
      return int(value)

  trend = trend_map[parameters['trend']]
  damped_trend = damped_trend_map[parameters['damped_trend']]
  seasonal = seasonal_map[parameters['seasonal']]
  seasonal_periods = int(parameters['seasonal_periods'])
  use_boxcox = use_boxcox_map(parameters['use_boxcox'])

  fit = ExponentialSmoothing(
    train_data, 
    trend=trend,
    damped_trend=damped_trend,
    seasonal=seasonal,
    seasonal_periods=seasonal_periods,
    use_boxcox=use_boxcox,
    initialization_method='estimated'
  ).fit()

  if fit.mle_retvals.success == False:
    raise RuntimeError(f'ExponentialSmoothing did not converge: {fit.mle_retvals.message}')
    
  fitted = fit.fittedvalues
  forecast = fit.forecast(pred_steps)
  forecast_error = fit.simulate(nsimulations=pred_steps, repetitions=1000
    ).std(axis=1)

  metrics = utils.calculate_metrics(
    train_data, fitted
  )


  return fitted, forecast, forecast_error, metrics

def fit(train_data, pred_steps, parameters):

  train_data_len = len(train_data)

  if train_data_len < 3:
    parameters = None
    return fit_naive(train_data, pred_steps)

  elif train_data_len < 10:
    parameters = None
    return fit_simple_exp_smooth(train_data, pred_steps)

  else:
    if parameters['seasonal'] != 'none':
      period = parameters['seasonal_periods']
      if ((train_data_len < 10 + np.ceil(period / 2)) or
          (train_data_len < 2 * period)):
        parameters['seasonal'] = 'none'
        parameters['seasonal_periods'] = 0
    return fit_exp_smooth(train_data, pred_steps, parameters)
