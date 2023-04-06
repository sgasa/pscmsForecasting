import numpy as np
from . parameter_validation import validate



def forecast(train_data, pred_steps, model, parameters):
  """
  train_data: (list like) of training data points
  pred_steps: (int) number of prediction steps
  model: one of ['ES', 'AutoETS', 'AutoArima', 'Prophet', 'NN', 'Boost']
  parameters: a dictionary with keys and possible values as given below

  For ES (Exponential Smoothing):
    parameters = {
        'trend': ['add', 'mul', 'none'],
        'damped_trend': ['true', 'false'],
        'seasonal': ['add', 'mul', 'none'],
        'seasonal_periods': ['2', '3', '4', '6', '12'],
        'use_boxcox': ['true', 'false', 'log']
    }
  For AutoETS:
    parameters = {
      'seasonality': [true', 'false']
    }
  For AutoArima:
    parameters = {
      'seasonality': ['true', 'false']
    }
  For Prophet:
    parameters = {
      'seasonality': [multiplicative', 'additive', 'none']
    }
  For NN (Neural Network):
    parameters = {
      'rnn_type': [lstm', 'rnn', 'gru'],
      'epochs': ['50', '100', '500', '1000', '2000'],
      'hidden_size': ['2', '4', '8', '12', '16', '24', '32', '64'],
      'learning_rate': ['0.001', '0.005', '0.01', '0.05', '0.1'],
      'sequence_length': ['2', '4', '8', '12', '16', '24', '32', '64'],
      'optimiser': ['Adam', 'SGD']
    }
  For Boost:
    parameters = {
      'sequence_length' : ['2', '4', '8', '12', '16', '24', '32', '64'],
      'num_leaves' : ['2', '4', '8', '12', '16', '24', '32', '64'],
      'max_depth' : ['2', '4', '8', '12', '16', '24', '32', '64'],
      'learning_rate' : ['0.001', '0.005', '0.01', '0.05', '0.1'],
      'num_boost_round': ['2', '4', '8', '16', '32', '64', '128', '256'],
      'early_stopping_rounds': ['0', '2', '4', '8', '16', '32', '64', '128'],
    }
  """

  validate(train_data, pred_steps, model, parameters)
  train_data = np.array(train_data).flatten()


  if model == 'ES':
    try:
      from . import exp_smooth 
    except:
      raise ImportError(f"Exponential Smoothing model is not supported in this platform")
    return exp_smooth.fit(train_data, pred_steps, parameters)

  elif model ==  'AutoETS':
    try:
      from . import autoETS
    except:
      raise ImportError(f"AutoETS model is not supported in this platform")
    return autoETS.fit(train_data, pred_steps, parameters)

  elif model ==  'AutoArima':
    try:
      from . import autoArima
    except:
      raise ImportError(f"AutoArima model is not supported in this platform")
    return autoArima.fit(train_data, pred_steps, parameters)

  elif model ==  'Prophet':
    try:
      from . import prophet 
    except:
      raise ImportError(f"Prophet model is not supported in this platform")
    return prophet.fit(train_data, pred_steps, parameters)

  elif model ==  'NN':
    try:
      from . import nnet 
    except:
      raise ImportError(f"Neural Network model is not supported in this platform")
    return nnet.fit(train_data, pred_steps, parameters)

  elif model == 'Boost':
    try:
      from . import boost 
    except:
      raise ImportError(f"Boost model is not supported in this platform")
    return boost.fit(train_data, pred_steps, parameters)





