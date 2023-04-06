import numbers

def validate(values, n, model, parameters):
  if (not hasattr(values, '__iter__')):
    raise TypeError("Variable 'values' must be an iterable")

  if (not all(
    [isinstance(i, numbers.Number) for i in values]
    )):
    raise ValueError("Variable 'values' must contain numbers only")

  if ((not isinstance(n, int)) or (n<1)):
    raise ValueError("Variable 'n' must be a positive integer")

  if model == 'ES':
    validateES(parameters)
  elif model == 'AutoETS':
    validateAutoETS(parameters)
  elif model == 'AutoArima':
    validateAutoArima(parameters)
  elif model == 'Prophet':
    validateProphet(parameters)
  elif model == 'NN':
    validateNN(parameters)
  elif model == 'Boost':
    validateBoost(parameters)
  else:
    raise ValueError(
      "Variable 'model' must be one of " + 
      "['ES', 'AutoETS', 'AutoArima', 'Prophet', 'NN', 'Boost']"
    )


def validateES(parameters):
  if ((not isinstance(parameters, dict)) or 
      (not all([key in parameters.keys() 
             for key in ['trend', 'damped_trend', 'seasonal',
                         'seasonal_periods', 'use_boxcox']]))
  ):
    error_msg = (
      "Variable 'parameters' must be a dictionary with keys \n " 
      " 'trend', 'damped_trend', 'seasonal', 'seasonal_periods', 'use_boxcox'"
    )
    raise TypeError(error_msg)

  if (parameters['trend'] not in ['add', 'mul', 'none']):
    raise ValueError("parameters['trend'] must be one of " +
                     "['add', 'mul', 'none']")

  if (parameters['damped_trend'] not in ['true', 'false']):
    raise ValueError("parameters['damped_trend'] must be one of " +
                     "['true', 'false']")

  if (parameters['seasonal'] not in ['add', 'mul', 'none']):
    raise ValueError("parameters['seasonal'] must be one of " +
                     "['add', 'mul', 'none']")

  if (parameters['seasonal_periods'] not in ['2', '3', '4', '6', '12']):
    raise ValueError("parameters['seasonal_periods'] must be one of " +
                     "['2', '3', '4', '6', '12']")
  parameters['seasonal_periods'] = int(parameters['seasonal_periods'])

  if (parameters['use_boxcox'] not in ['true', 'false', 'log']):
    raise ValueError("parameters['use_boxcox'] must be one of " +
                     "['true', 'false', 'log']")


def validateAutoETS(parameters):
  if ((not isinstance(parameters, dict)) or 
      (not all([key in parameters.keys() 
             for key in ['seasonality']]))
  ):
    error_msg = (
      "Variable 'parameters' must be a dictionary with keys \n " 
      " 'seasonality'"
    )
    raise TypeError(error_msg)

  if (parameters['seasonality'] not in ['true', 'false']):
    raise ValueError("parameters['seasonality'] must be one of " +
                     "['true', 'false']")


def validateAutoArima(parameters):
  if ((not isinstance(parameters, dict)) or 
      (not all([key in parameters.keys() 
             for key in ['seasonality']]))
  ):
    error_msg = (
      "Variable 'parameters' must be a dictionary with keys \n " 
      " 'seasonality'"
    )
    raise TypeError(error_msg)

  if (parameters['seasonality'] not in ['true', 'false']):
    raise ValueError("parameters['seasonality'] must be one of " +
                     "['true', 'false']")


def validateProphet(parameters):
  if ((not isinstance(parameters, dict)) or 
      (not all([key in parameters.keys() 
             for key in ['seasonality']]))
  ):
    error_msg = (
      "Variable 'parameters' must be a dictionary with keys \n " 
      " 'seasonality'"
    )
    raise TypeError(error_msg)

  if (parameters['seasonality'] not in ['multiplicative', 'additive', 'none']):
    raise ValueError("parameters['seasonality'] must be one of " +
                     "['multiplicative', 'additive', 'none']")


def validateNN(parameters):
  if ((not isinstance(parameters, dict)) or 
      (not all([key in parameters.keys() 
             for key in ['rnn_type', 'epochs', 'hidden_size', 'learning_rate', 'sequence_length', 'optimiser']]))
  ):
    error_msg = (
      "Variable 'parameters' must be a dictionary with keys \n " 
      " 'rnn_type', 'epochs', 'hidden_size', 'learning_rate', 'sequence_length', 'optimiser'"
    )
    raise TypeError(error_msg)

  if (parameters['rnn_type'] not in ['lstm', 'rnn', 'gru']):
    raise ValueError("parameters['rnn_type'] must be one of " +
                     "['lstm', 'rnn', 'gru']")
  elif (parameters['epochs'] not in ['50', '100', '500', '1000', '2000']):
    raise ValueError("parameters['epochs'] must be one of " +
                     "['50', '100', '500', '1000', '2000']")
  elif (parameters['hidden_size'] not in ['2', '4', '8', '12', '16', '24', '32', '64']):
    raise ValueError("parameters['hidden_size'] must be one of " +
                     "['2', '4', '8', '12', '16', '24', '32', '64']")
  elif (parameters['learning_rate'] not in ['0.001', '0.005', '0.01', '0.05', '0.1']):
    raise ValueError("parameters['learning_rate'] must be one of " +
                     "['0.001', '0.005', '0.01', '0.05', '0.1']")
  elif (parameters['sequence_length'] not in ['2', '4', '8', '12', '16', '24', '32', '64']):
    raise ValueError("parameters['sequence_length'] must be one of " +
                     "['2', '4', '8', '12', '16', '24', '32', '64']")
  elif (parameters['optimiser'] not in ['Adam', 'SGD']):
    raise ValueError("parameters['optimiser'] must be one of " +
                     "['Adam', 'SGD']")

  parameters['epochs'] = int(parameters['epochs'])
  parameters['hidden_size'] = int(parameters['hidden_size'])
  parameters['learning_rate'] = float(parameters['learning_rate'])
  parameters['sequence_length'] = int(parameters['sequence_length'])


def validateBoost(parameters):
  if ((not isinstance(parameters, dict)) or 
      (not all([key in parameters.keys() 
             for key in ['sequence_length', 'num_leaves', 'max_depth', 'learning_rate', 'num_boost_round', 'early_stopping_rounds']]))
  ):
    error_msg = (
      "Variable 'parameters' must be a dictionary with keys \n " 
      " 'sequence_length', 'num_leaves', 'max_depth', 'learning_rate', 'num_boost_round', 'early_stopping_rounds'"
    )
    raise TypeError(error_msg)

  if (parameters['sequence_length'] not in ['2', '4', '8', '12', '16', '24', '32', '64']):
    raise ValueError("parameters['sequence_length'] must be one of " +
                     "['2', '4', '8', '12', '16', '24', '32', '64']")
  elif (parameters['num_leaves'] not in ['2', '4', '8', '12', '16', '24', '32', '64']):
    raise ValueError("parameters['num_leaves'] must be one of " +
                     "['2', '4', '8', '12', '16', '24', '32', '64']")
  elif (parameters['max_depth'] not in ['2', '4', '8', '12', '16', '24', '32', '64']):
    raise ValueError("parameters['max_depth'] must be one of " +
                     "['2', '4', '8', '12', '16', '24', '32', '64']")
  elif (parameters['learning_rate'] not in ['0.001', '0.005', '0.01', '0.05', '0.1']):
    raise ValueError("parameters['learning_rate'] must be one of " +
                     "['0.001', '0.005', '0.01', '0.05', '0.1']")
  elif (parameters['num_boost_round'] not in ['2', '4', '8', '16', '32', '64', '128', '256']):
    raise ValueError("parameters['num_boost_round'] must be one of " +
                     "['2', '4', '8', '16', '32', '64', '128', '256']")
  elif (parameters['early_stopping_rounds'] not in ['0', '2', '4', '8', '16', '32', '64', '128']):
    raise ValueError("parameters['early_stopping_rounds'] must be one of " +
                     "['0', '2', '4', '8', '16', '32', '64', '128']")

  parameters['sequence_length'] = int(parameters['sequence_length'])
  parameters['num_leaves'] = int(parameters['num_leaves'])
  parameters['max_depth'] = int(parameters['max_depth'])
  parameters['learning_rate'] = float(parameters['learning_rate'])
  parameters['num_boost_round'] = int(parameters['num_boost_round'])
  parameters['early_stopping_rounds'] = int(parameters['early_stopping_rounds'])




