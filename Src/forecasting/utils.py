import numpy as np

class MinMaxScaler():
  """
  MinMaxScaler()
 
  Scales data between 0 and 1 

  Methods
  -------

  fit_transform : 
      Scales data between 0 and 1 and stores the scaling parameters for 
      use with other data sets, fit_transform method
  transform : 
      Scales data using the stored scaling parameters
  inverse_transform :
      Inverses the transformation using the stored scaling parameters

  """
  def __init__(self):
    self.x_min = None
    self.x_max = None

  def fit_transform(self, x):
    self.x_min = np.min(x.flatten())
    self.x_max = np.max(x.flatten())
    return (x - self.x_min) / (self.x_max - self.x_min)

  def transform(self, x):
    if self.x_min is None or self.x_max is None:
      raise ValueError('MinMaxScaler: fit_transform the scaler prior to using it')
    return (x - self.x_min) / (self.x_max - self.x_min)

  def inverse_transform(self, x):
    if self.x_min is None or self.x_max is None:
      raise ValueError('MinMaxScaler: fit_transform the scaler prior to using it')
    return x * (self.x_max - self.x_min) + self.x_min

def sliding_windows(data, seq_length):
  """
  sliding_windows(data, seq_length)

  Function for converting an array of data to a matrix of consecutive slices
  of length seq_length and a vector with the next value in the data array 
  for each slice. Useful for converting time series to arrays for supervised learning

  Parameters
  ----------
  data : array_like type 
    The data
  seq_length : integer
    The length of each sequence

  Returns
  -------
  training_array : np.array (n x seq_length)
      The array with the consecutive sequences
  targets : np.array (n x 1)
      The next data point for each of the sequences in array

  """

  if seq_length >= len(data):
    raise ValueError('sliding_windows: seq_length must be smaller than the length of data')
  if seq_length <= 0:
    raise ValueError('sliding_windows: seq_length must be a positive integer')
  x = []
  y = []

  for i in range(len(data)-seq_length):
      _x = data[i:(i+seq_length)]
      _y = data[i+seq_length]
      x.append(_x)
      y.append(_y)

  return np.array(x), np.array(y)

def calculate_metrics(data, predictions):

  valid_data = np.logical_and(~np.isnan(data), ~np.isnan(predictions))
  data = data[valid_data]
  predictions = predictions[valid_data]

  if ((len(data) == 0) or (len(predictions) == 0)):
    return {'mae': 0, 'mase': 0, 'rmse': 0}

  mae = np.mean(np.abs(data - predictions))

  if len(data) > 1:
    mase = mae / np.mean(np.abs(data[1:] - data[:-1]))
  else:
    mase = mae / data[0]

  rmse = np.sqrt(np.mean((data - predictions)**2))

  return {'mae': mae, 'mase': mase, 'rmse': rmse}