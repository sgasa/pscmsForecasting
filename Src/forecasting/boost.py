import numpy as np
import lightgbm as lgb
from . utils import MinMaxScaler, sliding_windows, calculate_metrics
    
class LGBModel():
  def __init__(self, num_leaves, max_depth, learning_rate, num_boost_round,
               early_stopping_rounds):
    self.num_leaves = num_leaves  
    self.max_depth = max_depth 
    self.learning_rate = learning_rate 
    self.num_boost_round = num_boost_round 
    self.early_stopping_rounds = early_stopping_rounds 
    self.model = None
    self.trainX = None
    self.trainY = None
    self.model_fit = None

  def train(self, trainY, trainX):
    training_data = lgb.Dataset(trainX, label=trainY)
    lgParams = {
      'num_leaves': self.num_leaves,
      'max_depth': self.max_depth,
      'learning_rate': self.learning_rate,
      'num_boost_round': self.num_boost_round,
      'early_stopping_rounds': self.early_stopping_rounds,
      'objective': 'regression',
      'metric': 'mape',
      'min_data_in_leaf': 2
    }
    self.model = lgb.train(
      lgParams, 
      training_data,
      valid_sets = [training_data]
    )
    self.trainX = trainX
    self.trainY = trainY
    self.model_fit = self.model.predict(trainX)

  def forecast(self, n_steps): 
    seq_len = self.trainX.shape[1]
    forecast_vector = np.zeros(seq_len + n_steps)
    forecast_vector[:seq_len] = self.trainY[-seq_len:]
    for i in range(n_steps):
      one_step_prediction = self.model.predict([forecast_vector[i:seq_len+i]])
      forecast_vector[seq_len+i] = one_step_prediction
    
    return forecast_vector[seq_len:]

def fit_boost(train_data, pred_steps, parameters):
  sequence_length = int(parameters['sequence_length'])
  num_leaves = int(parameters['num_leaves'])
  max_depth = int(parameters['max_depth'])
  learning_rate = float(parameters['learning_rate'])
  num_boost_round = int(parameters['num_boost_round'])
  early_stopping_rounds = int(parameters['early_stopping_rounds'])


  sc = MinMaxScaler()
  train_data_col = train_data.reshape(-1,)
  train_data_col = sc.fit_transform(train_data_col)
  train_x, train_y = sliding_windows(train_data_col, sequence_length)

  model = LGBModel(num_leaves, max_depth, learning_rate, num_boost_round,
                   early_stopping_rounds)
  model.train(train_y, train_x)

  model_fit = sc.inverse_transform(model.model_fit).flatten()
  metrics = calculate_metrics(
    train_data[sequence_length:], model_fit
  )
  fitted = np.concatenate((np.repeat(np.nan, sequence_length), model_fit))
  forecast = sc.inverse_transform(model.forecast(pred_steps)).astype(float)
  forecast_error = np.repeat(metrics['rmse'], pred_steps)

  return fitted, forecast, forecast_error, metrics

def fit(train_data, pred_steps, parameters):
  train_data_len = len(train_data)
  if train_data_len < 2 * parameters['sequence_length']:
    raise ValueError(
      f"sequence_length must be at least twice the number of training points"
    )
  return fit_boost(train_data, pred_steps, parameters)

