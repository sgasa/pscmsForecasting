"""
input_size = Variable
hidden_size = Variable
seq_size = Variable
batch_first=True
output_size=1
num_layers=1

For the RNNs:
  input: batch_size, seq_size, input_size
  h,c: num_layers, batch_size, hidden_size
  output: batch_size, seq_size, hidden_size

For the fc:
  input: *, hidden_size
  output: *, output_size


"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from . utils import MinMaxScaler, sliding_windows, calculate_metrics

class RNNModel(nn.Module):

  def __init__(self, input_size, hidden_size, type='lstm'):
      super(RNNModel, self).__init__()
      
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.batch_first = True
      self.output_size = 1
      self.num_layers = 1
      self.type = type
      
      if self.type == 'lstm':
        self.rnn = nn.LSTM(
          input_size=self.input_size,
          hidden_size=self.hidden_size,
          batch_first=self.batch_first
        )
      elif self.type == 'rnn':
        self.rnn = nn.RNN(
          input_size=self.input_size,
          hidden_size=self.hidden_size,
          batch_first=self.batch_first
        )
      elif self.type == 'gru':
        self.rnn = nn.GRU(
          input_size=self.input_size,
          hidden_size=self.hidden_size,
          batch_first=self.batch_first
        )
      else:
        raise ValueError("Parameter type must be one of 'lstm', 'rnn', or 'gru'")

      
      self.fc = nn.Linear(
        in_features=self.hidden_size,
        out_features=self.output_size
      )

  def forward(self, x):
    batch_size = x.size(0)
    if self.type == 'lstm':
      h_0 = (
        torch.zeros(self.num_layers, batch_size, self.hidden_size),
        torch.zeros(self.num_layers, batch_size, self.hidden_size)
      )
    else:
      h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    _, h_out = self.rnn(x, h_0)
    
    if self.type == 'lstm':
      h_out = h_out[0]

    h_out = h_out.view(-1, self.hidden_size)
    out = self.fc(h_out)
    
    return out

class ModelWrapper():
  def __init__(self, epochs=50, device=torch.device('cpu')):
    self.net = []
    self.optimiser = []
    self.criterion = nn.MSELoss()
    self.device = device
    self.epochs = epochs
    self.rmse = None
    self.model_fit = None
    self.trainX = None
    self.trainY = None

  def load_net(self, net):
    self.net = net.to(self.device)

  def load_criterion(self, criterion):
    self.criterion = criterion()

  def load_optimiser(self, optimiser, **params):
    self.optimiser = optimiser(self.net.parameters(), **params)

  def train(self, trainY, trainX):
    
    loss_array = []
    for _ in range(self.epochs):
  
      self.optimiser.zero_grad()
      output = self.net(trainX)
      loss = self.criterion(output, trainY)
      loss.backward()
      self.optimiser.step()

      loss_array.append(loss.item())

    with torch.no_grad():
      self.model_fit = self.net(trainX).detach().numpy()
      self.rmse = np.sqrt(np.mean((self.model_fit - trainY.detach().numpy())**2))

    self.trainX = trainX
    self.trainY = trainY

    return output, loss_array

  def forecast(self, n_steps):
    seq_len = self.trainX.size(1)
    forecast_tensor = torch.zeros(1, seq_len + n_steps, self.trainX.size(2))
    forecast_tensor[:, :seq_len, :] = self.trainY[-seq_len:].detach().clone()
    with torch.no_grad():
      for i in range(n_steps):
        one_step_prediction = self.net(forecast_tensor[:, i:seq_len+i, :])
        forecast_tensor[:, seq_len+i, :] = one_step_prediction

    return forecast_tensor[:, seq_len:, :].squeeze().numpy()
    
def fit_nnet(train_data, pred_steps, parameters):
  epochs = int(parameters['epochs'])
  hidden_size = int(parameters['hidden_size'])
  learning_rate = float(parameters['learning_rate'])
  sequence_length = int(parameters['sequence_length'])
  rnn_type = parameters['rnn_type']
  optimiser_map = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
  }

  sc = MinMaxScaler()
  train_data_col = train_data.reshape(-1,1)
  train_data_col = sc.fit_transform(train_data_col)
  train_x, train_y = sliding_windows(train_data_col, sequence_length)
  train_x = torch.Tensor(train_x)
  train_y = torch.Tensor(train_y)

  model = ModelWrapper(epochs=epochs)
  model.load_net(RNNModel(
    input_size=1, hidden_size=hidden_size, type=rnn_type
  ))
  model.load_optimiser(optimiser_map[parameters['optimiser']], lr=learning_rate)
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
  return fit_nnet(train_data, pred_steps, parameters)