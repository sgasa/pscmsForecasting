from flask import request
from flask_restx import Namespace, Resource, fields

from Src import forecasting 
from Src.error_handlers import pSCMSException

ns = Namespace('Forecasting')

parameters_dto = fields.Wildcard(fields.String)
metrics_dto = ns.model('metrics_dto', {
  'mae': fields.Float,
  'mase': fields.Float,
  'rmse': fields.Float,
})
input_dto = ns.model('input_dto', {
  'train_data': fields.List(fields.Float),
  'pred_steps': fields.Integer,
  'model': fields.String, 
  'parameters': fields.Wildcard(fields.String)
})
output_dto = ns.model('output_dto', {
  'fitted': fields.List(fields.Float),
  'forecast': fields.List(fields.Float),
  'forecast_error': fields.List(fields.Float),
  'metrics': fields.Nested(metrics_dto)
})


@ns.route('')
class ForecastingRoute(Resource):
  @ns.doc(
    "Forecasting",
    description="Get forecasts"
  )
  @ns.expect(input_dto, validate=True)
  @ns.marshal_with(output_dto)
  def post(self):
    req_json = request.get_json()

    try:
      fitted, forecast, forecast_error, metrics = forecasting.forecast(
        req_json['train_data'], req_json['pred_steps'], req_json['model'],
         req_json['parameters']
      )
    except Exception as e:
      raise pSCMSException(e.args[0])

    return {
      'fitted': fitted,
      'forecast': forecast,
      'forecast_error': forecast_error,
      'metrics': metrics
    }

