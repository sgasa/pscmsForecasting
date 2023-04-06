from flask_restx import Api
from werkzeug.exceptions import HTTPException

from Src import error_handlers 

from .forecasting import ns as forecasting_ns

api = Api(
  title='pscmsForecasting',
  version='1.0',
  description='Prediction SCMS Forecasting Engine',
  prefix="/",
  doc='/doc/'
)

api.error_handlers[HTTPException] = error_handlers.error_handler
api.error_handlers[error_handlers.pSCMSException] = error_handlers.pSCMSException_handler

api.add_namespace(forecasting_ns, path='/')
