from flask import jsonify

def error_handler(response):
  return {'error':response.description}, response.code

def pSCMSException_handler(err):
  return {'error': err.message}, 200

class pSCMSException(Exception):
  def __init__(self, message):
    self.message = message

  def response(self):
    return jsonify({'error': self.message})