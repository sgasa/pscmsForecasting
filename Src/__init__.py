from flask import Flask

def create_app():

    # Create and configure the app
    app = Flask(__name__)
    app.config.from_mapping(
      MYSECRET = 'Code',
      SECRET_KEY='dev',
      ERROR_INCLUDE_MESSAGE=False
    )

    # Allow trailing slashes on routes
    app.url_map.strict_slashes = False

    # CORS
    from flask_cors import CORS
    CORS(app, resources={r"*": {"origins": "*"}})

    # Initialise routes
    from .routes import api
    api.init_app(app)

    return app

    
