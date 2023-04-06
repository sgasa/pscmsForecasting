from dotenv import load_dotenv
load_dotenv()

import os
from waitress import serve
from Src import create_app

app = create_app()
port = os.environ.get('PORT')
port = 5000 if port is None else int(port)
serve(app, host='127.0.0.1', port=port)
