from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    CORS(app)

    with app.app_context():
        # Importa as rotas
        from .routes import init_routes
        init_routes(app)

    return app
