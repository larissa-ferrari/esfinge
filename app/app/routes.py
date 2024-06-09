from flask import Blueprint
from .controllers import restore_text


def init_routes(app):
    main_bp = Blueprint('main', __name__)

    # Define a rota para predição
    main_bp.route("/restore", methods=["POST"])(restore_text)

    app.register_blueprint(main_bp)
