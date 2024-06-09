### Estrutura de Diretórios

```plaintext
.
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── controllers.py
│   └── services.py
├── config.py
├── run.py
├── requirements.txt
```

### 1. `app/__init__.py`

Aqui inicializamos a aplicação Flask e configuramos as rotas.

```python
from flask import Flask

def create_app():
    app = Flask(__name__)

    with app.app_context():
        # Importa as rotas
        from .routes import init_routes
        init_routes(app)

    return app
```

### 2. `app/routes.py`

Aqui definimos as rotas da nossa aplicação.

```python
from flask import Blueprint
from .controllers import predict_missing_words

def init_routes(app):
    main_bp = Blueprint('main', __name__)

    # Define a rota para predição
    main_bp.route('/predict', methods=['POST'])(predict_missing_words)

    app.register_blueprint(main_bp)
```

### 3. `app/controllers.py`

Aqui colocamos a lógica do controlador, que lida com a lógica de entrada e saída da API.

```python
from flask import request, jsonify
from .services import get_predictions

def predict_missing_words():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Texto não fornecido'}), 400

    if '___' not in text:
        return jsonify({'error': 'Texto não contém a máscara "___"'}), 400

    predicted_words = get_predictions(text)

    return jsonify({'predictions': predicted_words})
```

### 4. `app/services.py`

Aqui implementamos a lógica de negócio, ou seja, a interação com o modelo de deep learning.

```python
from transformers import pipeline

# Carrega o modelo pré-treinado de preenchimento de máscara
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

def get_predictions(text):
    masked_text = text.replace('___', fill_mask.tokenizer.mask_token)
    predictions = fill_mask(masked_text)
    predicted_words = [pred['token_str'] for pred in predictions]
    return predicted_words
```

### 5. `config.py`

Se precisar de configurações adicionais para sua aplicação, você pode colocá-las aqui.

```python
# Configurações da aplicação (se necessário)
```

### 6. `run.py`

Este é o ponto de entrada da aplicação.

```python
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
```

### 7. `requirements.txt`

Liste as dependências do projeto.

```
flask
transformers
torch
```

### Como funciona a estrutura

1. **`app/__init__.py`**: Inicializa a aplicação Flask e registra as rotas.
2. **`app/routes.py`**: Define as rotas da aplicação e associa as rotas aos controladores correspondentes.
3. **`app/controllers.py`**: Contém a lógica do controlador que processa a entrada do usuário, valida os dados e chama os serviços.
4. **`app/services.py`**: Contém a lógica de negócio, incluindo a interação com o modelo de deep learning.
5. **`run.py`**: Inicia a aplicação Flask.

### Execução da Aplicação

Para executar a aplicação, utilize o comando:

```bash
python run.py
```

### Teste da API

Você pode testar a API com o mesmo comando `curl` mencionado anteriormente:

```bash
curl -X POST http://127.0.0.1:5000/restore -H "Content-Type: application/json" -d '{"text": "The quick brown ___ jumps over the lazy dog."}'
```

Essa estrutura modular torna o código mais organizado, facilitando a manutenção e a escalabilidade da aplicação.
