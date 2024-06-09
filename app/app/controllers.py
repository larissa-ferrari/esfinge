from flask import request, jsonify
from .services import restore_ancient_text


def restore_text():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Texto não fornecido'}), 400

    if '_' not in text:
        return jsonify({'error': 'Texto não contém a máscara "___"'})

    region, restored_text = restore_ancient_text(text)

    return jsonify({
        "region": region,
        "restored_text": restored_text
    })
