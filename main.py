from flask import Flask, request, jsonify, render_template, send_file
import os
from werkzeug.utils import secure_filename
import pandas as pd
from utils import allowed_file, mapping_process

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/secure/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'xlsx','csv'}

# -----------------------------------------------
# Rotas do sistema
# -----------------------------------------------

# Rota da Página "Home"
@app.route('/')
def home():
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template('content_home.html')
    return render_template('main.html', page_template='content_home.html')

# Rota da Página "Manual"
@app.route('/manual')
def manual():
    return render_template('content_manual.html')

# Rota da Página "Contatos"
@app.route('/contacts')
def contacts():
    return render_template('content_contacts.html')

# Rota da Página "The Tool"
@app.route('/tool', methods=['GET','POST'])
def tool():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Salva o arquivo enviado
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Diretório para salvar os resultados
                output_dir = app.config['UPLOAD_FOLDER']

                # Processa o dataset e gera os mapas e arquivo de predição
                resultados = mapping_process(filepath, output_dir)

                # Ler os mapas gerados
                with open(resultados['input_map'], 'r', encoding='utf-8') as file:
                    input_map_html = file.read()
                with open(resultados['output_map'], 'r', encoding='utf-8') as file:
                    output_map_html = file.read()

                # Extrair as tags <head> e <body>
                def extract_html_parts(html):
                    head = html.split('<head>')[1].split('</head>')[0]
                    body = html.split('<body>')[1].split('</body>')[0]
                    scripts = [script for script in html.split('<script') if '</script>' in script]
                    return head, body, scripts
                
                input_map_head, input_map_body, input_map_scripts = extract_html_parts(input_map_html)
                output_map_head, output_map_body, output_map_scripts = extract_html_parts(output_map_html)

                # Retorna os arquivos como links de download
                return jsonify({
                    "status": "success",
                    "input_map_url": f"/download?file={resultados['input_map']}",
                    "output_map_url": f"/download?file={resultados['output_map']}",
                    "prediction_url": f"/download?file={resultados['prediction']}"
                })
            
            except ValueError as ve:
                # Erros relacionados à validação ou ao processamento
                return jsonify({"status": "error", "message": str(ve)}), 400
            
            except Exception as e:
                # Outros erros inesperados
                return jsonify({"status": "error", "message": str(e)}), 500
            
        return jsonify({"status": "error", "message": "Arquivo inválido."}), 400
    
    return render_template('content_mapping.html')

# Rota para verificar os arquivos
@app.route('/download')
def download_file():
    file_path = request.args.get('file')
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Arquivo não encontrado."}), 404
    
    # Verificar se o arquivo é um HTML de mapa
    if file_path.endswith('.html'):
        return send_file(file_path)
    
    return send_file(file_path, as_attachment=True)

# -----------------------------------------------
# Rodando a API
# -----------------------------------------------

if __name__ == '__main__':
    # Certifique-se de que o diretório de uploads existe
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # app.run(ssl_context='adhoc')  # Ativando o HTTPS em ambiente local
    app.run() # Rodando sem o HTTPS localmente