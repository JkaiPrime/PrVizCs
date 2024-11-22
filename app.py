from flask import Flask, render_template, request, redirect, send_file, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Nenhum arquivo enviado.", 400

    file = request.files['file']
    if file.filename == '':
        return "Nome do arquivo inválido.", 400

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('analyze', filename=file.filename))

    return "Formato de arquivo inválido. Envie um arquivo CSV.", 400

@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = pd.read_csv(filepath)
        stats = df.describe(include='all').to_html(classes='table table-striped', border=0)

        # Pré-processar dados
        # Excluir colunas não numéricas ou converter para categórico se necessário
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return "Erro: Não há colunas numéricas para análise de correlação.", 400

        # Criar heatmap de correlação
        correlation_plot_path = os.path.join('static', f'correlation_{filename}.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.savefig(correlation_plot_path)
        plt.close()

        return render_template('analyze.html', stats=stats, correlation_plot=f'/{correlation_plot_path}')
    except Exception as e:
        return f"Erro ao analisar os dados: {e}", 500


@app.route('/train/<filename>', methods=['GET', 'POST'])
def train(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Separar características (X) e alvo (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Pré-processamento: lidar com colunas categóricas
    for column in X.select_dtypes(include='object').columns:
        X[column] = X[column].astype('category').cat.codes
    
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    if request.method == 'POST':
        model_type = request.form.get('model_type', 'RandomForest')
        n_estimators = int(request.form.get('n_estimators', 100))

        if model_type == 'RandomForest':
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            return f"Modelo '{model_type}' não suportado.", 400

        model.fit(X, y)
        accuracy = model.score(X, y)

        model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type}_model.pkl")
        joblib.dump(model, model_path)

        return render_template('train_result.html', accuracy=accuracy, model_type=model_type, model_path=model_path)

    return render_template('train_form.html', filename=filename)

@app.route('/download_model/<path:filename>')
def download_model(filename):
    filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
    try:
        return send_file(filepath, as_attachment=True)
    except FileNotFoundError:
        return f"Erro: O arquivo {filename} não foi encontrado.", 404

@app.route('/retrain/<filename>', methods=['POST'])
def retrain(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "Arquivo de dados não encontrado.", 404

    return redirect(url_for('train', filename=filename))

if __name__ == '__main__':
    app.run(debug=True)
