from flask import Flask, render_template, request, redirect, send_file, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        # Carregar dados
        df = pd.read_csv(filepath)

        # Estatísticas descritivas
        stats = df.describe().to_html(classes='table table-striped', border=0)

        # Criar gráfico de correlação
        correlation_plot_path = os.path.join('static', f'correlation_{filename}.png')
        plt.figure(figsize=(10, 8))
        pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(15, 10))
        plt.savefig(correlation_plot_path)
        plt.close()

        return render_template('analyze.html', stats=stats, correlation_plot=f'/{correlation_plot_path}')
    except Exception as e:
        return f"Erro ao analisar os dados: {e}", 500



from flask import request

@app.route('/train/<filename>', methods=['GET', 'POST'])
def train(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Separar características (X) e alvo (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Pré-processamento: lidar com colunas categóricas
    for column in X.select_dtypes(include='object').columns:
        X[column] = X[column].astype('category').cat.codes  # Converte para números
    
    if y.dtype == 'object':  # Converter y se for categórico
        y = y.astype('category').cat.codes

    if request.method == 'POST':
        model_type = request.form.get('model_type', 'RandomForest')
        n_estimators = int(request.form.get('n_estimators', 100))

        # Escolher o modelo
        if model_type == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        elif model_type == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'SVM':
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', random_state=42)
        else:
            return f"Modelo '{model_type}' não suportado.", 400

        # Treinar o modelo
        model.fit(X, y)

        # Avaliar o modelo
        accuracy = model.score(X, y)
        model_path = f"/download_model/{model_type}_model.pkl"

        return render_template('train_result.html', accuracy=accuracy, model_type=model_type)

    # Renderizar o formulário para escolha do modelo
    return render_template('train_form.html', filename=filename)



@app.route('/download_model/<filename>')
def download_model(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        return send_file(filepath, as_attachment=True)
    except FileNotFoundError:
        return f"Erro: O arquivo {filename} não foi encontrado.", 404



if __name__ == '__main__':
    app.run(debug=True)