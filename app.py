from flask import Flask, render_template, request, redirect, send_file, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer


# Configuração básica do log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_log.log"),
        logging.StreamHandler()
    ]
)

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


        # Heatmap de correlação
        numeric_df = df.select_dtypes(include=['number'])
        correlation_plot_path = None
        if not numeric_df.empty:
            correlation_plot_path = os.path.join('static', f'correlation_{filename}.png')
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.savefig(correlation_plot_path)
            plt.close()

        return render_template('analyze.html', stats=stats, correlation_plot=f'/{correlation_plot_path}' if correlation_plot_path else None, filename=filename)
    except Exception as e:
        return f"Erro ao analisar os dados: {e}", 500



@app.route('/train/<filename>', methods=['GET', 'POST'])
def train(filename):
    logging.info(f"Iniciando o treinamento com o arquivo: {filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        df = pd.read_csv(filepath)
        logging.info("Arquivo CSV carregado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao carregar o arquivo CSV: {e}")
        return f"Erro ao carregar o arquivo CSV: {e}", 500

    # Separar características (X) e alvo (y)
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        logging.info("Separação de características (X) e alvo (y) concluída.")
    except Exception as e:
        logging.error(f"Erro ao separar X e y: {e}")
        return f"Erro ao processar os dados: {e}", 500

    # Pré-processamento: lidar com colunas categóricas
    try:
        # Pré-processamento: lidar com colunas categóricas
        for column in X.select_dtypes(include='object').columns:
            X[column] = X[column].astype('category').cat.codes

        # Lidando com valores ausentes
        imputer = SimpleImputer(strategy='mean')  # Ou 'median' ou 'most_frequent'
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        if y.dtype == 'object':
            y = y.astype('category').cat.codes
        logging.info("Pré-processamento de colunas categóricas concluído.")
    except Exception as e:
        logging.error(f"Erro durante o pré-processamento: {e}")
        return f"Erro durante o pré-processamento: {e}", 500

    if request.method == 'POST':
        model_type = request.form.get('model_type', 'RandomForest')
        logging.info(f"Modelo selecionado: {model_type}")

        try:
            # Selecionar e configurar o modelo
            if model_type == 'RandomForest':
                n_estimators = int(request.form.get('n_estimators', 100))
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                logging.info(f"Configurando Random Forest com {n_estimators} estimadores.")
            
            elif model_type == 'KNN':
                n_neighbors = int(request.form.get('n_neighbors', 5))
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                logging.info(f"Configurando K-Nearest Neighbors com {n_neighbors} vizinhos.")
            
            elif model_type == 'LogisticRegression':
                model = LogisticRegression(max_iter=500)
                logging.info("Configurando Regressão Logística.")
            
            elif model_type == 'SVM':
                kernel = request.form.get('kernel', 'rbf')
                model = SVC(kernel=kernel, random_state=42)
                logging.info(f"Configurando SVM com kernel: {kernel}.")
            
            elif model_type == 'DecisionTree':
                max_depth = int(request.form.get('max_depth', None)) if request.form.get('max_depth') else None
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                logging.info(f"Configurando Árvore de Decisão com profundidade máxima: {max_depth}.")
            
            else:
                logging.warning(f"Modelo '{model_type}' não suportado.")
                return f"Modelo '{model_type}' não suportado.", 400

            # Treinamento do modelo
            model.fit(X, y)
            accuracy = model.score(X, y)
            logging.info(f"Treinamento concluído. Acurácia: {accuracy:.2f}")

            # Salvar o modelo treinado
            model_filename = f"{model_type}_model.pkl"
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
            joblib.dump(model, model_path)
            logging.info(f"Modelo salvo em: {model_path}")

            # Renderizar resultados
            return render_template(
                'train_result.html',
                accuracy=accuracy,
                model_type=model_type,
                model_path=url_for('download_model', model_type=model_type, filename=filename),  # Adição do filename
                filename=filename  # Adicionando filename ao contexto
            )


        except Exception as e:
            logging.error(f"Erro durante o treinamento do modelo '{model_type}': {e}")
            return f"Erro durante o treinamento: {e}", 500

    return render_template('train_form.html', filename=filename)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Verificar se os arquivos foram enviados
        if 'data_file' not in request.files or 'model_file' not in request.files:
            return "Arquivos de dados ou modelo ausentes.", 400

        data_file = request.files['data_file']
        model_file = request.files['model_file']

        # Validar os nomes dos arquivos
        if data_file.filename == '' or model_file.filename == '':
            return "Nome do arquivo inválido.", 400

        # Verificar formatos de arquivos
        if not data_file.filename.endswith('.csv') or not model_file.filename.endswith('.pkl'):
            return "Formato de arquivo inválido. Envie um arquivo CSV para dados e um arquivo PKL para o modelo.", 400

        # Salvar os arquivos no servidor
        data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.filename)
        model_filepath = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
        data_file.save(data_filepath)
        model_file.save(model_filepath)

        try:
            # Carregar o arquivo de dados
            df = pd.read_csv(data_filepath)

            # Carregar o modelo
            model = joblib.load(model_filepath)

            # Pré-processamento dos dados (igual ao usado no treinamento)
            for column in df.select_dtypes(include='object').columns:
                df[column] = df[column].astype('category').cat.codes

            # Fazer predições
            predictions = model.predict(df)

            # Adicionar predições ao DataFrame
            df['Prediction'] = predictions

            # Salvar os resultados em um arquivo CSV
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
            df.to_csv(result_path, index=False)

            return send_file(result_path, as_attachment=True)

        except Exception as e:
            return f"Erro ao realizar a predição: {e}", 500

    return render_template('predict.html')



@app.route('/download_model/<model_type>/<filename>', methods=['GET'])
def download_model(model_type, filename):
    # Construir o caminho completo para o arquivo do modelo
    model_filename = f"{model_type}_model.pkl"
    model_filepath = os.path.join(app.config['MODEL_FOLDER'], model_filename)

    # Verificar se o modelo existe antes de tentar baixá-lo
    if os.path.exists(model_filepath):
        return send_file(model_filepath, as_attachment=True)
    else:
        return f"Erro: O modelo '{model_type}' não foi encontrado. Certifique-se de que ele foi treinado primeiro.", 404



@app.route('/upload_new_csv/<filename>', methods=['GET', 'POST'])
def upload_new_csv(filename):
    if request.method == 'POST':
        # Verificar se um novo arquivo foi enviado
        if 'file' not in request.files:
            return "Nenhum arquivo enviado.", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Nome do arquivo inválido.", 400
        
        if file and file.filename.endswith('.csv'):
            new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(new_filepath)

            # Redirecionar para o treinamento com o novo arquivo
            return redirect(url_for('train', filename=file.filename))

        return "Formato de arquivo inválido. Envie um arquivo CSV.", 400

    return render_template('upload_new_csv.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
