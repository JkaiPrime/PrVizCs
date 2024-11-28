from flask import Flask, render_template, request, redirect, send_file, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import datetime
import logging
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

# Configuração básica do log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Configurações do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Função auxiliar para salvar gráficos otimizados
def save_plot(fig, filename, limite=65536):
    dpi = fig.get_dpi()
    largura, altura = fig.get_size_inches()
    largura_px = largura * dpi
    altura_px = altura * dpi

    if largura_px > limite or altura_px > limite:
        fator = min(limite / largura_px, limite / altura_px)
        largura_ajustada = largura * fator
        altura_ajustada = altura * fator
        logging.info(f"Ajustando gráfico para {int(largura_ajustada * dpi)}x{int(altura_ajustada * dpi)} px")
        fig.set_size_inches(largura_ajustada, altura_ajustada)

    path = os.path.join('static', filename)
    fig.savefig(path, bbox_inches='tight', format='png')
    plt.close(fig)
    return path

def save_additional_plots(data):
    """
    Gera gráficos adicionais para análise, como scatter plots, violin plots e gráficos de linha.
    """
    plots = {}

    # Limitar o número de pontos para gráficos grandes
    max_points = 5000
    if len(data) > max_points:
        logging.info(f"Reduzindo tamanho do DataFrame para {max_points} pontos.")
        data = data.sample(n=max_points, random_state=42)

    try:
        # Gráfico de barras
        if 'rating' in data.columns and 'metacritic' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='rating', y='metacritic', data=data, ax=ax)
            ax.set_title('Média de Metacritic por Rating')
            x_labels = ax.get_xticks()
            ax.set_xticks(x_labels[::4])
            plots['bar_plot'] = save_plot(fig, 'bar_plot.png')
        else:
            logging.warning("Colunas 'rating' ou 'metacritic' ausentes para o gráfico de barras.")

        # Boxplot
        if 'rating_top' in data.columns and 'playtime' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='rating_top', y='playtime', data=data, ax=ax)
            ax.set_title('Distribuição de Playtime por Rating Top')
            plots['box_plot'] = save_plot(fig, 'box_plot.png')
        else:
            logging.warning("Colunas 'rating_top' ou 'playtime' ausentes para o boxplot.")

        # Scatterplot
        if 'achievements_count' in data.columns and 'playtime' in data.columns and 'platforms' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Define limite para o eixo Y (opcionalmente ajustável)
            max_playtime = 10  # Substituir pelo valor máximo desejado
            filtered_data = data[data['playtime'] <= max_playtime]

            sns.scatterplot(x='achievements_count', y='playtime', hue='platforms', data=filtered_data, ax=ax, palette='viridis')
            ax.set_title('Conquistas vs Playtime por Plataforma')
            ax.set_ylim(0, max_playtime)  # Limita o eixo Y no gráfico
            ax.legend().remove()
            plots['scatter_plot'] = save_plot(fig, 'scatter_plot.png')
        else:
            logging.warning("Colunas 'achievements_count', 'playtime' ou 'platforms' ausentes para o scatterplot.")

        # Violinplot
        if 'genres' in data.columns and 'suggestions_count' in data.columns:
            genre_counts = data.groupby('genres')['suggestions_count'].sum().reset_index()

            # Ordenar os gêneros pela soma das sugestões e pegar os X mais frequentes
            top_genres = genre_counts.nlargest(15, 'suggestions_count')['genres']

            # Filtrar os dados para incluir apenas os top 10 gêneros
            filtered_data = data[data['genres'].isin(top_genres)]

            # Criar o gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(x='genres', y='suggestions_count', data=filtered_data, ax=ax)
            ax.set_title('Distribuição de Sugestões por Gênero')
            ax.legend().remove()
            plt.xticks(rotation=90)

            # Salvar o gráfico
            plots['violin_plot'] = save_plot(fig, 'violin_plot.png')
        else:
            logging.warning("Colunas 'genres' ou 'suggestions_count' ausentes para o violin plot.")

        # Linha
        if 'released' in data.columns and 'reviews_count' in data.columns:
            data['released'] = pd.to_datetime(data['released'], errors='coerce')
            data = data.sort_values('released')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x='released', y='reviews_count', data=data, ax=ax)
            ax.set_title('Tendência de Reviews ao Longo do Tempo')
            plt.xticks(rotation=45)
            plots['line_plot'] = save_plot(fig, 'line_plot.png')
        else:
            logging.warning("Colunas 'released' ou 'reviews_count' ausentes para o gráfico de linha.")

    except Exception as e:
        logging.error(f"Erro ao gerar gráficos adicionais: {e}")

    return plots

# Página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Upload de arquivo CSV
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Nenhum arquivo enviado.", 400

    file = request.files['file']
    column = request.form.get('column', None)
    if file.filename == '':
        return "Nome do arquivo inválido.", 400

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('analyze', filename=file.filename, column=column))

    return "Formato de arquivo inválido. Envie um arquivo CSV.", 400

# Análise de dados e geração de gráficos
@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    column = request.args.get('column', None)

    try:
        # Carregar o CSV
        df = pd.read_csv(filepath)

        # Gerar estatísticas descritivas
        stats = df.describe(include='all').to_html(classes='table table-striped', border=0)

        # Seleção de dados numéricos
        numeric_df = df.select_dtypes(include=['number'])

        # Caminhos únicos para os gráficos
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        correlation_plot_path = None
        histogram_plot_path = None

        # Heatmap de correlação
        if not numeric_df.empty:
            correlation_plot_path = f'correlation_{timestamp}.png'
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Mapa de Correlação')
            plt.savefig(os.path.join('static', correlation_plot_path))
            plt.close()

        # Histograma da coluna, se fornecida
        if column and column in df.columns:
            histogram_plot_path = f'histogram_{timestamp}.png'
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True, bins=30, color='skyblue')
            plt.title(f'Distribuição da coluna: {column}')
            plt.xlabel(column)
            plt.ylabel('Frequência')
            plt.savefig(os.path.join('static', histogram_plot_path))
            plt.close()

        # Gráficos adicionais
        additional_plots = save_additional_plots(df)

        # Renderizar a página com os gráficos
        return render_template(
            'analyze.html',
            stats=stats,
            correlation_plot=url_for('static', filename=correlation_plot_path) if correlation_plot_path else None,
            histogram_plot=url_for('static', filename=histogram_plot_path) if histogram_plot_path else None,
            bar_plot=url_for('static', filename="bar_plot.png"),
            box_plot=url_for('static', filename="box_plot.png"),
            scatter_plot=url_for('static', filename="scatter_plot.png"),
            violin_plot=url_for('static', filename="violin_plot.png"),
            line_plot=url_for('static', filename="line_plot.png"),
            filename=filename,
            timestamp=timestamp
        )

    except Exception as e:
        logging.error(f"Erro ao analisar os dados: {e}")
        return f"Erro ao analisar os dados: {e}", 500


# Treinamento de modelo
@app.route('/train/<filename>', methods=['GET', 'POST'])
def train(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        df = pd.read_csv(filepath)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

                # Separar dados numéricos e categóricos
        numeric_columns = X.select_dtypes(include=['number']).columns
        categorical_columns = X.select_dtypes(exclude=['number']).columns

        # Preenchimento de valores ausentes
        numeric_imputer = SimpleImputer(strategy='mean')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # Aplicar imputação
        X_numeric = pd.DataFrame(numeric_imputer.fit_transform(X[numeric_columns]), columns=numeric_columns)
        X_categorical = pd.DataFrame(categorical_imputer.fit_transform(X[categorical_columns]), columns=categorical_columns)

        # Codificar dados categóricos
        for column in X_categorical.columns:
            X_categorical[column] = X_categorical[column].astype('category').cat.codes

        # Combinar dados processados
        X = pd.concat([X_numeric, X_categorical], axis=1)

        # Se o alvo (y) for categórico, codificá-lo
        if y.dtype == 'object':
            y = y.astype('category').cat.codes


        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        if request.method == 'POST':
            model_type = request.form.get('model_type', 'RandomForest')
            logging.info(f"Modelo selecionado: {model_type}")

            if model_type == 'RandomForest':
                n_estimators = int(request.form.get('n_estimators', 100))
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            elif model_type == 'LogisticRegression':
                model = LogisticRegression(max_iter=500)
            elif model_type == 'KNN':
                n_neighbors = int(request.form.get('n_neighbors', 5))
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
            elif model_type == 'SVM':
                kernel = request.form.get('kernel', 'rbf')
                model = SVC(kernel=kernel, random_state=42)
            elif model_type == 'DecisionTree':
                max_depth = request.form.get('max_depth', None)
                model = DecisionTreeClassifier(max_depth=int(max_depth) if max_depth else None, random_state=42)
            else:
                return f"Modelo '{model_type}' não suportado.", 400

            model.fit(X, y)
            accuracy = model.score(X, y)

            model_filename = f"{model_type}_model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
            joblib.dump(model, model_path)

            return render_template(
                'train_result.html',
                accuracy=accuracy,
                model_type=model_type,
                model_path=url_for('download_model', model_filename=model_filename),
                filename=filename
            )

        return render_template('train_form.html', filename=filename)

    except Exception as e:
        logging.error(f"Erro ao treinar o modelo: {e}")
        return f"Erro ao treinar o modelo: {e}", 500

# Upload de novo CSV para treino
@app.route('/upload_new_csv/<filename>', methods=['GET', 'POST'])
def upload_new_csv(filename):
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Nenhum arquivo enviado.", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Nome do arquivo inválido.", 400
        
        if file and file.filename.endswith('.csv'):
            new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(new_filepath)
            return redirect(url_for('train', filename=file.filename))

        return "Formato de arquivo inválido. Envie um arquivo CSV.", 400

    return render_template('upload_new_csv.html', filename=filename)

# Predição com modelo treinado
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'data_file' not in request.files or 'model_file' not in request.files:
            return "Arquivos de dados ou modelo ausentes.", 400

        data_file = request.files['data_file']
        model_file = request.files['model_file']

        if data_file.filename == '' or model_file.filename == '':
            return "Nome do arquivo inválido.", 400

        if not data_file.filename.endswith('.csv') or not model_file.filename.endswith('.pkl'):
            return "Formato de arquivo inválido. Envie um arquivo CSV e um modelo PKL.", 400

        data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.filename)
        model_filepath = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
        data_file.save(data_filepath)
        model_file.save(model_filepath)

        try:
            # Carregar os dados e o modelo
            df = pd.read_csv(data_filepath)
            model = joblib.load(model_filepath)

            # Verificar as colunas esperadas pelo modelo
            expected_columns = model.feature_names_in_
            missing_columns = [col for col in expected_columns if col not in df.columns]
            extra_columns = [col for col in df.columns if col not in expected_columns]

            if missing_columns:
                return f"As seguintes colunas estão ausentes no arquivo enviado: {missing_columns}", 400

            # Reordenar e alinhar colunas
            df = df[expected_columns]

            # Processar dados categóricos, se necessário
            for column in df.select_dtypes(include='object').columns:
                df[column] = df[column].astype('category').cat.codes

            # Fazer predições
            predictions = model.predict(df)
            df['Prediction'] = predictions

            # Salvar resultados
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
            df.to_csv(result_path, index=False)
            return send_file(result_path, as_attachment=True)

        except Exception as e:
            logging.error(f"Erro ao realizar a predição: {e}")
            return f"Erro ao realizar a predição: {e}", 500

    return render_template('predict.html')



# Download de modelo treinado
@app.route('/download_model/<model_filename>', methods=['GET'])
def download_model(model_filename):
    model_filepath = os.path.join(app.config['MODEL_FOLDER'], model_filename)

    if os.path.exists(model_filepath):
        return send_file(model_filepath, as_attachment=True)
    else:
        return f"Modelo '{model_filename}' não encontrado.", 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)
