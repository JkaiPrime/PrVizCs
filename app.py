from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    df = pd.read_csv(filepath)

    # correlation_plot_path = os.path.join('static', 'correlation.png')
    # df.corr().style.background_gradient(cmap='coolwarm').to_excel("correlation.xlsx")
    # df[['price', 'rating']].plot(kind='scatter', x='price', y='rating', title='Preços Avaliaçoco Média')


if __name__ == '__main__':
    app.run(debug=True)