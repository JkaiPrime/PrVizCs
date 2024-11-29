# PrVizCs: Aplicação Web para Análise de Dados e Predição com Machine Learning

---

## Authors

- [@Jkai](https://github.com/JkaiPrime)
- [@Renato](https://github.com/Renatoleall)


---


## Descrição do Projeto
Desenvolvimento de uma aplicação web que permite a leitura de um conjunto de dados, possibilitando análise e visualização em um painel interativo (dashboard). A aplicação utilizará técnicas de machine learning para realizar predições sobre os dados apresentados. 

## Sumário
- [Descrição Geral](#descrição-geral)
- [Instalação e Configuração](#instalação-e-configuração)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Funcionalidades](#funcionalidades)
  - [Página Inicial](#página-inicial)
  - [Upload de Arquivo CSV](#upload-de-arquivo-csv)
  - [Análise de Dados](#análise-de-dados)
  - [Treinamento de Modelos](#treinamento-de-modelos)
  - [Upload de Novo CSV para Treinamento](#upload-de-novo-csv-para-treinamento)
  - [Predição com Modelos Treinados](#predição-com-modelos-treinados)
  - [Download de Modelos Treinados](#download-de-modelos-treinados)
- [Estrutura do Código](#estrutura-do-código)
- [Como Usar](#como-usar)
- [Observações Importantes](#observações-importantes)

---

## Descrição Geral
Este aplicativo web em Flask permite que usuários realizem análises de dados, visualizem gráficos, treinem modelos de Machine Learning e façam predições com modelos treinados.

As principais funcionalidades incluem:

1. Upload de arquivos CSV.
2. Geração de análises descritivas e gráficos avançados.
3. Treinamento de modelos de classificação como Random Forest, Logistic Regression, KNN, SVM e Decision Tree.
4. Predição e download de resultados.
---
## Instalação e Configuração
1. Requisitos
Certifique-se de que você possui:

- Python 3.7 ou superior.
- Dependências instaladas.

2. Instalação das Dependências
Instale as bibliotecas necessárias com o comando:
``` bash
pip install flask pandas matplotlib seaborn scikit-learn joblib
```
3. Configuração do Projeto
Crie as pastas necessárias no diretório principal:

``` bash
mkdir uploads models static templates
```

Coloque seus arquivos HTML (como index.html, analyze.html, etc.) na pasta templates/.

---

## Estrutura de Pastas
``` plaintext
|-- app.py                   # Código principal
|-- uploads/                 # Armazena arquivos CSV enviados
|-- models/                  # Armazena modelos treinados
|-- static/                  # Armazena gráficos gerados
|-- templates/               # Arquivos HTML para renderização
|-- app.log                  # Arquivo de log
```

---

## Funcionalidades

### Página Inicial
- **Rota**: `/`
- **Método**: `GET`
- **Descrição**: Exibe a página inicial com links para upload de arquivos e outras funcionalidades.
- **Template**: `index.html`

---

### Upload de Arquivo CSV
- **Rota**: `/upload`
- **Método**: `POST`
- **Descrição**: Faz o upload de um arquivo CSV para o servidor. O arquivo é salvo na pasta `uploads/`.
- **Parâmetros**:
  - `file`: Arquivo CSV enviado.
  - `column` *(opcional)*: Nome de uma coluna para geração de um histograma.
- **Fluxo**: Se o upload for bem-sucedido, o usuário será redirecionado para a página de análise.

---

### Análise de Dados
- **Rota**: `/analyze/<filename>`
- **Método**: `GET`
- **Descrição**: Realiza análise de dados e gera gráficos baseados no arquivo CSV enviado.
- **Saída**:
  - Estatísticas descritivas (exibidas em tabela HTML).
  - Gráficos gerados:
    - Heatmap de correlação.
    - Histograma (se uma coluna for especificada).
    - Gráficos adicionais: barras, boxplot, scatterplot, violinplot e linha.

---

### Treinamento de Modelos
- **Rota**: `/train/<filename>`
- **Métodos**: `GET`, `POST`
- **Descrição**: Treina um modelo de machine learning baseado nos dados enviados.
- **Modelos suportados**:
  - Random Forest
  - Logistic Regression
  - KNN
  - SVM
  - Decision Tree
- **Parâmetros do `POST`**:
  - `model_type`: Tipo de modelo.
  - Parâmetros específicos por modelo:
    - **Random Forest**: `n_estimators` (padrão: 100).
    - **KNN**: `n_neighbors` (padrão: 5).
    - **SVM**: `kernel` (padrão: 'rbf').
    - **Decision Tree**: `max_depth` *(opcional)*.
- **Saída**:
  - Acurácia do modelo.
  - Link para download do modelo treinado.

---

### Upload de Novo CSV para Treinamento
- **Rota**: `/upload_new_csv/<filename>`
- **Método**: `POST`
- **Descrição**: Permite enviar um novo arquivo CSV para treinar novamente um modelo.

---

### Predição com Modelos Treinados
- **Rota**: `/predict`
- **Métodos**: `GET`, `POST`
- **Descrição**: Realiza predições em um novo arquivo CSV usando um modelo previamente treinado.
- **Parâmetros do `POST`**:
  - `data_file`: Arquivo CSV com os dados para predição.
  - `model_file`: Arquivo `.pkl` do modelo treinado.
- **Saída**: Arquivo CSV com as predições.

---

### Download de Modelos Treinados
- **Rota**: `/download_model/<model_filename>`
- **Método**: `GET`
- **Descrição**: Permite o download de um modelo treinado da pasta `models/`.


## Estrutura do Código
### Configurações do Flask:

- Configurações básicas, incluindo pastas UPLOAD_FOLDER e MODEL_FOLDER.
- Criação das pastas necessárias, se não existirem.
### Funções Auxiliares:

- save_plot(fig, filename, limite): Salva gráficos ajustados para não exceder os limites do matplotlib.
- save_additional_plots(data): Gera gráficos como scatterplots, violin plots e gráficos de linha.

### Roteamento:

Diversas rotas implementadas para upload, análise, treinamento, predição e download.
### Treinamento de Modelos:

Implementação modular para diferentes tipos de modelos de machine learning.

### Predições:

- Carregamento do modelo e processamento dos dados para prever os resultados.
---
## Como Usar
### Iniciar o Servidor Execute o comando:

``` python
python app.py
```

### Acessar o Aplicativo Abra o navegador e acesse http://127.0.0.1:5001.

### Realizar Ações

- Faça upload de um arquivo CSV.
- Visualize análises e gráficos.
- Treine modelos de Machine Learning.
- Realize predições com novos dados.
## Observações Importantes
### Formato de Dados:

- O arquivo CSV deve ter valores numéricos na maior parte para análise e treinamento.
- A última coluna é considerada como y (variável alvo) para treinamento.
### Limitações de Gráficos:

- Os gráficos são ajustados automaticamente para não exceder 65.536 pixels.
### Erros:

- Erros no upload, análise ou treinamento são registrados no arquivo app.log.
### Templates HTML:

- Os templates HTML devem ser criados e colocados na pasta templates/.