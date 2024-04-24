**Tech Challenge FIAP**

Seguem abaixo as descrições da estrutura do projeto e seus principais arquivos:

**Pasta notebooks**

Importante: A ordem de execução é rodar primeiro o EDA e depois o Modelling.

A pasta notebooks contém os notebooks Jupyter para explorar e modelar os dados.

EDA.ipynb: Notebook de Análise Exploratória de Dados (EDA). Usamos este notebook para entender e analisar os dados de entrada.
Modelling.ipynb: Notebook de modelagem de dados. Aqui, experimentamos diferentes modelos de machine learning para prever resultados ou extrair insights valiosos dos dados.

**Pasta includes**

Nesta pasta contém arquivos Python com funções personalizadas para auxiliar nas tarefas de análise de dados e visualização.
- utils.py: Este arquivo contém funções customizadas para manipulação de dados, como:

    
    obter_csv_dados_aleatorios: Função para gerar um DataFrame com dados aleatórios a partir de um arquivo CSV.

    gerar_dados_futuros_com_limites: Função para simular dados futuros com base em limites definidos.

    Outras funções indispensáveis como limpar dados, etc

- graficos.py: Este arquivo disponibiliza funções para facilitar a criação de gráficos com alta legibilidade, incluindo:

    Gráficos horizontais e verticais.

    Gráficos de correlação.

    Outras visualizações úteis para análise de dados.

**Pasta planilhas**

Nesta pasta estão contidas as planilhas geradas durante a execução dos arquivos notebooks.

- 1_dados_sinteticos.csv: Esta planilha é gerada com dados embaralhados, contendo dados nan em colunas distintas e também de forma aleatória
- 2_dados_processados_treino.csv: Esta planilha é gerada na etapa "Separação dos Dados em Treino e Teste". Nesta planilha estão contidos o conjunto dos treinos
- 3_dados_processados_teste.csv: Esta planilha é gerada na etapa "Separação dos Dados em Treino e Teste". Nesta planilha estão contidos o conjunto dos testes
- 4_dados_processados_treino_target.csv: Esta planilha é gerada na etapa "Separação dos Dados em Treino e Teste". Nesta planilha estão contidos os alvos dos treinos
- 5_dados_processados_teste_target.csv: Esta planilha é gerada na etapa "Separação dos Dados em Treino e Teste". Nesta planilha estão contidos os alvos dos testes
- 6_dados_com_outliers.csv: Esta planilha demostra os outliers para cada coluna
- 7_dados_futuros.csv: Esta planilha é gerada para realizar previsões futuras, após escolha do melhor modelo