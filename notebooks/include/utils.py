import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def obter_csv_dados_aleatorios(num_linhas, ausencias_por_coluna):
    """
    Gera dados futuros com valores aleatórios dentro dos limites especificados para cada coluna.

    Parâmetros:
        - num_linhas: número de linhas a serem geradas
        - ausencias_por_coluna: Quantidade máxima de ausência de dados nas colunas (np.nan)

    Retorna:
        - Retorna os dados randomicos e acordo com os parâmetros recebidos e cálculo dos coeficientes
    """

    try:
        # Criar listas para cada coluna
        idades = np.random.randint(18, 65, size=num_linhas)
        generos = np.random.choice(['masculino', 'feminino'], size=num_linhas)
        imcs = np.random.uniform(18, 35, size=num_linhas).astype(float)
        filhos = np.random.randint(0, 4, size=num_linhas)
        fumante = np.random.choice(['sim', 'não'], size=num_linhas)
        regioes = np.random.choice(['sudoeste', 'sudeste', 'nordeste', 'noroeste'], size=num_linhas)

        # Criar DataFrame com os dados gerados
        dados = pd.DataFrame({
            'Idade': idades,
            'Gênero': generos,
            'IMC': imcs,
            'Filhos': filhos,
            'Fumante': fumante,
            'Região': regioes
        })

        # Introduzir valores nulos manualmente em algumas colunas
        for coluna in dados.columns:
            if coluna in ['Idade', 'Filhos']:
                # Convertendo para float e depois para inteiro
                indices_nans = np.random.choice(num_linhas, size=ausencias_por_coluna, replace=False)
                dados.loc[indices_nans, coluna] = np.nan
                dados[coluna] = dados[coluna].astype('Int64')
            else:
                indices_nans = np.random.choice(num_linhas, size=ausencias_por_coluna, replace=False)
                dados.loc[indices_nans, coluna] = np.nan

        # Obtendo encargos por coeficiente
        dados['Encargos'] = obter_encargo_por_coeficientes(dados, num_linhas)

        # Arredondando os valores das colunas para duas casas decimais
        dados['IMC'] = dados['IMC'].apply(lambda x: round(x, 2))

        # Salvar os dados aleatorios junto dos dados originais
        dados.to_csv("../planilhas/1_dados_sinteticos.csv", index=False, encoding='latin1')

        # Retornar os dados concatenados
        return dados

    except Exception as e:
        print("Ocorreu uma exceção:", e)
        return None  # Certificando de que a função retorna algo, mesmo em caso de exceção


def obter_indice_coeficientes(feature):
    """
    Gera o coeficiente que será usado posteriormente no cálculo dos encargos reais

    Parâmetros:
        - feature: Nome da coluna

    Retorna:
        - Retorna o coeficiente de acordo com a feature passada
    """

    if feature == 'Idade':
        coeficiente = 60
    elif feature == 'IMC':
        coeficiente = 30
    elif feature == 'Filhos':
        coeficiente = 400
    elif feature == 'Fumante_sim':
        coeficiente = 500
    else:
        coeficiente = 0 # valor default caso não caia em outra condição

    return coeficiente


def obter_encargo_por_coeficientes(dados, quantidade_maxima_ausencias):
    """
    Cálcula o valor dos encargos de forma aleatória, considerando algumas features com grau de impacto diferenciado

    Parâmetros:
        - dados: dataset do Pandas
        - quantidade_maxima_ausencias: número de linhas a serem geradas

    Retorna:
        - Retorna o valor dos encargos de acordo com o relacionamento entre features x coeficientes
    """

    # Criando uma cópia de dados adicionais apenas para cálculos dos coeficientes
    # Trato os valores ausentes da planilha mais a frente
    # Quero exibir neste momento dados vazios nas colunas randomizadas para esta necessidade
    dados_adicionais_aux = dados.copy()

    # Preencher valores ausentes de 'Idade' e 'Filhos'
    dados_adicionais_aux[['Idade', 'Filhos']] = dados_adicionais_aux[['Idade', 'Filhos']].fillna(0)

    # Preencher valores ausentes de 'Fumante' como não
    dados_adicionais_aux['Fumante'] = dados_adicionais_aux['Fumante'].fillna('não')

    # Preencher valores ausentes de 'IMC' com a média
    media_imc = dados_adicionais_aux['IMC'].mean()
    dados_adicionais_aux['IMC'] = dados_adicionais_aux['IMC'].fillna(media_imc)

    # Obtendo o índice dos coeficientes
    coeficiente_idade = obter_indice_coeficientes('Idade')
    coeficiente_genero_masculino = obter_indice_coeficientes('Gênero_masculino')
    coeficiente_genero_feminino = obter_indice_coeficientes('Gênero_feminino')
    coeficiente_imc = obter_indice_coeficientes('IMC')
    coeficiente_filhos = obter_indice_coeficientes('Filhos')
    coeficiente_fumante_sim = obter_indice_coeficientes('Fumante_sim')
    coeficiente_fumante_nao = obter_indice_coeficientes('Fumante_não')
    coeficiente_regiao_sudoeste = obter_indice_coeficientes('Região_sudoeste')
    coeficiente_regiao_sudeste = obter_indice_coeficientes('Região_sudeste')
    coeficiente_regiao_nordeste = obter_indice_coeficientes('Região_nordeste')
    coeficiente_regiao_noroeste = obter_indice_coeficientes('Região_noroeste')

    # Definindo coeficientes para cada variável independente
    coeficientes = {
        'Idade': coeficiente_idade,  # Idade tem impacto nos encargos
        'Gênero': {'masculino': coeficiente_genero_masculino, 'feminino': coeficiente_genero_feminino},  # Não tem impacto nos encargos
        'IMC': coeficiente_imc,  # tem pouco impacto nos encargos
        'Filhos': coeficiente_filhos,  # Ter filhos aumenta o encargo
        'Fumante': {'sim': coeficiente_fumante_sim, 'não': coeficiente_fumante_nao},  # Ser fumante aumenta o encargo
        'Região': {'sudoeste': coeficiente_regiao_sudoeste, 'sudeste': coeficiente_regiao_sudeste, 'nordeste': coeficiente_regiao_nordeste, 'noroeste': coeficiente_regiao_noroeste}  # Não tem impacto nos encargos
    }

    # Trazendo as colunas dos coeficientes para auxiliar o treinamento do modelo
    #dados['Coef_Idade'] = coeficientes['Idade'] * dados_adicionais_aux['Idade']
    #dados['Coef_IMC'] = round(coeficientes['IMC'] * dados_adicionais_aux['IMC'],2)
    #dados['Coef_Filhos'] = coeficientes['Filhos'] * dados_adicionais_aux['Filhos']
    #dados['Coef_Fumante'] = dados_adicionais_aux['Fumante'].map(coeficientes['Fumante'])

    # Gerando encargos com base nas variáveis independentes
    encargos = (
            coeficientes['Idade'] * dados_adicionais_aux['Idade'] +
            round(coeficientes['IMC'] * dados_adicionais_aux['IMC'],2) +
            coeficientes['Filhos'] * dados_adicionais_aux['Filhos'] +
            dados_adicionais_aux['Fumante'].map(coeficientes['Fumante']) +
            np.random.uniform(100, 1000, size=quantidade_maxima_ausencias)
    )

    # Arredondando os valores das colunas para duas casas decimais
    encargos = encargos.apply(lambda x: round(x, 2))

    return encargos


def limpar_dados(dados):
    """
    Limpeza dos dados para realização do pré processamento dos dados

    Parâmetros:
        - dados: dataset do Pandas

    Retorna:
        - Retorna os dados tratados
    """

    dados_aux = dados.copy()

    # Caso exista, removendo linhas com valores NaN
    # dados_aux = dados_aux.dropna()

    # Substituir valores nulos para o valor esperado (float e int)
    dados_aux['Idade'] = dados_aux['Idade'].fillna(0)
    dados_aux['Filhos'] = dados_aux['Filhos'].fillna(0)
    dados_aux['IMC'] = dados_aux['IMC'].fillna(0)

    #dados_aux['Coef_Idade'] = dados_aux['Coef_Idade'].fillna(0)
    #dados_aux['Coef_Filhos'] = dados_aux['Coef_Filhos'].fillna(0)
    #dados_aux['Coef_IMC'] = dados_aux['Coef_IMC'].fillna(0)
    #dados_aux['Coef_Fumante'] = dados_aux['Coef_Fumante'].fillna(0)

    # Substituir valores nulos de faatures categóricas para o valor esperado (Não informado)
    dados_aux['Gênero'] = dados_aux['Gênero'].fillna('Não informado')
    dados_aux['Fumante'] = dados_aux['Fumante'].fillna('Não informado')
    dados_aux['Região'] = dados_aux['Região'].fillna('Não informado')

    # Convertendo colunas para o tipo esperado
    dados_aux['Idade'] = dados_aux['Idade'].astype(int)
    dados_aux['Filhos'] = dados_aux['Filhos'].astype(int)
    #dados_aux['Coef_Idade'] = dados_aux['Coef_Idade'].astype(int)
    #dados_aux['Coef_Filhos'] = dados_aux['Coef_Filhos'].astype(int)

    return dados_aux


def categorizar_imc(imc):
    """
    Categorizar o índice de Massa Corporal (IMC) de acordo com os valores recebidos no parâmetro

    Parâmetros:
        - imc: Valor IMC capturado do dataset do Pandas

    Retorna:
        - Retorna a categoria do IMC
    """

    if imc is None or np.isnan(imc):
        return 'Não informado'
    if imc < 18.5:
        return 'Abaixo do peso'
    elif 18.5 <= imc < 24.9:
        return 'Peso normal'
    elif 24.9 <= imc < 29.9:
        return 'Sobrepeso'
    else:
        return 'Obeso'


def dados_especificos_coluna(dados, nome_coluna):
    # Criei uma cópia auxiliar para não mudar nada ainda em dados
    dados_aux = dados.copy()

    # Obter o tipo de dados da coluna específica
    tipo_dados_coluna = dados[nome_coluna].dtype

    dados_aux[nome_coluna] = dados_aux[nome_coluna].fillna(0)

    # Converto para int para ficar melhor no print
    if nome_coluna == "Filhos" or nome_coluna == "Idade":
        dados_aux[nome_coluna] = dados_aux[nome_coluna].astype(int)

    dados_filtrados = dados_aux[dados_aux[nome_coluna] > 0]

    menor_valor = dados_filtrados[nome_coluna].min()
    maior_valor = dados_filtrados[nome_coluna].max()
    valor_mais_frequente = dados_filtrados[nome_coluna].mode()[0]
    contador = dados_filtrados[dados_filtrados[nome_coluna] == valor_mais_frequente].shape[0]

    print(f"\nNa coluna {nome_coluna} ({tipo_dados_coluna}) a faixa dos dados está entre: {menor_valor} até {maior_valor}.")
    print(f"O valor mais frequente na coluna {nome_coluna} é: {valor_mais_frequente}, que aparece {int(contador)} vezes.")


def prever_encargos_futuros(best_model, dados):
    # Utilize o modelo treinado para fazer previsões dos encargos futuros
    previsoes = best_model.predict(dados)
    custos_previstos = list(map(lambda x: round(x, 2), previsoes))

    return custos_previstos


def obter_grupo_de_risco(best_model, dados):
    # Identifique grupos de indivíduos com diferentes níveis de risco
    encargos_futuros = prever_encargos_futuros(best_model, dados)
    # Aqui, vamos assumir que os valores acima da média são alto risco,
    # os valores abaixo da média são baixo risco e o resto é médio risco
    media = np.mean(encargos_futuros)

    # Classifique os clientes em grupos de risco
    grupos_risco = []
    for custo in encargos_futuros:
        if custo > media:
            grupos_risco.append("Alto Risco")
        elif custo < media:
            grupos_risco.append("Baixo Risco")
        else:
            grupos_risco.append("Médio Risco")

    return grupos_risco


def expectativa_plano_saude(dados):
    expectativa_plano_saude = []

    for index, row in dados.iterrows():
        if row['Encargos Futuro'] > row['Encargos Reais']:
            expectativa_plano_saude.append("Plano de saúde será mais caro")
        elif row['Encargos Futuro'] < row['Encargos Reais']:
            expectativa_plano_saude.append("Plano de saúde mais barato")
        else:
            expectativa_plano_saude.append("Plano de saúde com mesmo valor atual")
    
    return expectativa_plano_saude


def planejamento_estrategico(best_model, dados, encargos_futuros):
    # Segmentação de risco
    grupos_risco = obter_grupo_de_risco(best_model, dados)

    # Média dos encargos futuros
    media_encargos = np.mean(encargos_futuros)

    # Definição de limites com base na média
    limite_baixo = media_encargos * 0.8  # 20% abaixo da média
    limite_alto = media_encargos * 1.2  # 20% acima da média

    # Lista para armazenar sugestões estratégicas
    sugestoes_estrategicas = []

    # Fornece sugestões estratégicas com base no grupo de risco e custo
    for i, risco in enumerate(grupos_risco):
        custo = encargos_futuros[i]

        if risco == "Alto Risco":
            if custo < limite_baixo:
                sugestao = "Aumentar monitoramento e tratamentos preventivos para evitar complicações"
            elif custo > limite_alto:
                sugestao = "Reduzir custos otimizando tratamentos e buscando alternativas mais eficientes"
            else:
                sugestao = "Continuar com o plano de tratamento atual, com monitoramento regular"

        elif risco == "Médio Risco":
            if custo < limite_baixo:
                sugestao = "Manter consultas regulares e promover educação em saúde"
            elif custo > limite_alto:
                sugestao = "Reavaliar o plano de tratamento para evitar custos excessivos"
            else:
                sugestao = "Continuar acompanhamento periódico e ajuste conforme necessário"

        elif risco == "Baixo Risco":
            if custo < limite_baixo:
                sugestao = "Incentivar hábitos saudáveis e prevenção"
            elif custo > limite_alto:
                sugestao = "Monitorar a saúde para garantir que o risco permaneça baixo"
            else:
                sugestao = "Manter acompanhamento regular e monitorar indicadores de saúde"

        # Adiciona a sugestão à lista
        sugestoes_estrategicas.append(sugestao)

    return sugestoes_estrategicas

def verificar_se_modelo_tem_dados_nan_inf(model, x, y):
    """
    Verifica se há valores NaN ou Inf em um modelo após o treinamento.

    Args:
        model: Modelo de regressão do scikit-learn.
        x: Matriz de features.
        y: Vetor de targets.
    """

    # isinf = é uma função que verifica se um ou mais elementos de um array são infinitos(infinito positivo ou negativo)

    model.fit(x, y)

    # Verificação de NaN e Inf nos coeficientes (se aplicável)
    if hasattr(model, "coef_"):
        if np.any(np.isnan(model.coef_)):
            raise ValueError("NaN encontrado nos coeficientes do modelo")
        if np.any(np.isinf(model.coef_)):
            raise ValueError("Inf encontrado nos coeficientes do modelo")

    # Verificação de NaN e Inf nas previsões
    predictions = model.predict(x)
    if np.any(np.isnan(predictions)):
        raise ValueError("NaN encontrado nas previsões do modelo")
    if np.any(np.isinf(predictions)):
        raise ValueError("Inf encontrado nas previsões do modelo")


def gerar_dados_futuros_com_limites(novas_linhas, x_test, idade_minima=18):
    """
    Gera dados futuros com valores aleatórios dentro dos limites especificados para cada coluna.

    Parâmetros:
        - novas_linhas: número de linhas a serem geradas
        - x_test: DataFrame contendo os dados de teste
        - idade_minima: Limite inferior para a coluna 'Idade' (padrão é 18 anos)

    Retorna:
        - DataFrame contendo os dados futuros gerados
    """
    dados_futuros = pd.DataFrame()

    for coluna in x_test.columns:
        if x_test[coluna].dtype == 'int64' or x_test[coluna].dtype == 'int32':
            if coluna == 'Idade':
                # Garante que a idade esteja dentro dos limites desejados
                minimo = max(idade_minima, int(x_test[coluna].min()))
            else:
                minimo = int(x_test[coluna].min())

            maximo = int(x_test[coluna].max())
            valores = np.random.randint(minimo, maximo + 1, size=novas_linhas).astype(x_test[coluna].dtype)
        else:
            valores = np.round(np.random.uniform(x_test[coluna].min(), x_test[coluna].max(), size=novas_linhas), 2)

        dados_futuros[coluna] = valores

    return dados_futuros