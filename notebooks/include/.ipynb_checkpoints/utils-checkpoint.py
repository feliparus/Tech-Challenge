import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def incrementar_dados_aleatorios_csv(dados):
    try:
        # Gerar 200 linhas de dados
        num_linhas = 200

        # Criar listas para cada coluna
        idades = np.random.randint(18, 80, size=num_linhas).astype(float)
        generos = np.random.choice(['masculino', 'feminino'], size=num_linhas)
        imcs = np.random.uniform(18, 35, size=num_linhas).astype(float)
        filhos = np.random.randint(0, 5, size=num_linhas).astype(float)
        fumante = np.random.choice(['sim', 'não'], size=num_linhas)
        regioes = np.random.choice(['sudoeste', 'sudeste', 'nordeste', 'noroeste'], size=num_linhas)
        encargos = np.random.uniform(10000, 50000, size=num_linhas).astype(float)

        # Introduzir valores nulos manualmente em algumas colunas
        # Vamos forçar 20 ausências em cada coluna
        ausencias_por_coluna = 20

        indices_nans = np.random.choice(num_linhas, size=ausencias_por_coluna, replace=False)

        idades[indices_nans] = np.nan
        generos[indices_nans] = np.nan
        imcs[indices_nans] = np.nan
        filhos[indices_nans] = np.nan
        fumante[indices_nans] = np.nan
        regioes[indices_nans] = np.nan
        encargos[indices_nans] = np.nan

        # Criar DataFrame com os dados adicionais
        dados_adicionais = pd.DataFrame({
            'Idade': idades,
            'Gênero': generos,
            'IMC': imcs,
            'Filhos': filhos,
            'Fumante': fumante,
            'Região': regioes,
            'Encargos': encargos
        })

        # Concatenar os DataFrames 'dados' e 'dados_adicionais'
        dados = pd.concat([dados, dados_adicionais], ignore_index=True)

        # Arredondando os valores das colunas para duas casas decimais
        dados['IMC'] = dados['IMC'].apply(lambda x: round(x, 2))
        dados['Encargos'] = dados['Encargos'].apply(lambda x: round(x, 2))

        # Retornar os dados concatenados
        return dados
    except Exception as e:
        print("Ocorreu uma exceção:", e)


def categorizar_imc(imc):
    if imc < 18.5:
        return 'Abaixo do Peso'
    elif imc < 25:
        return 'Peso Normal'
    elif imc < 30:
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

    """
    if nome_coluna == "Fumante":
        dados_aux[nome_coluna] = dados_aux[nome_coluna].map({'sim': 1, 'não': 0})

    # Filtrar os dados para excluir valores iguais a zero (desconsidero ele nestas contagens - fiz a contagem dos valores ausentes no sumário)
    if tipo_dados_coluna == "float64" :
        dados_filtrados = dados_aux[dados_aux[nome_coluna] > 0]
    else:
        dados_filtrados = dados_aux.copy()
    """   

    dados_filtrados = dados_aux[dados_aux[nome_coluna] > 0]
        
    menor_valor = dados_filtrados[nome_coluna].min()
    maior_valor = dados_filtrados[nome_coluna].max()
    valor_mais_frequente = dados_filtrados[nome_coluna].mode()[0]
    contador = dados_filtrados[dados_filtrados[nome_coluna] == valor_mais_frequente].shape[0]
    
    print(f"\nNa coluna {nome_coluna} ({tipo_dados_coluna}) a faixa dos dados está entre: {menor_valor} até {maior_valor}.")
    print(f"O valor mais frequente na coluna {nome_coluna} é: {valor_mais_frequente}, que aparece {int(contador)} vezes.")
    

def montar_graficos(dados):
    # Contar o número de ocorrências de algumas colunas
    distribuicao_genero = dados['Gênero'].value_counts().sort_index()
    distribuicao_imc = dados['Categoria IMC'].value_counts().sort_index()
    distribuicao_filhos = dados['Filhos'].value_counts().sort_index()
    
    # Criar uma figura e uma grade de subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    
    # Ajustar o espaçamento entre os subplots
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.8)
    
    montar_grafico_barra_vertical(distribuicao_genero, axs[0, 0], 'Distribuição de Gênero', 'Número de Pessoas', 'Gênero')
    
    montar_grafico_barra_horizontal(distribuicao_imc, axs[0, 1], 'Distribuição de IMC', 'Número de Pessoas', 'Categoria')
    
    montar_grafico_histograma_idade(dados, axs[1, 0], 'Distribuição de Idade com Linha de Tendência', 'Idade', 'Densidade')
    
    montar_grafico_barra_horizontal(distribuicao_filhos, axs[1, 1], 'Distribuição do Número de Filhos', 'Número de Pessoas', 'Qtd. de Filhos')
    
    # Mostra os gráficos
    plt.show()


def montar_grafico_barra_vertical(dados, axs, titulo, eixo_x, eixo_y):
    dados.plot(kind='bar', title=titulo, ax=axs)

    # Rotacionar os rótulos do eixo x em 45 graus
    axs.set_xticklabels(dados.index, rotation=90, ha='left')
    
    axs.set_title(titulo, fontweight='bold')
    axs.set_xlabel(eixo_x, fontweight='bold')
    axs.set_ylabel(eixo_y, fontweight='bold')

    # Adicionar rótulos (indicadores) às barras
    for i, valor in enumerate(dados):
        axs.text(i, valor, str(valor), ha='center', va='bottom')


def montar_grafico_barra_horizontal(dados, axs, titulo, eixo_x, eixo_y):
    dados.plot(kind='barh', title=titulo, ax=axs)
    axs.set_title(titulo, fontweight='bold')
    axs.set_xlabel(eixo_x, fontweight='bold')
    axs.set_ylabel(eixo_y, fontweight='bold')

    # Adicionar rótulos (indicadores) às barras
    for i, valor in enumerate(dados):
        axs.text(valor, i, str(valor), ha='left', va='center')


def montar_grafico_pizza(dados, axs, titulo):
    dados.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=axs)

    # Removendo os rótulos na pizza (labels=None)
    # Adicionando uma legenda embaixo do gráfico
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
    
    # Removendo o rótulo do eixo y
    axs.set_ylabel('')
    
    axs.set_title(titulo, fontweight='bold')
    axs.set_ylabel('')  # Remover o rótulo do eixo y


def montar_grafico_correlacao(dados1, dados2, axs, titulo, eixo_x, eixo_y):
    axs.scatter(dados1, dados2, alpha=0.5, color='green')
    axs.set_title(titulo, fontweight='bold')
    axs.set_xlabel(eixo_x, fontweight='bold')
    axs.set_ylabel(eixo_y, fontweight='bold')

    # Adiciona a linha de tendência linear
    coeficiente_correlacao = np.corrcoef(dados1, dados2)[0, 1]
    linha_tendencia_x = np.array([dados1.min(), dados1.max()])
    linha_tendencia_y = coeficiente_correlacao * linha_tendencia_x + dados2.mean()
    axs.plot(linha_tendencia_x, linha_tendencia_y, color='red', linestyle='--', label='Linha de Tendência')

    # Exibe a legenda
    axs.legend()


# Este gráfico foi feito fixo para idade
def montar_grafico_histograma_idade(dados, axs, titulo, eixo_x, eixo_y):
    axs.hist(dados['Idade'], bins=10, color='skyblue', edgecolor='black', density=True, rwidth=0.1) # - tive que espaçar as barras para melhor visualização
    idade_mean = dados['Idade'].mean()
    idade_std = dados['Idade'].std()
    xmin, xmax = axs.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal_fit = (1 / (np.sqrt(2 * np.pi) * idade_std)) * np.exp(-(x - idade_mean) ** 2 / (2 * idade_std ** 2))

    # Adicionando rótulos aos valores nas barras
    for rect in axs.patches:
        height = rect.get_height()
        axs.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    
    axs.plot(x, normal_fit, 'r--', label='Linha de Tendência')
    axs.set_title(titulo, fontweight='bold')
    axs.set_xlabel(eixo_x, fontweight='bold')
    axs.set_ylabel(eixo_y, fontweight='bold')

