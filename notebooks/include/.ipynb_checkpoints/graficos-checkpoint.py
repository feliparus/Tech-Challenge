import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap


def montar_graficos_visualizacao_inicial(dados):
    """
    Monta e exibe gráficos para visualização inicial dos dados, incluindo distribuições de gênero, fumante, número de filhos, região e IMC.
    
    Parâmetros:
        - dados: DataFrame contendo os dados a serem visualizados.
    
    Retorna:
        - Exibe os gráficos na saída padrão.
    """
    
    dados_aux = dados.copy()

    # Mudando exibição dos dados vazios para aparecer no gráfico
    dados_aux['Gênero'] = dados_aux['Gênero'].fillna('Não informado')
    dados_aux['Fumante'] = dados_aux['Fumante'].fillna('Não informado')
    dados_aux['Região'] = dados_aux['Região'].fillna('Não informado')

    # Contar o número de ocorrências de algumas colunas
    distribuicao_genero = dados_aux['Gênero'].value_counts().sort_index()
    distribuicao_imc = dados_aux['Categoria_IMC'].value_counts().sort_index()
    distribuicao_filhos = dados_aux['Filhos'].value_counts().sort_index()
    distribuicao_regiao = dados_aux['Região'].value_counts().sort_index()
    distribuicao_fumante = dados_aux['Fumante'].value_counts().sort_index()

    # Criar uma figura e uma grade de subplots
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))

    # Ajustar o espaçamento entre os subplots
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.5)
    
    montar_grafico_barra_vertical(distribuicao_genero, axs[0, 0], 'Distribuição de Gênero', 'Número de Pessoas',
                                  'Gênero')

    montar_grafico_barra_vertical(distribuicao_fumante, axs[0, 1], 'Distribuição Por Fumante', 'Número de Pessoas',
                                  'Fumante')

    montar_grafico_barra_horizontal(distribuicao_filhos, axs[1, 0], 'Distribuição do Número de Filhos',
                                    'Número de Pessoas', 'Qtd. de Filhos')

    montar_grafico_barra_horizontal(distribuicao_regiao, axs[1, 1], 'Distribuição Por Região', 'Número de Pessoas',
                                    'Região')

    montar_grafico_barra_horizontal(distribuicao_imc, axs[2, 0], 'Distribuição de IMC', 'Número de Pessoas',
                                    'Categoria')

    # Remove o gráfico no axs[2, 1]
    axs[2, 1].remove()

    # Mostra os gráficos
    plt.show()


def montar_grafico_barra_vertical(dados, axs, titulo, eixo_x, eixo_y, medida_x=None, medida_y=None):
    """
    Monta e exibe um gráfico de barras verticais.
    
    Parâmetros:
        - dados: Series contendo os dados a serem plotados.
        - axs: Eixo no qual o gráfico será desenhado.
        - titulo: Título do gráfico.
        - eixo_x: Rótulo do eixo x.
        - eixo_y: Rótulo do eixo y.
        - medida_x: Unidade de medida opcional para o eixo x.
        - medida_y: Unidade de medida opcional para o eixo y.
    
    Retorna:
        - Exibe o gráfico de barras na saída padrão.
    """
    
    # Plotar o gráfico de barras
    dados.plot(kind='bar', title=titulo, ax=axs)

    # Rotacionar os rótulos do eixo x em 45 graus
    axs.set_xticklabels(dados.index, rotation=45, ha='right')

    # Adicionar rótulos às barras
    for i, valor in enumerate(dados):
        if medida_y:
            axs.text(i, valor, f'{valor} {medida_y}', ha='center', va='bottom', fontsize=12)
        else:
            axs.text(i, valor, str(valor), ha='center', va='bottom', fontsize=12)

    # Definir o título
    axs.set_title(titulo, fontsize=12, fontweight='bold')

    # Definir os rótulos dos eixos x e y
    axs.set_xlabel(eixo_x, fontsize=12)
    axs.set_ylabel(eixo_y, fontsize=12)


def montar_grafico_barra_horizontal(dados, axs, titulo, eixo_x, eixo_y):
    """
    Monta um gráfico de barras horizontais para os dados fornecidos.

    Parâmetros:
        - dados: Dados para o gráfico de barras horizontais.
        - axs: Eixo em que o gráfico será plotado.
        - titulo: Título do gráfico.
        - eixo_x: Rótulo do eixo x.
        - eixo_y: Rótulo do eixo y.
    """

    # Definir a largura máxima dos rótulos das barras
    largura_maxima_rotulos = 20  # Ajuste conforme necessário

    # Quebrar os rótulos das barras
    if isinstance(dados.index, pd.Index):
        rotulos_ajustados = [textwrap.fill(str(label), largura_maxima_rotulos) for label in dados.index]
        dados.index = rotulos_ajustados

    # Plotar o gráfico de barras horizontais
    dados.plot(kind='barh', title=titulo, ax=axs)

    # Configurar título e rótulos dos eixos
    axs.set_title(titulo, fontsize=12, fontweight='bold')
    axs.set_xlabel(eixo_x, fontsize=12)
    axs.set_ylabel(eixo_y, fontsize=12)

    # Adicionar rótulos às barras
    for i, valor in enumerate(dados):
        axs.text(valor, i, str(valor), ha='left', va='center', fontsize=12)


def montar_grafico_pizza(dados, axs, titulo):
    """
    Monta um gráfico de pizza para os valores de grupos de risco.

    Parâmetros:
        - dados: dataset do Pandas
        - axs: Eixo em que o gráfico de pizza será plotado.
        - titulo: Título do gráfico de pizza.
    """
    
    # Plotar gráfico de pizza
    dados.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=axs, colors=['red', 'orange', 'green'])

    # Adicionar uma legenda embaixo do gráfico
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=1)

    # Remover rótulo do eixo y
    axs.set_ylabel('')

    # Configurar título
    axs.set_title(titulo, fontsize=12, fontweight='bold')


def montar_graficos_relacionamento_encargos(dados):
    """
    Monta e exibe gráficos para analisar o relacionamento entre os encargos médicos e variáveis como idade e número de filhos.
    
    Parâmetros:
        - dados: DataFrame contendo os dados para análise.
    
    Retorna:
        - Exibe os gráficos na saída padrão.
    """
    
    # Criar uma figura e uma grade de subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    # Ajustar o espaçamento entre os subplots
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.2)

    montar_grafico_correlacao(dados['Idade'], dados['Encargos'], axs[0],
                              'Correlação entre Idade e Encargos Médicos', 'Idade', 'Encargos Médicos')

    montar_grafico_linha_com_media(dados, axs[1], 'Filhos', 'Encargos', 'Encargos Médicos por Filho',
                                   'Número de Filhos', 'Encargos Médicos')

    # Mostra os gráficos
    plt.show()


def montar_grafico_correlacao(dados1, dados2, axs, titulo, eixo_x, eixo_y):
    """
    Monta e exibe um gráfico de dispersão para visualizar a correlação entre duas variáveis.
    
    Parâmetros:
        - dados1: Valores da primeira variável.
        - dados2: Valores da segunda variável.
        - axs: Eixo no qual o gráfico será desenhado.
        - titulo: Título do gráfico.
        - eixo_x: Rótulo do eixo x.
        - eixo_y: Rótulo do eixo y.
    
    Retorna:
        - Exibe o gráfico de dispersão na saída padrão.
    """
    
    axs.scatter(dados1, dados2, alpha=1, color='green')
    axs.set_title(titulo, fontsize=12, fontweight='bold')
    axs.set_xlabel(eixo_x, fontsize=12)
    axs.set_ylabel(eixo_y, fontsize=12)

    # Definindo os limites e intervalos do eixo x
    # plt.xticks(range(0, int(dados1.max()) + 1, 1))


def montar_grafico_linha_com_media(dados, axs, campo1, campo2, titulo, eixo_x, eixo_y):
    """
    Monta e exibe um gráfico de linha com a média de uma variável agrupada por outra variável.
    
    Parâmetros:
        - dados: DataFrame contendo os dados a serem plotados.
        - axs: Eixo no qual o gráfico será desenhado.
        - campo1: Variável para agrupamento.
        - campo2: Variável para calcular a média.
        - titulo: Título do gráfico.
        - eixo_x: Rótulo do eixo x.
        - eixo_y: Rótulo do eixo y.
    
    Retorna:
        - Exibe o gráfico de linha na saída padrão.
    """
    
    # Calcular a média dos encargos por filho e arredondar para 2 casas decimais
    media = round(dados.groupby(campo1)[campo2].mean(), 2)

    # Criar o gráfico de linha
    media.plot(marker='o', linestyle='-', label='Encargos Médicos por Filho')

    # Adicionar rótulos aos eixos e título
    axs.set_title(titulo, fontsize=12, fontweight='bold')
    axs.set_xlabel(eixo_x, fontsize=12)
    axs.set_ylabel(eixo_y, fontsize=12)

    # Adicionar os valores numéricos aos pontos de dados com ajuste de posição horizontal
    for x, y in zip(media.index, media.values):
        if y > media.mean():  # Se o valor for maior que a média, posicione acima do ponto
            va = 'bottom'
            xytext = (0, 5)
        else:  # Caso contrário, posicione abaixo do ponto
            va = 'top'
            xytext = (0, -5)

        axs.annotate(f'{y}', xy=(x, y), xytext=xytext, textcoords='offset points', fontsize=12, ha='center', va=va)

    # Definir os marcadores de posição do eixo x para pular a cada 1 unidade
    plt.xticks(range(int(media.index.min()), int(media.index.max()) + 1, 1))

    # Adicionar legenda
    plt.legend(loc='upper center')


# Este gráfico foi feito fixo para idade
def montar_grafico_histograma_idade(dados, titulo, eixo_x, eixo_y):
    """
    Monta e exibe um histograma para visualização da distribuição da idade nos dados.
    
    Parâmetros:
        - dados: DataFrame contendo os dados para análise.
        - titulo: Título do gráfico.
        - eixo_x: Rótulo do eixo x.
        - eixo_y: Rótulo do eixo y.
    
    Retorna:
        - Exibe o histograma na saída padrão.
    """
    
    # Calcula o intervalo para os bins começando do valor mínimo de idade
    min_idade = int(np.floor(dados['Idade'].min()))
    max_idade = int(np.ceil(dados['Idade'].max()))
    intervalo = 5
    bins = range(min_idade, max_idade + intervalo, intervalo)

    # Criar uma figura e uma grade de subplots
    fig, ax = plt.subplots(figsize=(10, 5))

    # Criar o gráfico de linha da distribuição das idades
    counts, bins, patches = ax.hist(dados['Idade'], bins=bins, edgecolor='black', rwidth=0.8)

    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel(eixo_x)
    ax.set_ylabel(eixo_y)

    # Adicionar rótulos às barras
    for count, bin, patch in zip(counts, bins, patches):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        plt.text(x, y, f'{count:.0f}', ha='center', va='bottom', fontsize='12')

    # Mostra os gráficos
    plt.show()


def montar_graficos_dados_futuros(dados_futuros):
    """
    Monta e exibe gráficos para análise dos dados futuros, incluindo comparação entre encargos reais e futuros, distribuição por expectativa de plano de saúde, planos estratégicos e grupos de risco.
    
    Parâmetros:
        - dados_futuros: DataFrame contendo os dados futuros a serem analisados.
    
    Retorna:
        - Exibe os gráficos na saída padrão.
    """
    
    # Criar subplots com um layout de 2x2
    fig, axs = plt.subplots(4, 1, figsize=(15, 30))

    # Ajustar o espaçamento entre os subplots
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.7, wspace=0.3, hspace=1.0)

    # Contar o número de ocorrências de algumas colunas
    distribuicao_grupos_risco = dados_futuros['Grupos Risco'].value_counts().sort_index()
    distribuicao_expectativa_plano_saude = dados_futuros['Expectativa Plano de Saúde'].value_counts().sort_index()
    distribuicao_planos_estrategicos = dados_futuros['Planos estratégicos'].value_counts().sort_index()

    # Gráfico Comparação entre Encargos Reais e Encargos Futuro
    indices = range(len(dados_futuros))
    axs[0].scatter(indices, dados_futuros['Encargos Reais'], color='blue', label='Encargos Reais', marker='o')
    axs[0].scatter(indices, dados_futuros['Encargos Futuro'], color='red', label='Encargos Futuro', marker='o')

    # Adicionando linhas de conexão entre pares correspondentes
    for i in indices:
        axs[0].plot([i, i], [dados_futuros['Encargos Reais'][i], dados_futuros['Encargos Futuro'][i]], color='gray',
                       linestyle='-', linewidth=0.5)

    # Configurações para o gráfico de comparação entre encargos reais e futuros
    axs[0].set_xlabel('Índice', fontsize=12)
    axs[0].set_ylabel('Valor', fontsize=12)
    axs[0].set_title('Comparação entre Encargos Reais e Encargos Futuro')
    axs[0].legend()

    montar_grafico_barra_horizontal(distribuicao_expectativa_plano_saude, axs[1], 'Distribuição por Expectativa Plano de Saúde',
                                    'Expectativa', 'Qtd. de Associados')

    # Gráfico de Barras para Distribuição de Risco
    montar_grafico_barra_horizontal(distribuicao_planos_estrategicos, axs[2], 'Distribuição por Planos Estratégicos', 'Planos estratégicos',
                                  'Quantidade')

    # Gráfico de Barras para Distribuição de Risco
    montar_grafico_barra_vertical(distribuicao_grupos_risco, axs[3], 'Distribuição por Grupos de Risco',
                                  'Grupos de Risco',
                                  'Quantidade')

    # Configurar layout
    plt.tight_layout()
    plt.show()