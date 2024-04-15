import matplotlib.pyplot as plt
import numpy as np


def montar_graficos_visualizacao_inicial(dados):
    dados_aux = dados.copy()
    dados_aux['Gênero'] = dados_aux['Gênero'].fillna('Não informado')

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
    
    montar_grafico_barra_vertical(distribuicao_genero, axs[0, 0], 'Distribuição de Gênero', 'Número de Pessoas', 'Gênero')

    montar_grafico_barra_vertical(distribuicao_fumante, axs[0, 1], 'Distribuição Por Fumante', 'Número de Pessoas',
                                  'Fumante')

    montar_grafico_barra_horizontal(distribuicao_filhos, axs[1, 0], 'Distribuição do Número de Filhos', 'Número de Pessoas', 'Qtd. de Filhos')

    montar_grafico_barra_horizontal(distribuicao_regiao, axs[1, 1], 'Distribuição Por Região', 'Número de Pessoas', 'Região')

    montar_grafico_barra_horizontal(distribuicao_imc, axs[2, 0], 'Distribuição de IMC', 'Número de Pessoas',
                                    'Categoria')

    # Remove o gráfico no axs[2, 1]
    axs[2, 1].remove()

    # Mostra os gráficos
    plt.show()


def montar_grafico_barra_vertical(dados, axs, titulo, eixo_x, eixo_y, medida_x=None, medida_y=None):
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
    dados.plot(kind='barh', title=titulo, ax=axs)
    axs.set_title(titulo, fontsize=12, fontweight='bold')
    axs.set_xlabel(eixo_x, fontsize=12)
    axs.set_ylabel(eixo_y, fontsize=12)

    # Adicionar rótulos (indicadores) às barras
    for i, valor in enumerate(dados):
        axs.text(valor, i, str(valor), ha='left', va='center', fontsize=12)


def montar_grafico_pizza(dados, axs, titulo):
    dados.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=axs)

    # Adicionando uma legenda embaixo do gráfico
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
    
    # Removendo o rótulo do eixo y
    axs.set_ylabel('')
    
    axs.set_title(titulo, fontsize=12, fontweight='bold')
    axs.set_ylabel('')  # Remover o rótulo do eixo y


def montar_graficos_relacionamento_encargos(dados):
    # Criar uma figura e uma grade de subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    # Ajustar o espaçamento entre os subplots
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.2)

    montar_grafico_correlacao(dados['Idade'], dados['Encargos'], axs[0],
                              'Correlação entre Idade e Encargos Médicos', 'Idade', 'Encargos Médicos')

    montar_grafico_linha_com_media(dados, axs[1], 'Filhos', 'Encargos', 'Encargos Médios por Filho', 'Número de Filhos',
                                   'Encargos Médios')

    # Mostra os gráficos
    plt.show()


def montar_grafico_correlacao(dados1, dados2, axs, titulo, eixo_x, eixo_y):
    axs.scatter(dados1, dados2, alpha=1, color='green')
    axs.set_title(titulo, fontsize=12, fontweight='bold')
    axs.set_xlabel(eixo_x, fontsize=12)
    axs.set_ylabel(eixo_y, fontsize=12)

    # Definindo os limites e intervalos do eixo x
    # plt.xticks(range(0, int(dados1.max()) + 1, 1))


def montar_grafico_linha_com_media(dados, axs, campo1, campo2, titulo, eixo_x, eixo_y):
    # Calcular a média dos encargos por filho e arredondar para 2 casas decimais
    media = round(dados.groupby(campo1)[campo2].mean(), 2)

    # Criar o gráfico de linha
    media.plot(marker='o', linestyle='-', label='Encargos Médios por Filho')

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
