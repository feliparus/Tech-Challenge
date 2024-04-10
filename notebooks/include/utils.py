import pandas as pd
import numpy as np


def incrementar_dados_aleatorios_csv(dados):
    try:
        # Gerar 2000 linhas de dados
        num_linhas = 2000

        # Criar listas para cada coluna
        idades = np.random.randint(18, 80, size=num_linhas).astype(float)
        generos = np.random.choice(['masculino', 'feminino'], size=num_linhas)
        imcs = np.random.uniform(18, 35, size=num_linhas).astype(float)
        filhos = np.random.randint(0, 4, size=num_linhas).astype(float)
        fumante = np.random.choice(['sim', 'não'], size=num_linhas)
        regioes = np.random.choice(['sudoeste', 'sudeste', 'nordeste', 'noroeste'], size=num_linhas)

        # Introduzir valores nulos manualmente em algumas colunas
        # Vamos forçar 200 ausências em cada coluna
        ausencias_por_coluna = 200

        indices_nans = np.random.choice(num_linhas, size=ausencias_por_coluna, replace=False)

        idades[indices_nans] = np.nan
        generos[indices_nans] = np.nan
        imcs[indices_nans] = np.nan
        filhos[indices_nans] = np.nan
        fumante[indices_nans] = np.nan
        regioes[indices_nans] = np.nan

        # Criar DataFrame com os dados adicionais
        dados_adicionais = pd.DataFrame({
            'Idade': idades,
            'Gênero': generos,
            'IMC': imcs,
            'Filhos': filhos,
            'Fumante': fumante,
            'Região': regioes
        })

        # Definir coeficientes para cada variável independente
        coeficientes = {
            'Idade': 500, # Maior peso para a idade
            'Gênero': {'masculino': 0, 'feminino': 0}, # Não tem impacto nos encargos
            'IMC': 100,
            'Filhos': 300, # Ter filhos aumenta o encargo
            'Fumante': {'sim': 200, 'não': 0}, # Ser fumante aumenta o encargo
            'Região': {'sudoeste': 0, 'sudeste': 0, 'nordeste': 0, 'noroeste': 0} # Não tem impacto nos encargos
        }

        # Gerar encargos com base nas variáveis independentes
        dados_adicionais['Encargos'] = (
            coeficientes['Idade'] * dados_adicionais['Idade'] +
            coeficientes['Filhos'] * dados_adicionais['Filhos'] +
            dados_adicionais['Fumante'].map(coeficientes['Fumante']) +
            np.random.uniform(1000, 10000, size=num_linhas)  # Ajuste a amplitude para 50000
        )

        # Concatenar os DataFrames 'dados' e 'dados_adicionais'
        dados = pd.concat([dados, dados_adicionais], ignore_index=True)

        # Arredondando os valores das colunas para duas casas decimais
        dados['IMC'] = dados['IMC'].apply(lambda x: round(x, 2))
        dados['Encargos'] = dados['Encargos'].apply(lambda x: round(x, 2))

        # Salvar os dados aleatorios junto dos dados originais
        dados.to_csv("../planilhas/2_dados_aleatorios_sobre_original.csv", index=False, encoding='latin1')

        # Retornar os dados concatenados
        return dados
    except Exception as e:
        print("Ocorreu uma exceção:", e)


def categorizar_imc(imc):
    if imc is None or np.isnan(imc):
        return 'Valor Inválido'  # Ou qualquer outra categoria que você queira definir para valores inválidos
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
    custos_previstos = list(map(lambda x: round(x,2), previsoes))

    return custos_previstos


def segmentacao_de_risco(best_model, dados):
    # Identifique grupos de indivíduos com diferentes níveis de risco
    previsoes = best_model.predict(dados)
    # Aqui, vamos assumir que os valores acima da média são alto risco,
    # os valores abaixo da média são baixo risco e o resto é médio risco
    media = np.mean(previsoes)

    # Classifique os clientes em grupos de risco
    grupos_risco = []
    for custo in previsoes:
        if custo > media:
            grupos_risco.append("Alto Risco")
        elif custo < media:
            grupos_risco.append("Baixo Risco")
        else:
            grupos_risco.append("Médio Risco")

    return grupos_risco


def analise_de_sensibilidade(best_model, dados, variavel_alterada, novo_valor):
    # Copiar os dados dos clientes para fazer as alterações
    dados_alterados = dados.copy()

    # Alterar o valor da variável específica nos dados alterados
    dados_alterados[variavel_alterada] = novo_valor

    # Fazer previsões com o modelo usando os dados alterados
    custos_previstos_alterados = best_model.predict(dados_alterados)
    custos_previstos_alterados = list(map(lambda x: round(x,2), custos_previstos_alterados))

    return custos_previstos_alterados


def otimizacao_de_recursos(custos_previstos):
    # Suponha que a otimização de recursos envolva alocar mais recursos para grupos de alto risco
    recursos_otimizados = []
    for custo in custos_previstos:
        if custo > 18000:  # Exemplo de um limite arbitrário para custo alto
            recursos_otimizados.append("Alocar mais recursos")
        else:
            recursos_otimizados.append("Manter recursos")
    
    return recursos_otimizados


def planejamento_estrategico(best_model, dados):
    # Utilize as informações obtidas com o modelo para desenvolver planos estratégicos
    # Supondo que isso envolva análise dos grupos de risco e previsões de custos
    grupos_risco = segmentacao_de_risco(best_model, dados)
    custos_previstos = prever_encargos_futuros(best_model, dados)
    
    # Definir os planos estratégicos com base nas análises realizadas
    planos_estrategicos = []

    # Exemplo de planos estratégicos com base nos insights obtidos
    for grupo, custo in zip(grupos_risco, custos_previstos):
        if grupo == "Alto Risco" and custo > 13000:  # Exemplo de condição arbitrária
            planos_estrategicos.append("Implementar programas de saúde preventiva para este grupo")
        elif grupo == "Médio Risco":
            planos_estrategicos.append("Realizar campanhas de conscientização sobre saúde")
        else:
            planos_estrategicos.append("Rever políticas de cobertura de seguro")
    
    return planos_estrategicos