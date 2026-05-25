import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os


def extrair_metricas_json(json_str):
    """
    Tenta carregar a string JSON e extrair as probabilidades matemáticas e o veredito.
    Retorna (Sucesso, Falência, Veredito)
    """
    if not isinstance(json_str, str):
        return pd.Series([np.nan, np.nan, "Erro"])

    try:
        dados = json.loads(json_str)
        sucesso = dados.get("probabilidade_sucesso_0_a_100", np.nan)
        falencia = dados.get("probabilidade_falencia_0_a_100", np.nan)
        veredito = dados.get("veredito_final", "N/A")
        return pd.Series([sucesso, falencia, veredito])
    except json.JSONDecodeError:
        return pd.Series([np.nan, np.nan, "JSON Inválido"])


def gerar_analises():
    caminho_entrada = "resultados/resultados_experimento.csv"

    print(f"Carregando os resultados estruturados de: {caminho_entrada}...")
    try:
        df = pd.read_csv(caminho_entrada)
    except FileNotFoundError:
        print(
            "[ERRO] Arquivo não encontrado. Certifique-se de que o experimento já rodou."
        )
        return

    print("Extraindo métricas matemáticas dos JSONs...")

    # Extraindo dados das 3 Personas
    df[["Prob_Sucesso_Gen", "Prob_Falencia_Gen", "Veredito_Gen"]] = df[
        "Resposta_Generica"
    ].apply(extrair_metricas_json)
    df[["Prob_Sucesso_Adv", "Prob_Falencia_Adv", "Veredito_Adv"]] = df[
        "Resposta_Advogado_Diabo"
    ].apply(extrair_metricas_json)
    df[["Prob_Sucesso_Anali", "Prob_Falencia_Anali", "Veredito_Anali"]] = df[
        "Resposta_Analitica"
    ].apply(extrair_metricas_json)

    # Removendo linhas onde a extração falhou para qualquer um dos 3 modelos
    df_clean = df.dropna(
        subset=["Prob_Sucesso_Gen", "Prob_Sucesso_Adv", "Prob_Sucesso_Anali"]
    )

    # =====================================================================
    # GRÁFICO 1: Comparação de Risco Médio (Genérico vs Diabo v vs Analista)
    # =====================================================================
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))

    medias = {
        "Genérico (Controle)": [
            df_clean["Prob_Sucesso_Gen"].mean(),
            df_clean["Prob_Falencia_Gen"].mean(),
        ],
        "Adv. do Diabo (Intervenção)": [
            df_clean["Prob_Sucesso_Adv"].mean(),
            df_clean["Prob_Falencia_Adv"].mean(),
        ],
        "Analista": [
            df_clean["Prob_Sucesso_Anali"].mean(),
            df_clean["Prob_Falencia_Anali"].mean(),
        ],
    }

    df_grafico = pd.DataFrame(
        medias, index=["Probabilidade de Sucesso (%)", "Probabilidade de Falência (%)"]
    )

    # Criando gráfico de barras com 3 cores (Azul, Verde, Vermelho)
    ax = df_grafico.plot(
        kind="bar",
        figsize=(12, 7),
        color=["#4C72B0", "#55A868", "#C44E52"],
        edgecolor="black",
    )

    plt.title(
        "Impacto da Persona na Previsão do LLM (Viés vs. Realismo vs. Ceticismo vs. Análise vs. Visão vs. Comitê)",
        fontsize=14,
        pad=15,
    )
    plt.ylabel("Porcentagem Média Atribuída pela IA", fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.ylim(0, 115)  # Espaço extra no topo para os números não cortarem
    plt.legend(title="Condição Experimental", fontsize=11, loc="upper right")

    # Centralizando os números exatamente em cima de cada barra
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 2),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    caminho_grafico = "resultados/grafico_probabilidades.png"
    plt.savefig(caminho_grafico, dpi=300)
    print(f"Gráfico salvo com sucesso em: {caminho_grafico}")

    # =====================================================================
    # MÉTRICAS PARA A REDAÇÃO DO TCC
    # =====================================================================
    print("\n" + "=" * 60)
    print("INSIGHTS QUANTITATIVOS PARA O TCC (TESTE A/B/C)")
    print("=" * 60)

    print("1. AVALIAÇÃO MÉDIA DE SUCESSO (Expectativa Positiva):")
    print(f"   - Genérico:   {df_clean['Prob_Sucesso_Gen'].mean():.1f}%")
    print(f"   - Adv. Diabo: {df_clean['Prob_Sucesso_Adv'].mean():.1f}%\n")
    print(f"   - Analista:   {df_clean['Prob_Sucesso_Anali'].mean():.1f}%")

    print("2. AVALIAÇÃO MÉDIA DE RISCO (Criticidade):")
    print(f"   - Genérico:   {df_clean['Prob_Falencia_Gen'].mean():.1f}%")
    print(f"   - Adv. Diabo: {df_clean['Prob_Falencia_Adv'].mean():.1f}%\n")
    print(f"   - Analista:   {df_clean['Prob_Falencia_Anali'].mean():.1f}%")

    # Contagem de Vereditos exatos
    print("3. DISTRIBUIÇÃO DE VEREDITOS DE APROVAÇÃO (TOTAL: 20 STARTUPS):")
    print(
        "   [Genérico]     Aprovadas:",
        (df_clean["Veredito_Gen"].str.contains("Aprovada", case=False, na=False)).sum(),
    )
    print(
        "   [Adv. Diabo]   Aprovadas:",
        (df_clean["Veredito_Adv"].str.contains("Aprovada", case=False, na=False)).sum(),
    )
    print(
        "   [Analista]     Aprovadas:",
        (
            df_clean["Veredito_Anali"].str.contains("Aprovada", case=False, na=False)
        ).sum(),
    )

    caminho_csv_final = "resultados/dados_finais_tcc_analisados.csv"
    df.to_csv(caminho_csv_final, index=False)
    print(f"\nPlanilha final salva em: {caminho_csv_final}")


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    gerar_analises()
