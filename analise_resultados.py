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
        # Tenta interpretar a string como JSON
        dados = json.loads(json_str)

        sucesso = dados.get("probabilidade_sucesso_0_a_100", np.nan)
        falencia = dados.get("probabilidade_falencia_0_a_100", np.nan)
        veredito = dados.get("veredito_final", "N/A")

        return pd.Series([sucesso, falencia, veredito])
    except json.JSONDecodeError:
        # Se a IA alucinou e não gerou um JSON perfeito nesta linha
        return pd.Series([np.nan, np.nan, "JSON Inválido"])


def gerar_analises():
    caminho_entrada = "resultados/resultados_experimento.csv"

    print(f"Carregando os resultados estruturados de: {caminho_entrada}...")
    try:
        df = pd.read_csv(caminho_entrada)
    except FileNotFoundError:
        print(
            f"[ERRO] Arquivo não encontrado. Certifique-se de que o experimento já rodou."
        )
        return

    print("Extraindo métricas matemáticas dos JSONs...")

    # Extraindo dados da Resposta Genérica
    df[["Prob_Sucesso_Gen", "Prob_Falencia_Gen", "Veredito_Gen"]] = df[
        "Resposta_Generica"
    ].apply(extrair_metricas_json)

    # Extraindo dados da Resposta Advogado do Diabo
    df[["Prob_Sucesso_Adv", "Prob_Falencia_Adv", "Veredito_Adv"]] = df[
        "Resposta_Advogado_Diabo"
    ].apply(extrair_metricas_json)

    # Removendo linhas onde a extração falhou para não sujar a média
    df_clean = df.dropna(subset=["Prob_Sucesso_Gen", "Prob_Sucesso_Adv"])

    # =====================================================================
    # GRÁFICO 1: Comparação de Risco Médio (Genérico vs Advogado do Diabo)
    # =====================================================================
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    medias = {
        "Prompt Genérico (Controle)": [
            df_clean["Prob_Sucesso_Gen"].mean(),
            df_clean["Prob_Falencia_Gen"].mean(),
        ],
        "Advogado do Diabo (Intervenção)": [
            df_clean["Prob_Sucesso_Adv"].mean(),
            df_clean["Prob_Falencia_Adv"].mean(),
        ],
    }

    df_grafico = pd.DataFrame(
        medias, index=["Probabilidade de Sucesso (%)", "Probabilidade de Falência (%)"]
    )

    # Criando gráfico de barras lado a lado
    ax = df_grafico.plot(
        kind="bar", figsize=(10, 6), color=["#4C72B0", "#C44E52"], edgecolor="black"
    )

    plt.title(
        "Impacto da Persona na Previsão do LLM (Mitigação de Sicofância)",
        fontsize=14,
        pad=15,
    )
    plt.ylabel("Porcentagem Média Atribuída pela IA", fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.ylim(0, 100)  # Força o eixo Y ir até 100%
    plt.legend(title="Condição Experimental", fontsize=11)

    # Adicionando os números exatos em cima das barras
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%", (p.get_x() * 1.005, p.get_height() * 1.02)
        )

    plt.tight_layout()
    caminho_grafico = "resultados/grafico_probabilidades.png"
    plt.savefig(caminho_grafico, dpi=300)
    print(f"Gráfico 1 salvo com sucesso em: {caminho_grafico}")

    # =====================================================================
    # MÉTRICAS PARA A REDAÇÃO DO TCC
    # =====================================================================
    print("\n" + "=" * 60)
    print("INSIGHTS QUANTITATIVOS PARA O TCC (RESULTADOS DO TESTE A/B)")
    print("=" * 60)

    media_suc_gen = df_clean["Prob_Sucesso_Gen"].mean()
    media_suc_adv = df_clean["Prob_Sucesso_Adv"].mean()
    delta_sucesso = media_suc_gen - media_suc_adv

    print(f"1. AVALIAÇÃO DE SUCESSO (Otimismo):")
    print(f"   - Média Genérica: {media_suc_gen:.1f}%")
    print(f"   - Média Adv. Diabo: {media_suc_adv:.1f}%")
    print(
        f"   --> O viés de otimismo foi reduzido em {delta_sucesso:.1f} pontos percentuais.\n"
    )

    media_fal_gen = df_clean["Prob_Falencia_Gen"].mean()
    media_fal_adv = df_clean["Prob_Falencia_Adv"].mean()
    delta_falencia = media_fal_adv - media_fal_gen

    print(f"2. AVALIAÇÃO DE RISCO (Criticidade):")
    print(f"   - Média Genérica: {media_fal_gen:.1f}%")
    print(f"   - Média Adv. Diabo: {media_fal_adv:.1f}%")
    print(
        f"   --> A percepção de risco aumentou em {delta_falencia:.1f} pontos percentuais.\n"
    )

    # Contagem de Vereditos
    print("3. DISTRIBUIÇÃO DE VEREDITOS:")
    print(
        "   [Genérico]     Aprovadas:",
        (df_clean["Veredito_Gen"].str.contains("Aprovada", case=False, na=False)).sum(),
    )
    print(
        "   [Adv. Diabo]   Aprovadas:",
        (df_clean["Veredito_Adv"].str.contains("Aprovada", case=False, na=False)).sum(),
    )

    # Salva os dados processados para fácil leitura no Excel
    caminho_csv_final = "resultados/dados_finais_tcc_analisados.csv"
    df.to_csv(caminho_csv_final, index=False)
    print(f"\nPlanilha final com os dados extraídos salva em: {caminho_csv_final}")


if __name__ == "__main__":
    # Garante que a pasta resultados existe antes de rodar
    os.makedirs("resultados", exist_ok=True)
    gerar_analises()
