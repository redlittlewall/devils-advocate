import argparse
import pandas as pd
import requests
import json
import sys
from tqdm import tqdm

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
# Para usar outros modelos locais via Ollama (ex: 'mistral', 'gemma', 'phi3'),
# basta alterar a variável MODEL_NAME abaixo (após baixar com 'ollama pull <nome>').
#
# Para utilizar APIs em nuvem (ex: OpenAI ChatGPT, Anthropic Claude, Google Gemini),
# você precisará alterar a URL abaixo e modificar o payload na função 'consultar_llama'
# para respeitar a documentação específica de cada provedor, incluindo a chave de API.

LLM_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# Parâmetros de Inferência
TEMPERATURE = 0.0  # 0.0 para determinismo (TCC), até 1.0 para criatividade
MAX_TOKENS = 800  # Limite do tamanho da resposta gerada (Ollama usa 'num_predict')

# Templates de Prompts
PROMPT_GENERICO_BASE = (
    "Você é um consultor sênior de negócios digitais. Avalie a premissa de startup abaixo "
    "utilizando estritamente a ótica do 'Lean Canvas' e 'Design Thinking'.\n\n"
    "DADOS DA STARTUP:\n"
    "Setor: {setor}\n"
    "Premissa: {premissa}\n\n"
    "REGRAS DE SAÍDA: Responda EXCLUSIVAMENTE em formato JSON válido, sem nenhum texto antes ou depois, usando exatamente estas chaves:\n"
    "{{\n"
    '  "analise_desejabilidade": "(Avaliação sobre a dor do mercado e atratividade da solução)",\n'
    '  "analise_viabilidade": "(Avaliação sobre o modelo de receita, custos e concorrência)",\n'
    '  "analise_praticabilidade": "(Avaliação sobre a capacidade técnica de execução)",\n'
    '  "principal_forca": "(O maior ponto positivo ou diferencial do negócio)",\n'
    '  "principal_fraqueza": "(A falha mais crítica ou maior risco)",\n'
    '  "probabilidade_sucesso_0_a_100": (número inteiro entre 0 e 100),\n'
    '  "probabilidade_falencia_0_a_100": (número inteiro entre 0 e 100),\n'
    '  "veredito_final": "(Aprovada / Rejeitada / Necessita Pivotagem)"\n'
    "}}"
)

PROMPT_DIABO_BASE = (
    "Aja como um 'Advogado do Diabo' implacável e auditor de riscos de Venture Capital. "
    "Sua meta é encontrar a falha fatal que levará este negócio à falência, combatendo qualquer viés de otimismo.\n\n"
    "DADOS DA STARTUP:\n"
    "Setor: {setor}\n"
    "Premissa: {premissa}\n\n"
    "REGRAS DE SAÍDA: Responda EXCLUSIVAMENTE em formato JSON válido, sem nenhum texto antes ou depois, usando exatamente estas chaves para o Stress-Test:\n"
    "{{\n"
    '  "analise_desejabilidade": "(Ataque a premissa: Por que os clientes NÃO vão querer isso?)",\n'
    '  "analise_viabilidade": "(Ataque a premissa: Onde o modelo de negócios e a receita vão quebrar?)",\n'
    '  "analise_praticabilidade": "(Ataque a premissa: Quais são os gargalos técnicos ou operacionais fatais?)",\n'
    '  "principal_forca": "(Seja cético, mas aponte o único ponto que talvez faça sentido)",\n'
    '  "principal_fraqueza": "(O motivo exato da ruína projetada)",\n'
    '  "probabilidade_sucesso_0_a_100": (número inteiro entre 0 e 100),\n'
    '  "probabilidade_falencia_0_a_100": (número inteiro entre 0 e 100),\n'
    '  "veredito_final": "(Aprovada / Rejeitada / Necessita Pivotagem)"\n'
    "}}"
)


def consultar_llm(prompt: str, temperature: float = 0.0) -> str:
    """
    Envia um prompt para a API local do LLM e retorna a resposta.

    Args:
        prompt (str): O texto com as instruções e dados para o LLM.
        temperature (float): Controle de aleatoriedade (0.0 = determinístico).

    Returns:
        str: A resposta gerada pelo modelo ou uma mensagem de erro.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,  # Queremos a resposta completa de uma vez, não em pedaços (streaming)
        "format": "json",
        "options": {"temperature": temperature, "num_predict": MAX_TOKENS},
    }

    try:
        response = requests.post(LLM_API_URL, json=payload)
        response.raise_for_status()  # Lança exceção se o status HTTP não for 200 (OK)

        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        print("\n[ERRO CRÍTICO] Não foi possível conectar ao LLM.")
        print(
            "Verifique se o LLM está rodando (tente acessar http://localhost:11434 se estiver rodando o llama localmente) ou configurado corretamente."
        )
        sys.exit(1)  # Encerra o script para não processar o CSV em vão
    except Exception as e:
        return f"[ERRO DURANTE A REQUISIÇÃO]: {str(e)}"


# =====================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# =====================================================================
def executar_experimento(
    arquivo_entrada: str, arquivo_saida: str, limite_linhas: int = None
):
    print(f"Carregando dataset: {arquivo_entrada}...")
    try:
        df = pd.read_csv(arquivo_entrada)

        if limite_linhas is not None:
            print(
                f"\n[MODO TESTE ATIVADO] Processando apenas as primeiras {limite_linhas} startups."
            )
            df = df.head(limite_linhas)
    except FileNotFoundError:
        print(
            f"[ERRO] O arquivo '{arquivo_entrada}' não foi encontrado na pasta atual."
        )
        return

    # Com base na amostra do seu arquivo, a coluna se chama 'Setor_Industria'.
    # Caso seja 'Setor', o código ajusta automaticamente.
    coluna_setor = "Setor_Industria" if "Setor_Industria" in df.columns else "Setor"

    # Validação rápida de integridade dos dados
    colunas_necessarias = ["ID_Startup", coluna_setor, "Premissa_Inicial_Input"]
    for col in colunas_necessarias:
        if col not in df.columns:
            print(f"[ERRO] Coluna esperada '{col}' não encontrada no dataset.")
            print(f"Colunas disponíveis: {df.columns.tolist()}")
            return

    # Listas para armazenar as respostas (mantém a ordem exata do DataFrame)
    respostas_genericas = []
    respostas_advogado_diabo = []

    print(f"Iniciando inferência com o modelo {MODEL_NAME} (Temperatura: 0)")
    print(f"Total de startups a processar: {len(df)}")

    # iterrows() com tqdm para exibir uma barra de progresso elegante no terminal
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processando Startups"):
        setor = row[coluna_setor]
        premissa = row["Premissa_Inicial_Input"]

        # -----------------------------------------------------------
        # CONDIÇÃO 1: Prompt Genérico
        # -----------------------------------------------------------
        prompt_generico = PROMPT_GENERICO_BASE.format(setor=setor, premissa=premissa)
        resposta_gen = consultar_llm(prompt_generico, temperature=TEMPERATURE)
        respostas_genericas.append(resposta_gen)

        # -----------------------------------------------------------
        # CONDIÇÃO 2: Prompt Advogado do Diabo
        # -----------------------------------------------------------
        prompt_diabo = PROMPT_DIABO_BASE.format(setor=setor, premissa=premissa)
        resposta_adv = consultar_llm(prompt_diabo, temperature=TEMPERATURE)
        respostas_advogado_diabo.append(resposta_adv)

        # SALVAMENTO PARCIAL (CHECKPOINT)
        df_parcial = df.iloc[: index + 1].copy()
        df_parcial["Resposta_Generica"] = respostas_genericas
        df_parcial["Resposta_Advogado_Diabo"] = respostas_advogado_diabo
        df_parcial.to_csv(arquivo_saida, index=False, encoding="utf-8")

    # Anexando as novas colunas ao DataFrame original
    df["Resposta_Generica"] = respostas_genericas
    df["Resposta_Advogado_Diabo"] = respostas_advogado_diabo

    # Salvando o resultado final
    print(f"\nSalvando resultados em: {arquivo_saida}...")
    df.to_csv(arquivo_saida, index=False, encoding="utf-8")
    print("Processamento concluído com sucesso!")


if __name__ == "__main__":
    # Configura a leitura de argumentos do terminal
    parser = argparse.ArgumentParser(
        description="Script para experimento LLM com Startups."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Número máximo de startups para processar (para testes rápidos)",
    )
    args = parser.parse_args()

    ARQUIVO_INPUT = "dataset_startups_w_balanced_tags.csv"
    ARQUIVO_OUTPUT = "resultados_experimento.csv"

    executar_experimento(ARQUIVO_INPUT, ARQUIVO_OUTPUT, limite_linhas=args.limit)
