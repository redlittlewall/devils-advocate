# Mitigação de vieses otimistas de inteligência artificial na validação de modelos de negócios digitais
Este repositório contém o código-fonte, metodologia e estrutura de dados do experimento científico conduzido para o Trabalho de Conclusão de Curso (Especialização em Gestão de Negócios Digitais e Inteligência Artificial – ESALQ/USP) focado em Inteligência Artificial aplicada ao ecossistema de inovação e startups.

Alinhado com os princípios de **Ciência Aberta (Open Science)**, este projeto foi desenhado para ser totalmente **reprodutível**, **auditável** e **transparente**. Qualquer pesquisador pode replicar exatamente os mesmos testes e resultados localmente sem depender de APIs pagas ou proprietárias.

---

## 🔬 Resumo do Experimento

O objetivo deste estudo é analisar como um Grande Modelo de Linguagem (*Large Language Model* - LLM), executando localmente sob condições rígidas de determinismo, comporta-se ao avaliar premissas iniciais de negócios de startups reais (tanto casos de sucesso histórico quanto de falhas).

O experimento submete cada startup a duas personas (condições de contorno de prompt) distintas:
1. **Condição 1 (Prompt Genérico):** Atua como um assistente virtual padrão de negócios, tendendo à análise de viabilidade tradicional.
2. **Condição 2 (Prompt Advogado do Diabo):** Atua estritamente de forma implacável, focada em auditoria de risco e mitigação do viés de otimismo (*sicofancia*), tentando falsificar a tese de negócio de forma agressiva.

Para garantir o rigor científico, a temperatura do modelo é configurada em **0.0**, removendo a estocasticidade e garantindo que as respostas sejam determinísticas e reproduzíveis.

---

## 🛠️ Requisitos de Infraestrutura e Configuração

### 1. Modelo de IA Local (Ollama)
Este projeto utiliza o **Ollama** para gerenciar e executar os modelos localmente.

1. Baixe e instale o Ollama conforme o seu sistema operativo através do site oficial: [ollama.com](https://ollama.com)
2. No seu terminal, faça o download do modelo específico utilizado no experimento (`llama3`):
   ```bash
   ollama pull llama3
   ```
3. Certifique-se de que o serviço do Ollama está a ser executado em background (por padrão, ele expõe a API em `http://localhost:11434`).

### 2. Ambiente de Execução Python
Devido às diretrizes modernas de isolamento do ecossistema Python (PEP 668), é obrigatório o uso de um ambiente virtual (*Virtual Environment*) para instalar as dependências de forma segura, sem interferir com o sistema operativo.

No terminal, dentro da pasta do projeto, execute os seguintes passos:

**Instalar suporte a ambientes virtuais (caso ainda não possua - sistemas Linux/Debian):**
```bash
sudo apt update
sudo apt install python3-venv
```

**Criar o ambiente virtual local:**
```bash
python3 -m venv .venv
```

**Ativar o ambiente virtual:**
* No Linux/macOS:
  ```bash
  source .venv/bin/activate
  ```
* No Windows (Prompt de Comando):
  ```cmd
  .venv\Scripts\activate
  ```
* No Windows (PowerShell):
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

*Nota: O seu terminal exibirá o prefixo `(.venv)` indicando que o isolamento está ativo.*

**Instalar as dependências científicas:**
```bash
pip install pandas requests tqdm matplotlib seaborn 
```

---

## 📁 Estrutura de Dados (Dataset)

O projeto consome um arquivo estruturado `dataset_startups_w_balanced_tags.csv` localizado no mesmo diretório do script. O dataset é composto por colunas com metadados e as premissas de negócio anonimizadas:

* `ID_Startup`: Identificador único da startup no experimento.
* `Nome_Anonimizado`: Nome fictício ou real mascarado utilizado para evitar viés histórico do LLM.
* `Setor_Industria`: O segmento de mercado da empresa (ex: *HealthTech*, *FinTech*, *MediaTech*).
* `Premissa_Inicial_Input`: A tese de negócio explícita que o modelo deve analisar.

### Output Gerado
Após a execução do script, os dados processados serão armazenados no diretório `resultados/`. Um novo arquivo CSV será gerado contendo o dataset original enriquecido com as respostas da IA. 

Para garantir a extração analítica e matemática, as saídas do modelo são forçadas para o formato **JSON estruturado**, contendo métricas comparáveis (ex: `probabilidade_sucesso_0_a_100` e `probabilidade_falencia_0_a_100`):
* `Resposta_Generica`: O parecer analítico (em JSON) gerado sob a persona de Assistente de Negócios, ancorado no framework *Lean Canvas*.
* `Resposta_Advogado_Diabo`: O parecer de stress-test (em JSON) gerado sob a persona de auditor de risco, ancorado em frameworks de *Venture Capital*.

---

## 🚀 Como Executar o Experimento

Com o Ollama a rodar e o seu ambiente virtual ativo, basta executar o script principal:

```bash
# Para rodar o experimento completo com toda a base de dados:
python3 experimento_tcc.py

# Para rodar testes rápidos (ex: processar apenas as primeiras 2 startups):
python3 experimento_tcc.py --limit 2
```

O script possui uma barra de progresso em tempo real que indicará o progresso e o tempo estimado para o encerramento do processamento em massa do dataset.

---

## 📊 Reprodutibilidade e Rigor Científico

Para assegurar os princípios do Open Science, os parâmetros de inferência HTTP foram estritamente isolados da seguinte forma:
```json
{
  "model": "llama3",
  "stream": false,
  "format": "json",
  "options": {
    "temperature": 0.0
  }
}
```
* **Ausência de Histórico de Sessão (Stateless):** Cada requisição HTTP efetuada para o Ollama é independente. O modelo avalia cada prompt sem conhecimento prévio de startups anteriores do dataset ou de personas opostas executadas na mesma linha, garantindo a neutralidade do contexto.
* **Formato Estruturado Estrito:** A requisição exige que a resposta do modelo seja um JSON válido ("format": "json"), eliminando a aleatoriedade e variações de prosa que inviabilizariam a análise quantitativa.
* **Temperatura Zero:** Reduz o sampling probabilístico a zero, forçando o modelo a selecionar sempre os tokens mais prováveis (*greedy decoding*), permitindo que outros cientistas gerem exatamente o mesmo output textual utilizando o mesmo seed.

---

## ✒️ Como Citar Este Projeto

Caso utilize este código, o dataset ou a metodologia adaptada deste repositório em sua investigação académica ou artigos científicos, por favor cite conforme a estrutura abaixo:

```bibtex
@misc{chiari2026mitigacao,
  author       = {Chiari, Murilo Ferrarezi and Valotto, Daniel},
  title        = {Mitigação de vieses otimistas de inteligência artificial na validação de modelos de negócios digitais},
  howpublished = {Trabalho de Conclusão de Curso (Especialização em Gestão de Negócios Digitais e Inteligência Artificial) - ESALQ/USP},
  year         = {2026},
  note         = {Autor correspondente: murilofch@gmail.com | Orientador: danielsvalotto@gmail.com},
  url          = {[https://github.com/SEU_USUARIO/SEU_REPOSITORIO](https://github.com/SEU_USUARIO/SEU_REPOSITORIO)}
}
```

---

## 📄 Licença

Este projeto está sob a licença **MIT License** - consulte o arquivo `LICENSE` para mais detalhes. O uso livre para fins científicos, educacionais e comerciais é totalmente encorajado, desde que mantidos os créditos originais.