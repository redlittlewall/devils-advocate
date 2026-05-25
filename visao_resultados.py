import pandas as pd
import json
import re

# 1. Carregar os dados do seu arquivo atualizado
df = pd.read_csv("resultados/resultados_experimento.csv")

prompts = ["Gen", "Anali", "Adv"]
colunas_resposta_csv = {
    "Gen": "Resposta_Generica",
    "Anali": "Resposta_Analitica",
    "Adv": "Resposta_Advogado_Diabo",
}

p_nomes_bonitos = {
    "Gen": "Genérico",
    "Anali": "Analítico",
    "Adv": "Advogado do Diabo",
}


def extrair_campo_json(texto_celula, campo_alvo):
    if pd.isna(texto_celula):
        return "Não registrado."
    texto_str = str(texto_celula).strip()
    try:
        dados = json.loads(texto_str)
        if campo_alvo in dados:
            val = dados[campo_alvo]
            if isinstance(val, dict) and len(val) > 0:
                primeira_chave = list(val.keys())[0]
                return str(
                    val[primeira_chave][0]
                    if isinstance(val[primeira_chave], list)
                    else val[primeira_chave]
                )
            return str(val)
    except:
        pass
    padrao = r'"' + re.escape(campo_alvo) + r'"\s*:\s*(.*?)(?=\s*,\s*"|\s*\}\s*$)'
    match = re.search(padrao, texto_str, re.DOTALL)
    if match:
        conteudo = match.group(1).strip()
        if conteudo.startswith("{"):
            sub_match = re.search(r'":\s*\[?\s*"(.*?)"', conteudo)
            if sub_match:
                return sub_match.group(1)
        return (
            conteudo.strip('"')
            .strip("]")
            .strip("[")
            .strip('"')
            .replace("\\n", " ")
            .replace('\\"', '"')
        )
    return "Verificar no CSV original."


# 2. Processar a matriz e injetar no template HTML/JavaScript
linhas_html = ""
dados_modais_js = {}

for idx, row in df.iterrows():
    startup_id = str(row["ID_Startup"])
    nome_startup = str(row["Nome_Real"])
    setor = str(row["Setor_Industria"])
    status_real = str(row["Status_Real"]).upper()
    tag_real = str(row["Rotulo_Categorico"])
    motivo_real = str(row["Motivo_Real_Gabarito"])
    premissa_input = str(row["Premissa_Inicial_Input"])

    detalhes_prompts_html = ""

    for p in prompts:
        col_resp = colunas_resposta_csv[p]
        texto_forca = extrair_campo_json(row[col_resp], "principal_forca")
        texto_fraqueza = extrair_campo_json(row[col_resp], "principal_fraqueza")

        v_cru = str(row[f"Veredito_{p}"]).lower()
        if "rejeitada" in v_cru:
            style_veredito = "background-color: #f2d7d5; color: #922b21; border-left: 4px solid #c0392b;"
            veredito_final = "REJEITADA ❌"
        elif "condições" in v_cru:
            style_veredito = "background-color: #fdebd0; color: #b9770e; border-left: 4px solid #f39c12;"
            veredito_final = "APROVADA COM CONDIÇÕES ⚠️"
        else:
            style_veredito = "background-color: #d4efdf; color: #196f3d; border-left: 4px solid #27ae60;"
            veredito_final = "APROVADA"

        val_prob = row[f"Prob_Sucesso_{p}"]
        if pd.isna(val_prob):
            prob = 0
            prob_str = "N/A"
        else:
            prob = int(val_prob)
            prob_str = f"{prob}%"

        alpha = max(0.1, min(1.0, prob / 100.0))
        style_prob = (
            f"background-color: rgba(46, 204, 113, {alpha}); font-weight: bold;"
        )

        style_gabarito = (
            "background-color: #fadbd8; color: #78281f; font-weight: bold;"
            if status_real == "FALHA"
            else "background-color: #d4efdf; color: #145a32; font-weight: bold;"
        )
        tag_badge_class = "badge-fail" if status_real == "FALHA" else "badge-success"

        linhas_html += f"""
        <tr onclick="abrirModal('{startup_id}')" style="cursor: pointer;">
            <td>{startup_id}</td>
            <td><b>{nome_startup}</b></td>
            <td>{setor}</td>
            <td style="{style_gabarito} text-align: center;">{status_real}</td>
            <td><span class="badge {tag_badge_class}">{tag_real}</span></td>
            <td>{p_nomes_bonitos[p]}</td>
            <td style="{style_veredito}">{veredito_final}</td>
            <td style="{style_prob} text-align: center;">{prob_str}</td>
        </tr>
        """

        detalhes_prompts_html += f"""
        <div class="prompt-card">
            <h4>{p_nomes_bonitos[p]} (Veredito: {veredito_final} | Nota de Sucesso: {prob_str})</h4>
            <p><b>Argumento de Sucesso (Força):</b> {texto_forca}</p>
            <p><b>Argumento de Falha (Fraqueza):</b> {texto_fraqueza}</p>
        </div>
        """

    dados_modais_js[startup_id] = {
        "nome": nome_startup,
        "setor": setor,
        "status": status_real,
        "tag": tag_real,
        "motivo": motivo_real,
        "premissa": premissa_input,
        "detalhes_ia": detalhes_prompts_html,
    }

js_dict_str = json.dumps(dados_modais_js, ensure_ascii=False)

# Template Estrutural da Interface Web
html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <title>Painel Interativo de Análise - TCC MBA</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; background-color: #f4f6f7; color: #2c3e50; }
        h2 { color: #2c3e50; margin-bottom: 5px; font-weight: 600; }
        .subtitle { color: #7f8c8d; margin-top: 0; margin-bottom: 25px; font-size: 14px; }
        .table-container { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; background: white; }
        th { background-color: #34495e; color: white; padding: 14px 10px; font-size: 12px; text-align: left; text-transform: uppercase; letter-spacing: 0.5px; }
        td { padding: 12px 10px; font-size: 13px; border-bottom: 1px solid #eaeded; }
        tr:hover { background-color: #ebf5fb !important; transition: 0.2s; }
        tr:nth-child(even) { background-color: #fafcfc; }
        
        .badge { padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; display: inline-block; }
        .badge-fail { background-color: #fdedd0; color: #e67e22; border: 1px solid #f5b041; }
        .badge-success { background-color: #e8f8f5; color: #1e8449; border: 1px solid #2ecc71; }
        
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.6); backdrop-filter: blur(3px); }
        .modal-content { background-color: #ffffff; margin: 4% auto; padding: 25px; border-radius: 10px; width: 75%; max-height: 85vh; overflow-y: auto; box-shadow: 0 5px 25px rgba(0,0,0,0.2); animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        .close-btn { color: #95a5a6; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }
        .close-btn:hover { color: #2c3e50; }
        .modal-header { border-bottom: 2px solid #34495e; padding-bottom: 10px; margin-bottom: 20px; }
        
        .grid-gabarito { display: flex; gap: 20px; margin-bottom: 20px; background: #f8f9f9; border-radius: 6px; border-left: 5px solid #34495e; padding: 15px; }
        .grid-cell { flex: 1; min-width: 0; }
        .prompt-card { background: #fdfefe; border: 1px solid #e5e8e8; padding: 15px; margin-bottom: 12px; border-radius: 6px; }
        .prompt-card h4 { margin-top: 0; color: #2c3e50; margin-bottom: 8px; font-size: 14px; border-bottom: 1px dashed #d5dbdb; padding-bottom: 4px; }
        .prompt-card p { margin: 5px 0; font-size: 12.5px; }
        .tip { background: #eaf2f8; padding: 8px 12px; font-size: 11px; color: #2471a3; border-radius: 4px; margin-bottom: 15px; font-weight: bold; }
    </style>
</head>
<body>

    <h2>Painel Interativo de Validação Qualitativa (Mundo Real vs LLM)</h2>
    <p class="subtitle">Mestrado/MBA - Análise de Cadeia de Pensamento (CoT) e Impacto dos Rótulos Categóricos na Detecção de Falhas Estruturais</p>
    
    <div class="tip">💡 DICA ANALÍTICA: Clique em qualquer linha da tabela para expandir o painel e ver o cenário real completo e as justificativas detalhadas de cada prompt.</div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th style="width: 50px;">ID</th>
                    <th style="width: 130px;">Startup</th>
                    <th style="width: 120px;">Setor</th>
                    <th style="width: 100px; text-align: center;">Status Real</th>
                    <th style="width: 180px;">Tag de Mercado (Gabarito)</th>
                    <th style="width: 160px;">Prompt / Persona</th>
                    <th style="width: 180px;">Veredito da IA</th>
                    <th style="width: 90px; text-align: center;">Prob. Sucesso</th>
                </tr>
            </thead>
            <tbody>
                {LINHAS_TABELA}
            </tbody>
        </table>
    </div>

    <div id="modalDetalhes" class="modal">
        <div class="modal-content">
            <span class="close-btn" onclick="fecharModal()">&times;</span>
            <div id="conteudoDinamicoModal"></div>
        </div>
    </div>

    <script>
        const dadosStartups = {DADOS_JSON};

        function abrirModal(id) {
            const data = dadosStartups[id];
            if(!data) return;
            
            const corStatus = data.status === 'FALHA' ? '#78281f' : '#145a32';
            const bgStatus = data.status === 'FALHA' ? '#fadbd8' : '#d4efdf';
            const badgeClass = data.status === 'FALHA' ? 'badge-fail' : 'badge-success';

            let htmlModal = `
                <div class="modal-header">
                    <h2 style="margin:0;">${data.nome} <span style="font-size:14px; font-weight:normal; color:#7f8c8d;">(${data.setor})</span></h2>
                </div>
                
                <div class="grid-gabarito">
                    <div class="grid-cell" style="border-right: 1px solid #e5e8e8; padding-right: 15px;">
                        <h3 style="margin-top:0; font-size:14px; color:#34495e;">📋 PREMISSA SUBMETIDA À IA (INPUT)</h3>
                        <p style="font-size:12px; line-height:1.5; margin:0;">${data.premissa}</p>
                    </div>
                    <div class="grid-cell" style="padding-left: 5px;">
                        <h3 style="margin-top:0; font-size:14px; color:#34495e;">🔍 DESFECHO DE MERCADO (MUNDO REAL)</h3>
                        <p style="margin:5px 0;">
                            <span style="background-color: ${bgStatus}; color: ${corStatus}; font-weight:bold; padding:4px 8px; border-radius:4px; font-size:11px;">${data.status}</span> 
                            <span class="badge ${badgeClass}">${data.tag}</span>
                        </p>
                        <p style="font-size:12px; line-height:1.5; margin:8px 0 0 0;"><b>Histórico Real:</b> ${data.motivo}</p>
                    </div>
                </div>

                <h3 style="font-size:15px; color:#2c3e50; margin-bottom:10px; border-bottom: 2px solid #bdc3c7; padding-bottom:5px;">🧠 CADEIA DE RACIOCÍNIO E JUSTIFICATIVAS TEXTUAIS DA IA</h3>
                ${data.detalhes_ia}
            `;

            document.getElementById('conteudoDinamicoModal').innerHTML = htmlModal;
            document.getElementById('modalDetalhes').style.display = 'block';
        }

        function fecharModal() {
            document.getElementById('modalDetalhes').style.display = 'none';
        }

        window.onclick = function(event) {
            const modal = document.getElementById('modalDetalhes');
            if (event.target == modal) {
                modal.style.display = 'none';
            }}
    </script>
</body>
</html>
"""

html_final = html_template.replace("{LINHAS_TABELA}", linhas_html).replace(
    "{DADOS_JSON}", js_dict_str
)

with open("painel_interativo_tcc.html", "w", encoding="utf-8") as f:
    f.write(html_final)

print("✨ Painel Avançado Gerado! Abra 'painel_interativo_tcc.html' no navegador.")
