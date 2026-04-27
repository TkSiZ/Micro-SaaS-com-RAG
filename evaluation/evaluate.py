import sys
import os
import textwrap
import math

# Garante que src/ e raiz do projeto estão no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Importa o pipeline já existente no projeto
from src.embeddings import retrieval
from src.llm import generate_answer

# ============================================================
# 0) Configurações
# ============================================================
RAG_MODEL    = "llama3"        # modelo usado no pipeline
JUDGE_MODEL  = "llama3"        # juiz RAGAS
INTERP_MODEL = "qwen2.5:3b"   # modelo leve para interpretar os resultados
EMBED_MODEL  = "intfloat/multilingual-e5-large"  # mesmo embedding do pipeline

# ============================================================
# 1) Conjunto de teste com ground truth
# ============================================================
"""
O ground truth deve ser baseado SOMENTE no que está no PDF music_theory_complete.pdf.
Nunca invente ou use informações externas — isso zeraria context_precision e context_recall.
As respostas foram elaboradas com base no conteúdo real do livro de Toby W. Rush,
"Music Theory for Musicians and Normal People".
"""
PERGUNTAS_E_RESPOSTAS = [
    {
        "question": "What is a perfect interval?",
        "ground_truth": (
            "Perfect intervals include unisons, fourths, fifths, and octaves. "
            "They are called perfect because of their pure sound and historical significance. "
            "A perfect interval remains perfect when inverted."
        ),
    },
    {
        "question": "What is a diatonic triad?",
        "ground_truth": (
            "A diatonic triad is a chord built using only the notes of a particular key signature, "
            "with no accidentals. Diatonic means 'from the key'. "
            "All diatonic triads can be found by building triads on each note of the scale "
            "using only the notes within that key."
        ),
    },
    {
        "question": "What is binary form in music?",
        "ground_truth": (
            "Binary form consists of two contrasting sections, referred to as A and B. "
            "It is one of the simplest musical forms, defined by the arrangement of these "
            "two sections and the keys being used."
        ),
    },
    {
        "question": "What is a suspension in harmony?",
        "ground_truth": (
            "A suspension is a non-harmonic tone that is held over from a previous chord "
            "and then resolves downward. It is identified by two numbers: "
            "the note of suspension and the note of resolution."
        ),
    },
    {
        "question": "What is music notation?",
        "ground_truth": (
            "Music notation is the art of recording music in written form. "
            "Modern music notation is essentially a stylized graph of pitch versus time. "
            "It uses a staff of five lines on which notes are placed to indicate pitch."
        ),
    },
    {
        "question": "What are the species of counterpoint?",
        "ground_truth": (
            "Species counterpoint is a method of learning counterpoint through progressive rules. "
            "In the third species, notes should not leap more than once in the same direction, "
            "and all intervals larger than a third, including perfect fourths, must be counterbalanced."
        ),
    },
    {
        "question": "What is inflection in music intervals?",
        "ground_truth": (
            "Inflection is the quality of an interval beyond its numeric distance. "
            "For unisons, fourths, fifths, and octaves, inflection is described as perfect, "
            "augmented, or diminished. Some theorists use the term 'quality' instead of inflection."
        ),
    },
]

# ============================================================
# 2) Coleta de respostas e contextos via pipeline do projeto
# ============================================================
"""
Reutiliza as funções retrieval() e generate_answer() já existentes em src/.
O RAGAS precisa de:
  - question: a pergunta feita
  - answer: a resposta gerada pelo LLM
  - contexts: lista de chunks recuperados (lista de strings)
  - ground_truth: a resposta de referência correta
"""
print("🔍 Executando perguntas no pipeline RAG...\n")

questions     = []
answers       = []
contexts      = []
ground_truths = []

for item in PERGUNTAS_E_RESPOSTAS:
    pergunta = item["question"]
    print(f"  ❓ {pergunta}")

    chunks_recuperados = retrieval(pergunta, top_k=5)
    resposta = generate_answer(pergunta)

    questions.append(pergunta)
    answers.append(resposta)
    contexts.append(chunks_recuperados)
    ground_truths.append(item["ground_truth"])

    print(f"  💬 {resposta[:120]}...\n")

# ============================================================
# 3) Monta o Dataset RAGAS
# ============================================================
"""
Dataset.from_dict() converte as listas em um dataset HuggingFace.
Os nomes das chaves são obrigatórios — o RAGAS os espera exatamente assim.
Internamente o RAGAS pode renomear 'question' para 'user_input' no DataFrame de saída.
"""
ragas_dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

# ============================================================
# 4) Configura o LLM juiz e embeddings para o RAGAS
# ============================================================
"""
O RAGAS usa um LLM separado como "juiz" para calcular as métricas.
Por padrão ele tentaria usar a OpenAI — substituímos pelo LLaMA 3 local.
LangchainLLMWrapper e LangchainEmbeddingsWrapper adaptam os objetos
do LangChain para o formato interno do RAGAS.
temperature=0 é essencial para o juiz ser determinístico e consistente.
max_workers=1 força execução sequencial para evitar TimeoutError por
sobrecarga do Ollama local.
"""
print("⚙️  Configurando RAGAS com LLaMA 3 como juiz...\n")

run_config = RunConfig(
    timeout=180,
    max_retries=5,
    max_wait=30,
    max_workers=1,   # execução sequencial — evita sobrecarga do Ollama
)

judge_llm   = LangchainLLMWrapper(
    ChatOllama(model=JUDGE_MODEL, temperature=0),
    run_config=run_config,
)
judge_embed = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBED_MODEL)
)

metricas = [faithfulness, answer_relevancy, context_precision, context_recall]
for m in metricas:
    m.llm        = judge_llm
    m.embeddings = judge_embed

# ============================================================
# 5) Avalia e exibe resultados
# ============================================================
print("📊 Rodando avaliação RAGAS...\n")
resultado = evaluate(
    dataset=ragas_dataset,
    metrics=metricas,
    run_config=run_config,
    raise_exceptions=False,
)
df = resultado.to_pandas()

COLUNAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
COL_PERGUNTA = "user_input" if "user_input" in df.columns else "question"

print("\n" + "="*65)
print("           RESULTADOS DA AVALIAÇÃO RAGAS")
print("="*65)
print(df[[COL_PERGUNTA] + COLUNAS].to_string(index=False))

print("\n📈 Médias:")
medias = {}
for c in COLUNAS:
    if c in df.columns:
        col_values = df[c].dropna()
        if len(col_values) > 0:
            v = col_values.mean()
            medias[c] = v
            barra = "█" * int(v * 20)
            print(f"  {c:<22}: {v:.4f}  {barra}  (n={len(col_values)}/{len(df)})")
        else:
            print(f"  {c:<22}: sem dados válidos")

csv_path = "evaluation/ragas_resultados.csv"
os.makedirs("evaluation", exist_ok=True)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n💾 Resultados salvos em: {csv_path}")

# ============================================================
# 6) Intérprete LLM — Qwen2.5:3b analisa os resultados
# ============================================================
"""
Um LLM leve interpreta os scores em linguagem natural.
O Qwen2.5:3b é suficiente porque todo o contexto já está no prompt —
o modelo só precisa raciocinar sobre os números e gerar o diagnóstico.
"""
print("\n" + "="*65)
print("       INTERPRETAÇÃO AUTOMÁTICA (qwen2.5:3b)")
print("="*65 + "\n")

linhas_resultados = []
for _, row in df.iterrows():
    linha = f"- Pergunta: \"{row[COL_PERGUNTA]}\"\n"
    for c in COLUNAS:
        v = row[c]
        linha += f"  {c}={'NaN' if math.isnan(v) else f'{v:.2f}'} | "
    linhas_resultados.append(linha.rstrip(" | "))

resumo_str = "\n".join(linhas_resultados)
medias_str = " | ".join(f"{k}={v:.2f}" for k, v in medias.items())

PROMPT_INTERPRETACAO = f"""Você é um especialista em avaliação de sistemas RAG (Retrieval-Augmented Generation).

Abaixo estão os resultados de uma avaliação RAGAS de um pipeline RAG sobre um livro de teoria musical.

MÉTRICAS (escala 0 a 1, quanto maior melhor):
- faithfulness: a resposta gerada é fiel ao contexto recuperado? (baixo = alucinação)
- answer_relevancy: a resposta é relevante para a pergunta? (baixo = resposta tangencial)
- context_precision: os chunks recuperados são precisos e relevantes? (baixo = ruído no retriever)
- context_recall: o contexto recuperado cobre a resposta esperada? (baixo = retriever perdendo informações)

MÉDIAS GERAIS:
{medias_str}

DETALHAMENTO POR PERGUNTA:
{resumo_str}

Com base nesses resultados:
1. Identifique os pontos críticos do pipeline.
2. Explique em linguagem simples o que cada problema significa na prática.
3. Sugira melhorias concretas e priorizadas (o que corrigir primeiro).
4. Dê uma nota geral ao pipeline de 0 a 10 e justifique.

Responda em português, de forma clara e objetiva.
"""

print("🧠 Chamando qwen2.5:3b para interpretar...\n")
interp_llm = ChatOllama(model=INTERP_MODEL, temperature=0.3)

try:
    resposta_interp = interp_llm.invoke(PROMPT_INTERPRETACAO)
    interpretacao   = resposta_interp.content

    for linha in interpretacao.splitlines():
        print(textwrap.fill(linha, width=80) if linha.strip() else "")

    interp_path = "evaluation/ragas_interpretacao.txt"
    with open(interp_path, "w", encoding="utf-8") as f:
        f.write("INTERPRETAÇÃO AUTOMÁTICA — qwen2.5:3b\n")
        f.write("="*65 + "\n\n")
        f.write(f"Prompt enviado:\n{PROMPT_INTERPRETACAO}\n\n")
        f.write("="*65 + "\n\nResposta:\n\n")
        f.write(interpretacao)

    print(f"\n💾 Interpretação salva em: {interp_path}")

except Exception as e:
    print(f"⚠️  Erro ao chamar o intérprete: {e}")
    print("   Verifique se o modelo está disponível: ollama pull qwen2.5:3b")

# ============================================================
# NOTAS IMPORTANTES
# ============================================================
"""
DIFERENÇAS EM RELAÇÃO AO EXEMPLO DO PROFESSOR:
─────────────────────────────────────────────────────────────
1. Usa ChromaDB (projeto) em vez de FAISS
   → As funções retrieval() e generate_answer() do src/ são
     reutilizadas diretamente, sem duplicar o pipeline.

2. Perguntas em inglês
   → O PDF music_theory_complete.pdf está em inglês.
     Perguntas em inglês maximizam context_precision e recall
     porque os chunks recuperados também estão em inglês.
     O prompt em src/llm.py garante que a resposta saia em PT-BR.

3. Ground truth baseado no conteúdo real do livro
   → Toby W. Rush, "Music Theory for Musicians and Normal People"
     Verificado nas páginas 3, 11, 21, 31, 36 e 51 do PDF.

4. max_workers=1 para evitar TimeoutError
   → Execução sequencial das chamadas ao juiz LLaMA 3,
     evitando sobrecarga do Ollama local.

ARMADILHAS COMUNS (conforme aula do Prof. Fabio Santos):
─────────────────────────────────────────────────────────────
- Ground truth inventado → context_precision e recall = 0
- Perguntas sem resposta no PDF → retriever nunca cobre o ground truth
- LLM juiz muito pequeno → erros na avaliação de faithfulness
- Não verificar nome da coluna → erro ao acessar df["question"]
─────────────────────────────────────────────────────────────
"""