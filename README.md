# 🎵 Tutor de Teoria Musical — Micro SaaS com RAG

Sistema de perguntas e respostas sobre teoria musical baseado em um pipeline RAG (*Retrieval-Augmented Generation*) com modelos de linguagem e embeddings open-source.

---

## 📌 Definição do Problema

Estudantes de teoria musical frequentemente precisam consultar conceitos específicos em livros extensos — escalas, intervalos, cadências, modos, harmonia funcional — sem saber em qual capítulo encontrá-los. O processo é lento e fragmentado.

Este projeto resolve esse problema oferecendo um **tutor conversacional** capaz de responder perguntas em linguagem natural com base no conteúdo de materiais indexados, sempre em português brasileiro, sem depender de APIs pagas ou conexão com serviços externos.

**Domínio escolhido:** Educação musical  
**Relevância prática:** Pode ser usado por estudantes de conservatórios, cursos técnicos e autodidatas como ferramenta de estudo assistido.

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                   INDEXAÇÃO (offline)                │
│                                                     │
│  PDF  →  Extração de texto  →  Chunking  →  Embeddings  →  ChromaDB  │
│         (PyMuPDF / pdfplumber)  (RecursiveCharacter)  (E5-large)      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  INFERÊNCIA (online)                 │
│                                                     │
│  Pergunta  →  Embedding  →  Busca vetorial  →  Top-5 chunks          │
│                              (ChromaDB / cosseno)                     │
│                                    ↓                                  │
│                             Prompt + contexto  →  LLaMA 3  →  Resposta│
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Decisões Técnicas

### Extração de texto
Utiliza **PyMuPDF** como extrator primário por sua velocidade e fidelidade ao texto. Caso falhe (PDFs escaneados ou corrompidos), faz fallback automático para **pdfplumber**.

### Estratégia de Chunking
Utiliza `RecursiveCharacterTextSplitter` com:

- **Chunk size:** 512 caracteres
- **Overlap:** 100 caracteres (~20%)
- **Separadores:** `\n\n` → `\n` → `. ` → ` `

**Justificativa:** O PDF de teoria musical, após extração, perde a hierarquia visual (títulos viram texto plano), inviabilizando chunking por estrutura de documento. O splitter recursivo respeita fronteiras naturais de parágrafo antes de recorrer a cortes arbitrários. O overlap de 20% garante que conceitos que se estendem por múltiplos parágrafos não sejam perdidos nas fronteiras entre chunks. O tamanho de 512 caracteres é suficiente para conter uma definição completa de um conceito musical sem trazer contexto irrelevante de outras seções.

### Modelo de Embeddings
**`intfloat/multilingual-e5-large`** via `sentence-transformers`

**Justificativa:** O domínio envolve texto técnico em inglês (PDF fonte) com consultas do usuário em português. O modelo multilingual-e5-large foi treinado especificamente para retrieval semântico em múltiplos idiomas, permitindo que perguntas em português recuperem corretamente chunks em inglês. Com 560M de parâmetros, oferece qualidade superior a modelos menores sem exigir GPU dedicada.

### Banco Vetorial
**ChromaDB** com persistência em disco.

**Justificativa:** Solução embarcada sem necessidade de servidor separado, adequada para o escopo do projeto. Usa similaridade por cosseno com índice HNSW internamente — eficiente para coleções de médio porte sem sacrificar precisão.

### Algoritmo de Busca
**Similaridade por cosseno** com `top_k=5`.

**Justificativa:** Embeddings de texto são vetores de alta dimensão. A distância por cosseno mede o ângulo entre vetores, ignorando a magnitude — o que é mais adequado para semântica do que distância euclidiana, pois dois textos com o mesmo significado devem ter vetores apontando na mesma direção independentemente do tamanho.

### Modelo de Linguagem
**LLaMA 3 (8B)** via Ollama (self-hosted)

**Justificativa:** Modelo open-source com bom desempenho em tarefas de Q&A em múltiplos idiomas. Roda localmente via Ollama, garantindo privacidade dos dados e funcionamento sem dependência de APIs pagas. O tamanho 8B é o ponto de equilíbrio entre qualidade de resposta e viabilidade de execução em hardware comum.

---

## ⚙️ Instalação

### Pré-requisitos

- [uv](https://astral.sh/uv) — gerenciador de pacotes Python
- [Ollama](https://ollama.com/download) — para rodar o LLaMA 3 localmente

### 1. Instalar o uv (Windows PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Instalar o Ollama e baixar o modelo

Baixe e instale o Ollama em [ollama.com/download](https://ollama.com/download), depois:

```bash
ollama pull llama3
```

### 3. Clonar o repositório e instalar dependências

```bash
git clone https://github.com/<seu-usuario>/Micro-SaaS-com-RAG.git
cd Micro-SaaS-com-RAG
uv sync
```

---

## 🚀 Uso

### Passo 1 — Indexar os documentos (apenas na primeira vez)

Coloque os PDFs em `data/raw/` e execute:

```bash
uv run python main.py
```

Saída esperada:
```
Carregando os PDFs...
Processando: music_theory_complete.pdf
OK — 523847 caracteres extraídos
Criando os chunks...
Indexando no vectorstore...
512 chunks indexados
Pipeline finalizada!
```

### Passo 2 — Subir a interface

```bash
uv run streamlit run app.py
```

Acesse em: `http://localhost:8501`

### Exemplos de perguntas

- *O que é uma cadência plagal?*
- *Explique os modos gregos*
- *Quais são os tipos de intervalos musicais?*
- *Como funciona a harmonia funcional?*

---

## 📁 Estrutura do Projeto

```
Micro-SaaS-com-RAG/
├── data/
│   └── raw/                  # PDFs de entrada
├── src/
│   ├── config.py             # Parâmetros globais (chunk size, modelos, paths)
│   ├── ingestion.py          # Extração de texto dos PDFs
│   ├── chunking.py           # Divisão em chunks com overlap
│   ├── embeddings.py         # Geração de embeddings e indexação no ChromaDB
│   ├── llm.py                # Geração de resposta com LLaMA 3 via Ollama
│   └── pipeline.py           # Orquestração do pipeline de indexação
├── vectorstore/              # Banco vetorial persistido (ChromaDB)
├── app.py                    # Interface Streamlit
├── main.py                   # Script de indexação
└── pyproject.toml            # Dependências do projeto
```

---

## 📊 Avaliação

A avaliação foi realizada com o framework **RAGAS** (*Retrieval-Augmented Generation Assessment*), usando o próprio **LLaMA 3** como juiz e o **multilingual-e5-large** como modelo de embedding — os mesmos utilizados no pipeline, garantindo consistência metodológica. Foram avaliadas 7 perguntas sobre o conteúdo do livro, com ground truth baseado exclusivamente no texto do PDF.

### Métricas avaliadas

| Métrica | O que mede | Resultado |
|---|---|---|
| **Faithfulness** | A resposta é fiel ao contexto recuperado? (baixo = alucinação) | **0.917** |
| **Answer Relevancy** | A resposta endereça a pergunta feita? (baixo = resposta tangencial) | **0.957** |
| **Context Precision** | Os chunks recuperados são relevantes? (baixo = ruído no retriever) | **0.885** |
| **Context Recall** | O retriever encontrou tudo que precisava? (baixo = chunks perdidos) | **0.952** |

### Resultados por pergunta

| Pergunta | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| What is a perfect interval? | 1.000 | 0.947 | 0.806 | 0.667 |
| What is a diatonic triad? | 1.000 | 0.903 | 0.806 | 1.000 |
| What is binary form in music? | 0.833 | 0.996 | 1.000 | 1.000 |
| What is a suspension in harmony? | 1.000 | 0.981 | 1.000 | 1.000 |
| What is music notation? | 1.000 | 0.983 | 1.000 | 1.000 |
| What are the species of counterpoint? | 0.750 | 0.965 | 0.583 | 1.000 |
| What is inflection in music intervals? | 0.833 | 0.927 | 1.000 | 1.000 |

### Análise de casos de falha

Dois casos apresentaram degradação de métricas:

**"What is a perfect interval?"** — context_recall de 0.667, o mais baixo observado. O retriever não trouxe todos os chunks necessários para cobrir completamente o ground truth. Hipótese: o conteúdo sobre intervalos perfeitos está distribuído em múltiplas páginas do livro, e o chunking por caracteres fragmentou as definições entre chunks não consecutivos.

**"What are the species of counterpoint?"** — context_precision de 0.583, o mais baixo observado. O retriever trouxe chunks com ruído — trechos relacionados a contraponto mas não diretamente relevantes para a pergunta. Hipótese: o conteúdo de contraponto por espécies é técnico e denso, com terminologia compartilhada entre seções distintas, o que confunde a busca por similaridade semântica.

### Metodologia e limitações da avaliação

A avaliação usou **LLM-as-a-judge** com o mesmo modelo que gera as respostas (LLaMA 3), Os resultados são consistentes com avaliação manual realizada durante o desenvolvimento, o que sugere que o viés não foi determinante. O resultado completo está disponível no doc `evaluation/ragas_resultados.csv`.

---

## ⚠️ Limitações

- O sistema responde apenas com base nos documentos indexados — perguntas fora do escopo do material resultam em "não encontrei nos materiais"
- PDFs escaneados (imagens) não são suportados sem OCR adicional
- O tempo de resposta depende do hardware disponível — em CPUs sem GPU, o LLaMA 3 pode levar 10–30 segundos por resposta
- Não há re-ranking dos chunks recuperados — uma melhoria futura seria adicionar um CrossEncoder para refinar a ordenação dos resultados

---

## 👥 Autores

Desenvolvido como projeto acadêmico — Universidade do Estado do Amazonas (UEA), Escola Superior de Tecnologia.

- Matheus Takashi Maruoka Vieira
- Rubens Takashi Maruoka Vieira
- Vinícius Castro Coutinho
