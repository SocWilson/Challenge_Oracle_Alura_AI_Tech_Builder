Challenge Alura Agente — Tutor Académico Virtual 🎓

Agente de IA que atua como tutor de um curso iniciante de Agentes de IA com Python, Pandas e LangChain. Os alunos conversam em linguagem natural para tirar dúvidas técnicas (Python, Pandas, LLMs, LangChain) e administrativas (calendarização, docentes, prazos, suporte) — sempre com base no material oficial do curso, sem inventar informação.

🔗 Demo online: (será adicionado após o deploy no Streamlit Community Cloud)

Índice
Objetivo do MVP
Arquitetura
Estrutura do repositório
O material de referência
Exemplos de perguntas e respostas
Como usar localmente
Deploy no Streamlit Community Cloud
Limitações conhecidas
Objetivo do MVP
Processar um documento — o agente lê dois PDFs fixos (o material do curso e a calendarização), extrai o texto e organiza-o para consulta.
Responder perguntas — um agente de IA busca a resposta nesse material e devolve-a em linguagem natural, clara e direta.
Rodar na nuvem — a aplicação é implantada no Streamlit Community Cloud, acessível publicamente.
Arquitetura

O projeto segue um pipeline de RAG (Retrieval-Augmented Generation):

┌──────────────┐     ┌───────────────┐     ┌─────────────────────┐     ┌───────────────┐
│  2 PDFs      │ --> │   Limpeza +   │ --> │  Embeddings (Gemini) │ --> │  Índice FAISS  │
│  fixos       │     │   Chunking    │     │                      │     │                │
│ (data/)      │     │  (LangChain)  │     │                      │     │                │
└──────────────┘     └───────────────┘     └─────────────────────┘     └───────┬────────┘
                                                                                │
                                                                                ▼
┌──────────────┐   ┌────────────────┐   ┌────────────────┐   ┌───────────────────────┐
│  Resposta    │ <- │  LLM (Groq /   │ <- │  Prompt de     │ <-│  Retriever: busca os  │
│  do tutor    │   │  gpt-oss-20b)  │   │  Tutor Académico│  │  4 trechos mais         │
│              │   │                │   │  + contexto     │  │  parecidos com a       │
│              │   │                │   │                 │  │  pergunta              │
└──────────────┘   └────────────────┘   └────────────────┘   └───────────────────────┘

Componentes e por que foram escolhidos:

Camada	Tecnologia	Motivo
Orquestração	LangChain	Padroniza carregamento de PDFs, chunking e prompts
Carregamento	PyPDFLoader	Extrai o texto de cada página do PDF, com metadados (número de página)
Divisão em chunks	RecursiveCharacterTextSplitter (chunk_size=1000, overlap=150)	Pedaços grandes o suficiente para cobrir um parágrafo, com sobreposição para não cortar uma ideia ao meio
Embeddings	Google Gemini (gemini-embedding-001)	Transforma cada chunk em um vetor numérico, para permitir busca por significado
Banco vetorial	FAISS	Guarda os vetores localmente e permite busca rápida por similaridade
LLM de geração	Groq (openai/gpt-oss-20b)	Gera a resposta final; gratuito e rápido
Interface	Streamlit	Interface de chat simples, com deploy gratuito na nuvem

Por que dois provedores de IA (Gemini + Groq)? Cada provedor tem um limite gratuito de utilização (free tier). Ao usar o Gemini só para os embeddings e a Groq só para gerar as respostas, nenhum dos dois esgota o limite sozinho — uma escolha de resiliência, não de complexidade desnecessária.

Desempenho: as etapas de carregar os PDFs, gerar embeddings e montar o índice FAISS só são executadas uma vez (@st.cache_resource), mesmo que o Streamlit reexecute o script a cada pergunta — evitando reprocessar os documentos e gastar o limite gratuito da API a cada interação.

Estrutura do repositório
challenge-alura-agente/
├── app.py                        # aplicação completa (carregamento, RAG, interface Streamlit)
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml               # chaves de API (NÃO vai para o GitHub)
└── data/
    ├── Curso_IA_2.pdf              # material de referência do curso
    ├── calendarizacao_curso.pdf    # calendarização das aulas e prazos
    └── gerar_material_curso.py     # script que gera o Curso_IA_2.pdf
O material de referência

O Curso_IA_2.pdf é a documentação oficial do curso, com:

Introdução ao Curso — formato (100% online, aulas ao vivo) e plataformas usadas
Lista de Docentes — um professor por módulo (Python, Pandas, LLMs, LangChain)
Parte 1 — Guia de Boas-Vindas e Primeiros Passos (acesso, pré-requisitos, instalação)
Parte 2 — Regulamento do Estudante (horários, canais de comunicação, política de entregas)
Parte 3 — Base de Conhecimento / FAQ técnico (Python, Pandas, LLMs, LangChain, com exemplos de código)
Parte 4 — Guia de Projetos e Exercícios Práticos
Parte 5 — Política de Suporte e Dúvidas

O calendarizacao_curso.pdf complementa com as datas exatas de cada módulo e os prazos de entrega dos trabalhos finais.

Exemplos de perguntas e respostas

P: O que é Python? R: Python é uma linguagem de programação de alto nível, de propósito geral, conhecida pela sintaxe simples e fácil de ler. É muito usada em automação, análise de dados, ciência de dados e inteligência artificial — e é a linguagem usada como base em todo este curso.

P: Quem é o professor do módulo de LangChain? R: O módulo 4 (LangChain) é lecionado pela Profa. Beatriz Lima Ferreira, desenvolvedora especializada em aplicações com modelos de linguagem, com experiência na construção de agentes e chatbots para produtos reais.

P: Qual o prazo de entrega do trabalho final do módulo de Pandas? R: O prazo de entrega do trabalho final do módulo de Pandas é 15/09/2026, que coincide com a data de início do módulo seguinte (LLMs).

P: Como funcionam as aulas, é presencial? R: Não, as aulas são totalmente online, transmitidas ao vivo através das plataformas Miits ou Zuum, conforme o módulo. As sessões ficam gravadas e disponíveis na plataforma para quem não conseguir assistir ao vivo.

P: Qual a distância entre o cesto de basquete e o chão? R: Não encontrei essa informação na documentação fornecida.

(a última pergunta demonstra que o tutor não inventa respostas fora do escopo do material)

Como usar localmente
1. Pré-requisitos
Python 3.10 ou superior
Uma chave de API gratuita da Groq
Uma chave de API gratuita do Google AI Studio (para os embeddings do Gemini)
2. Clonar o repositório e criar o ambiente virtual
bash
git clone https://github.com/SEU_USUARIO/challenge-alura-agente.git
cd challenge-alura-agente

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3. Instalar as dependências
bash
pip install -r requirements.txt
4. Configurar as chaves de API

Crie a pasta .streamlit e, dentro dela, o ficheiro secrets.toml:

toml
GOOGLE_API_KEY = "sua_chave_gemini_aqui"
GROQ_API_KEY = "sua_chave_groq_aqui"

Este ficheiro nunca é enviado ao GitHub (está no .gitignore).

5. Rodar a aplicação
bash
streamlit run app.py

Abre automaticamente em http://localhost:8501.

Deploy no Streamlit Community Cloud

(passo a passo a ser executado após validação — o código já está pronto para isto sem alterações)

Enviar o repositório para o GitHub (sem o .streamlit/secrets.toml, que fica de fora via .gitignore)
Em share.streamlit.io, conectar o repositório e apontar para app.py
Em Settings → Secrets, colar o mesmo conteúdo do secrets.toml local:
toml
   GOOGLE_API_KEY = "sua_chave_gemini_aqui"
   GROQ_API_KEY = "sua_chave_groq_aqui"
Deploy — a aplicação fica publicamente acessível
Limitações conhecidas
O tutor responde com base apenas nos dois PDFs processados — não tem acesso à internet nem a conhecimento externo ao contexto recuperado.
Os documentos são fixos (data/Curso_IA_2.pdf e data/calendarizacao_curso.pdf); para usar outro material, é necessário substituir os ficheiros e reiniciar a aplicação (o índice fica em cache).
Depende de dois provedores de IA externos (Google Gemini e Groq); se algum deles estiver indisponível ou sem chave configurada, a aplicação mostra um aviso claro em vez de falhar silenciosamente.