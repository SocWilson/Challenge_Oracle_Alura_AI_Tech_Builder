"""
Tutor Académico Virtual — Challenge Alura Agente
==================================================

O que este programa faz:
1. Lê dois PDFs fixos da pasta data/ (o material do curso e a calendarização).
2. Divide o texto em pedaços menores (chunks) e transforma-os em vetores
   (embeddings) usando o Gemini.
3. Guarda esses vetores num índice FAISS, para conseguir encontrar
   rapidamente os trechos mais parecidos com a pergunta do aluno.
4. Usa um modelo da Groq para gerar a resposta final, com base apenas
   nesses trechos — nunca inventando informação.
5. Mostra tudo isto numa interface de chat, feita com Streamlit.

Porque dois provedores (Gemini + Groq)?
Cada provedor tem um limite gratuito de utilização (free tier). Ao usar o
Gemini só para os embeddings e a Groq só para gerar as respostas, nenhum dos
dois esgota o limite sozinho.
"""

import re

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------------------------------
# Configuração da página
# --------------------------------------------------------------------------

st.set_page_config(page_title="Tutor Académico Virtual", page_icon="🎓")

# Os dois PDFs que o tutor conhece. São fixos porque este projeto é um tutor
# especializado NESTE curso — não um leitor genérico de qualquer documento.
CAMINHO_MATERIAL_CURSO = "data/Curso_IA_2.pdf"
CAMINHO_CALENDARIZACAO = "data/calendarizacao_curso.pdf"


# --------------------------------------------------------------------------
# Passo 1 — Ler e limpar os PDFs
# --------------------------------------------------------------------------

def limpar_texto(texto: str) -> str:
    """Remove espaços/quebras de linha repetidos que os PDFs costumam deixar."""
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def carregar_documentos() -> list:
    """Lê os dois PDFs e devolve a lista de páginas (uma página = um Document)."""
    loader_curso = PyPDFLoader(CAMINHO_MATERIAL_CURSO)
    loader_calendario = PyPDFLoader(CAMINHO_CALENDARIZACAO)

    documentos = loader_curso.load() + loader_calendario.load()

    for doc in documentos:
        doc.page_content = limpar_texto(doc.page_content)

    return documentos


# --------------------------------------------------------------------------
# Passo 2 — Dividir em chunks e criar o índice vetorial (FAISS)
# --------------------------------------------------------------------------
#
# @st.cache_resource faz com que esta função só seja executada UMA VEZ,
# mesmo que o Streamlit reexecute o resto do script a cada interação do
# utilizador (é assim que o Streamlit funciona). Sem este cache, cada
# pergunta no chat reprocessaria os PDFs e chamaria a API de embeddings de
# novo — lento e desperdiça o limite gratuito da API.

@st.cache_resource(show_spinner="A preparar o material do curso...")
def montar_indice():
    documentos = carregar_documentos()

    # chunk_size=1000: pedaços de ate 1000 caracteres, um tamanho intermedio
    # que cobre um paragrafo inteiro sem ficar grande demais para o modelo.
    # chunk_overlap=150: sobreposicao entre pedacos vizinhos, para nao cortar
    # uma frase importante bem no meio, entre um chunk e o seguinte.
    divisor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = divisor.split_documents(documentos)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )

    indice = FAISS.from_documents(chunks, embeddings)
    return indice


# --------------------------------------------------------------------------
# Passo 3 — Montar o LLM (Groq) e o prompt do tutor
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def montar_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,  # 0 = respostas mais consistentes, menos "criativas"
        api_key=st.secrets["GROQ_API_KEY"],
    )


PROMPT = ChatPromptTemplate.from_template("""
Você é um Tutor Académico especializado em Python, Pandas, LLMs e LangChain.

A sua função é ajudar o estudante a aprender e a tirar dúvidas com base
EXCLUSIVAMENTE no conteúdo fornecido no contexto abaixo.

Regras:
1. Não invente informações que não estejam no contexto.
2. Responda em português.
3. Quando a pergunta envolver calendarização, prazos, aulas ou horários,
   utilize as informações do documento de calendarização.
4. Quando a pergunta for técnica (Python, Pandas, LLMs, LangChain), utilize
   o documento do material do curso.
5. Explique os conceitos de forma didática e clara, com exemplos de código
   quando fizer sentido.
6. Se a resposta não estiver disponível no contexto, diga claramente:
   "Não encontrei essa informação na documentação fornecida."
7. Quando possível, indique a página onde encontrou a informação.

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:
""")


# --------------------------------------------------------------------------
# Passo 4 — Função que responde a uma pergunta (busca + prompt + LLM)
# --------------------------------------------------------------------------

def extrair_texto(resposta) -> str:
    """A Groq pode devolver o conteúdo como texto simples ou como uma lista
    de blocos — esta função trata os dois casos e devolve sempre uma string."""
    conteudo = resposta.content

    if isinstance(conteudo, str):
        return conteudo

    if isinstance(conteudo, list):
        textos = [
            bloco.get("text", "")
            for bloco in conteudo
            if isinstance(bloco, dict) and bloco.get("type") == "text"
        ]
        return "\n".join(textos)

    return str(conteudo)


def perguntar(pergunta: str, indice, llm) -> str:
    """Recebe a pergunta do aluno e devolve a resposta do tutor."""

    # k=4: buscamos os 4 trechos mais parecidos com a pergunta. E' um
    # numero pequeno o suficiente para nao sobrecarregar o prompt, mas
    # cobre bem tanto duvidas tecnicas quanto de calendarizacao.
    retriever = indice.as_retriever(search_kwargs={"k": 4})
    documentos_relevantes = retriever.invoke(pergunta)

    contexto = "\n\n".join(
        f"[Página {doc.metadata.get('page_label', 'N/A')}]\n{doc.page_content}"
        for doc in documentos_relevantes
    )

    mensagens = PROMPT.invoke({"context": contexto, "question": pergunta})
    resposta = llm.invoke(mensagens)

    return extrair_texto(resposta)


# --------------------------------------------------------------------------
# Interface do Streamlit
# --------------------------------------------------------------------------

st.title("🎓 Tutor Académico Virtual")
st.caption("Curso Iniciante em Agentes de IA com Python, Pandas e LangChain")

# Verificação simples: sem as chaves configuradas, mostra um aviso claro
# em vez de deixar o programa falhar com um erro confuso.
if "GOOGLE_API_KEY" not in st.secrets or "GROQ_API_KEY" not in st.secrets:
    st.error(
        "Faltam chaves de API. Configure GOOGLE_API_KEY e GROQ_API_KEY em "
        ".streamlit/secrets.toml (local) ou em Settings → Secrets (Streamlit Cloud)."
    )
    st.stop()

indice = montar_indice()
llm = montar_llm()

# Guarda o histórico da conversa entre interações (senão o Streamlit
# "esqueceria" as mensagens anteriores a cada rerun).
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for autor, texto in st.session_state.mensagens:
    with st.chat_message(autor):
        st.markdown(texto)

pergunta = st.chat_input("Pergunte sobre Python, Pandas, LLMs, LangChain ou o curso...")

if pergunta:
    st.session_state.mensagens.append(("user", pergunta))
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("A consultar o material..."):
            resposta = perguntar(pergunta, indice, llm)
            st.markdown(resposta)

    st.session_state.mensagens.append(("assistant", resposta))
    