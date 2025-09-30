# app.py
# Importando as bibliotecas necessárias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import re
import numpy as np

# Configuração para evitar problemas de GUI em ambientes de servidor
import matplotlib
matplotlib.use('Agg')

# Importações do LangChain e Google
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# Importações para o RAG com ChromaDB
from langchain_chroma import Chroma
from langchain.schema import Document

# --------------------------------------------------------------------------------
# Configuração e Constantes
# --------------------------------------------------------------------------------
# Diretórios para salvar arquivos temporários e o banco de dados vetorial
TEMP_DATA_FILE = "temp_df.pkl"
CHROMA_DB_DIR = "chroma_db_eda"
PLOTS_DIR = "plots"

# Cria o diretório de plots se não existir
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# --- NOVA LÓGICA PARA CARREGAR A API KEY DE FORMA SEGURA ---
try:
    # Tenta carregar a chave do Streamlit Secrets (para o ambiente de produção/nuvem)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    # Se não encontrar, avisa o usuário para configurar localmente
    st.warning("Chave de API do Google não encontrada. Configure-a no arquivo .streamlit/secrets.toml para uso local.")
    GOOGLE_API_KEY = None
# --- FIM DA NOVA LÓGICA ---

# --------------------------------------------------------------------------------
# Funções Auxiliares
# --------------------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """Substitui caracteres inválidos em nomes de arquivos por underscores."""
    s = re.sub(r'[\\/*?:"<>|()]', "_", name)
    return "_".join(s.split())

# --------------------------------------------------------------------------------
# LÓGICA DO RAG (MEMÓRIA DE LONGO PRAZO)
# --------------------------------------------------------------------------------
@st.cache_resource
def initialize_rag():
    """Inicializa o banco de dados vetorial ChromaDB e os embeddings."""
    if not GOOGLE_API_KEY:
        return None
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings_model)
        return vector_store
    except Exception as e:
        st.error(f"Erro ao inicializar o RAG: {e}")
        return None

def save_analysis_to_rag(vector_store, question: str, answer: str):
    """Salva uma análise (pergunta e resposta) no banco de dados vetorial."""
    if vector_store:
        try:
            document = Document(page_content=f"Pergunta do Usuário: {question}\n\nConclusão do Agente: {answer}", metadata={"source": "AgentAnalysis"})
            vector_store.add_documents([document])
            print(f"INFO: Análise salva no RAG com sucesso.")
        except Exception as e:
            print(f"ERRO: Falha ao salvar análise no RAG: {e}")

# --------------------------------------------------------------------------------
# FERRAMENTAS DO AGENTE
# --------------------------------------------------------------------------------
@tool
def search_past_analyses(query: str) -> str:
    # ... (código das ferramentas permanece o mesmo)
    if "vector_store" in st.session_state and st.session_state.vector_store:
        results = st.session_state.vector_store.similarity_search(query, k=3)
        if not results:
            return "Nenhuma análise anterior relevante foi encontrada."
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        return f"Análises passadas encontradas que podem ser relevantes:\n\n{context}"
    return "O banco de dados de análises passadas não está disponível."
@tool
def get_dataframe_info(query: str) -> str:
    # ... (código das ferramentas permanece o mesmo)
    try:
        df = pd.read_pickle(TEMP_DATA_FILE)
    except FileNotFoundError:
        return "Erro: Nenhum arquivo de dados está carregado. Peça ao usuário para carregar um arquivo CSV."
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    return f"Resumo do Arquivo Atual:\n{info_str}\n\nEstatísticas Descritivas:\n{df.describe().to_string()}"
@tool
def get_all_variability(query: str) -> str:
    # ... (código das ferramentas permanece o mesmo)
    try:
        df = pd.read_pickle(TEMP_DATA_FILE)
    except FileNotFoundError:
        return "Erro: Nenhum arquivo de dados está carregado."
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return "Nenhuma coluna numérica encontrada no arquivo para calcular a variabilidade."
    variability = pd.DataFrame({'Desvio Padrão': numeric_df.std(), 'Variância': numeric_df.var()}).reset_index().rename(columns={'index': 'Coluna'})
    return f"A variabilidade para as colunas numéricas é a seguinte:\n{variability.to_markdown(index=False)}"
@tool
def plot_distribution(column_name: str) -> str:
    # ... (código das ferramentas permanece o mesmo)
    try:
        df = pd.read_pickle(TEMP_DATA_FILE)
    except FileNotFoundError:
        return "Erro: Nenhum arquivo de dados está carregado."
    if column_name not in df.columns:
        return f"Erro: A coluna '{column_name}' não foi encontrada. Colunas disponíveis: {', '.join(df.columns)}"
    fig, ax = plt.subplots(figsize=(10, 6))
    safe_column_name = sanitize_filename(column_name)
    if pd.api.types.is_numeric_dtype(df[column_name]):
        sns.histplot(df[column_name], kde=True, ax=ax)
        ax.set_title(f'Distribuição de {column_name}')
        file_name = f"dist_hist_{safe_column_name}.png"
    else:
        top_n = 20
        order = df[column_name].value_counts().nlargest(top_n).index
        sns.countplot(y=df[column_name], ax=ax, order=order)
        ax.set_title(f'Contagem para {column_name} (Top {top_n})')
        file_name = f"dist_count_{safe_column_name}.png"
    plt.tight_layout()
    file_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(file_path)
    plt.close(fig)
    return f"Gráfico de distribuição salvo como '{file_path}'."
@tool
def plot_scatterplot(column_x: str, column_y: str) -> str:
    # ... (código das ferramentas permanece o mesmo)
    try:
        df = pd.read_pickle(TEMP_DATA_FILE)
    except FileNotFoundError:
        return "Erro: Nenhum arquivo de dados está carregado."
    if column_x not in df.columns or column_y not in df.columns:
        return f"Erro: Uma ou ambas as colunas não foram encontradas. Colunas disponíveis: {', '.join(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[column_x]) or not pd.api.types.is_numeric_dtype(df[column_y]):
        return f"Erro: Ambas as colunas devem ser numéricas para um gráfico de dispersão."
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=column_x, y=column_y, ax=ax)
    ax.set_title(f'Relação entre {column_x} e {column_y}')
    plt.tight_layout()
    safe_x = sanitize_filename(column_x)
    safe_y = sanitize_filename(column_y)
    file_name = f"scatter_{safe_x}_vs_{safe_y}.png"
    file_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(file_path)
    plt.close(fig)
    return f"Gráfico de dispersão salvo como '{file_path}'."
@tool
def plot_boxplot(numeric_column: str, categorical_column: str) -> str:
    # ... (código das ferramentas permanece o mesmo)
    try:
        df = pd.read_pickle(TEMP_DATA_FILE)
    except FileNotFoundError:
        return "Erro: Nenhum arquivo de dados está carregado."
    if numeric_column not in df.columns or categorical_column not in df.columns:
        return f"Erro: Uma ou ambas as colunas não foram encontradas. Colunas disponíveis: {', '.join(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        return f"Erro: A coluna '{numeric_column}' deve ser numérica para um boxplot."
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df, x=numeric_column, y=categorical_column, ax=ax)
    ax.set_title(f'Distribuição de {numeric_column} por {categorical_column}')
    plt.tight_layout()
    safe_num = sanitize_filename(numeric_column)
    safe_cat = sanitize_filename(categorical_column)
    file_name = f"boxplot_{safe_num}_by_{safe_cat}.png"
    file_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(file_path)
    plt.close(fig)
    return f"Boxplot salvo como '{file_path}'."
@tool
def plot_correlation_heatmap() -> str:
    # ... (código das ferramentas permanece o mesmo)
    try:
        df = pd.read_pickle(TEMP_DATA_FILE)
    except FileNotFoundError:
        return "Erro: Nenhum arquivo de dados está carregado."
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return "Erro: São necessárias pelo menos duas colunas numéricas para gerar um mapa de calor de correlação."
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Mapa de Calor de Correlação das Variáveis Numéricas')
    plt.tight_layout()
    file_name = "correlation_heatmap.png"
    file_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(file_path)
    plt.close(fig)
    return f"Mapa de calor de correlação salvo como '{file_path}'."
@tool
def plot_lineplot(time_column: str, value_column: str) -> str:
    # ... (código das ferramentas permanece o mesmo)
    try:
        df = pd.read_pickle(TEMP_DATA_FILE)
    except FileNotFoundError:
        return "Erro: Nenhum arquivo de dados está carregado."
    if time_column not in df.columns or value_column not in df.columns:
        return f"Erro: Uma ou ambas as colunas não foram encontradas. Colunas disponíveis: {', '.join(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"Erro: A coluna de valor '{value_column}' deve ser numérica."
    sample_size = 50000
    plot_df = df
    title_note = ""
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=42).sort_values(by=time_column)
        title_note = f"\n(usando amostra de {sample_size} pontos)"
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=plot_df, x=time_column, y=value_column, ax=ax)
    ax.set_title(f'Tendência de {value_column} ao longo de {time_column}{title_note}')
    plt.tight_layout()
    safe_time = sanitize_filename(time_column)
    safe_value = sanitize_filename(value_column)
    file_name = f"lineplot_{safe_value}_over_{safe_time}.png"
    file_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(file_path)
    plt.close(fig)
    return f"Gráfico de linhas salvo como '{file_path}'."

# --------------------------------------------------------------------------------
# LÓGICA PRINCIPAL DA APLICAÇÃO STREAMLIT
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Gemini")

with st.sidebar:
    st.header("1. Carregue seus Dados")
    # REMOVIDO o campo de texto para a chave. A aplicação agora usa a chave dos Secrets.
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.to_pickle(TEMP_DATA_FILE)
            st.success(f"Arquivo '{uploaded_file.name}' carregado!")
            st.session_state.messages = [{"role": "assistant", "content": f"Olá! Analisando o arquivo '{uploaded_file.name}'. Como posso ajudar?"}]
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            if os.path.exists(TEMP_DATA_FILE): os.remove(TEMP_DATA_FILE)

    if os.path.exists(TEMP_DATA_FILE):
        st.header("2. Exemplos de Perguntas")
        st.info(
            "**Básicas:**\n"
            "- Descreva os dados.\n"
            "- Qual a distribuição da coluna 'Amount'?\n\n"
            "**Gráficos:**\n"
            "- Qual a relação entre 'V10' e 'V12'?\n"
            "- Mostre o mapa de calor de correlação."
        )
    elif not GOOGLE_API_KEY:
         st.error("A chave de API não foi configurada. A aplicação não pode funcionar.")
    else:
        st.warning("Carregue um arquivo CSV para começar.")

# --- Inicialização do Agente e da Memória ---
available_tools = [
    get_dataframe_info, get_all_variability, plot_distribution,
    search_past_analyses, plot_scatterplot, plot_boxplot,
    plot_correlation_heatmap, plot_lineplot
]
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um cientista de dados assistente... (prompt completo omitido por brevidade)"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state and GOOGLE_API_KEY:
    try:
        # CORRIGIDO: Usando o nome de modelo estável 'gemini-pro'
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4, google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
        agent = create_tool_calling_agent(llm, available_tools, prompt_template)
        st.session_state.agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
        st.session_state.vector_store = initialize_rag()
    except Exception as e:
        st.error(f"Erro ao inicializar o agente do LangChain: {e}")
        if "agent_executor" in st.session_state: del st.session_state.agent_executor

# --- Área de Chat Principal ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path)

if os.path.exists(TEMP_DATA_FILE) and "agent_executor" in st.session_state:
    if prompt := st.chat_input("Faça sua pergunta sobre o arquivo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando..."):
                try:
                    chat_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages[:-1]]
                    response = st.session_state.agent_executor.invoke({"input": prompt, "chat_history": chat_history})
                    agent_response_text = response.get('output', 'Não foi possível gerar uma resposta.')
                    assistant_message = {"role": "assistant", "content": agent_response_text, "images": []}
                    if "intermediate_steps" in response:
                        for _, observation in response["intermediate_steps"]:
                            filenames = re.findall(r"'(.*?\.png)'", str(observation))
                            for filename in filenames:
                                if os.path.exists(filename):
                                    assistant_message["images"].append(filename)
                    st.markdown(agent_response_text)
                    for img_path in assistant_message["images"]:
                        st.image(img_path)
                    st.session_state.messages.append(assistant_message)
                    if len(agent_response_text) > 100:
                        save_analysis_to_rag(st.session_state.vector_store, prompt, agent_response_text)
                except Exception as e:
                    error_message = f"Ocorreu um erro inesperado: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
elif not GOOGLE_API_KEY:
     st.info("A configuração da API Key não foi encontrada. A aplicação está inativa.")
else:
    st.info("Bem-vindo! Por favor, carregue um arquivo CSV para iniciar a análise.")