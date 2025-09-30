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
def initialize_rag(_api_key):
    """Inicializa o banco de dados vetorial ChromaDB e os embeddings."""
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_api_key)
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
    """
    Pesquisa em análises e conclusões passadas para responder a uma pergunta.
    Use esta ferramenta PRIMEIRO se a pergunta do usuário for sobre 'conclusões', 'resumos anteriores' ou 'análises já feitas'.
    """
    if "vector_store" in st.session_state and st.session_state.vector_store:
        results = st.session_state.vector_store.similarity_search(query, k=3)
        if not results:
            return "Nenhuma análise anterior relevante foi encontrada."
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        return f"Análises passadas encontradas que podem ser relevantes:\n\n{context}"
    return "O banco de dados de análises passadas não está disponível."

@tool
def get_dataframe_info(query: str) -> str:
    """
    Retorna um resumo completo do DataFrame ATUAL, incluindo colunas, tipos de dados, valores ausentes e estatísticas descritivas.
    Esta deve ser a PRIMEIRA ferramenta a ser usada para entender um novo conjunto de dados.
    """
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
    """
    Calcula o desvio padrão e a variância para TODAS as colunas numéricas do arquivo ATUAL.
    Use para perguntas sobre 'variabilidade', 'dispersão', 'desvio padrão' ou 'variância' dos dados.
    """
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
    """
    Cria e salva um gráfico de distribuição (histograma ou contagem) para uma ÚNICA coluna específica do arquivo ATUAL.
    """
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
    """
    Cria e salva um gráfico de dispersão (scatterplot) para visualizar a relação entre DUAS colunas NUMÉRICAS.
    """
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
    """
    Cria e salva um boxplot para comparar a distribuição de uma coluna NUMÉRICA através das categorias de uma coluna CATEGÓRICA.
    """
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
    """
    Calcula e salva um mapa de calor (heatmap) da matriz de correlação para TODAS as colunas numéricas do arquivo. Não precisa de argumentos.
    """
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

# --- FERRAMENTA ATUALIZADA ---
@tool
def plot_lineplot(time_column: str, value_column: str) -> str:
    """
    Cria e salva um gráfico de linhas para mostrar a tendência de uma coluna NUMÉRICA ao longo de uma coluna de TEMPO ou SEQUENCIAL.
    Se o dataset for muito grande, uma amostra dos dados será usada para gerar o gráfico mais rapidamente.
    """
    try:
        df = pd.read_pickle(TEMP_DATA_FILE)
    except FileNotFoundError:
        return "Erro: Nenhum arquivo de dados está carregado."
    
    if time_column not in df.columns or value_column not in df.columns:
        return f"Erro: Uma ou ambas as colunas não foram encontradas. Colunas disponíveis: {', '.join(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"Erro: A coluna de valor '{value_column}' deve ser numérica."

    # --- LÓGICA DE OTIMIZAÇÃO ---
    sample_size = 50000
    plot_df = df
    title_note = ""
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=42).sort_values(by=time_column)
        title_note = f"\n(usando amostra de {sample_size} pontos)"
    # --- FIM DA LÓGICA ---

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
# --- FIM DA FERRAMENTA ATUALIZADA ---

# --------------------------------------------------------------------------------
# LÓGICA PRINCIPAL DA APLICAÇÃO STREAMLIT
# --------------------------------------------------------------------------------

st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Gemini")

with st.sidebar:
    st.header("1. Configuração")
    api_key_input = st.text_input("Google API Key ou a que consta no documento enviado por email", type="password", help="Insira sua chave de API do Google Gemini.")
    if api_key_input:
        st.session_state.GOOGLE_API_KEY = api_key_input

    st.header("2. Carregue seus Dados")
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
        st.header("3. Exemplos de Perguntas")
        st.info(
            "**Básicas:**\n"
            "- Descreva os dados.\n"
            "- Qual a distribuição da coluna 'Amount'?\n"
            "- Qual a variabilidade dos dados?\n\n"
            "**Novos Gráficos:**\n"
            "- Qual a relação entre 'V10' e 'V12'?\n"
            "- Compare 'Amount' por 'Class' com um boxplot.\n"
            "- Mostre o mapa de calor de correlação.\n"
            "- Mostre a tendência de 'Amount' ao longo de 'Time'."
        )
    else:
        st.warning("Carregue um arquivo CSV e insira sua API Key para começar.")

# --- Inicialização do Agente e da Memória ---

# --- Ferramentas Disponíveis para o Agente Utilizar ---
available_tools = [
    get_dataframe_info, 
    get_all_variability, 
    plot_distribution, 
    search_past_analyses,
    plot_scatterplot,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_lineplot
]

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """Você é um cientista de dados assistente. Sua principal tarefa é analisar o arquivo CSV carregado na sessão ATUAL.

        **Seu Fluxo de Trabalho e Ferramentas:**
        1.  **Entendimento Inicial:** Ao analisar um novo arquivo, SEMPRE comece usando `get_dataframe_info`.
        2.  **Memória:** Se a pergunta for sobre 'conclusões' ou 'análises passadas', use `search_past_analyses`.
        3.  **Análise Univariada (1 variável):**
            - Para um resumo de variabilidade (desvio padrão, variância), use `get_all_variability`.
            - Para visualizar a distribuição de UMA coluna, use `plot_distribution`.
        4.  **Análise Bivariada (2 variáveis):**
            - Para ver a relação entre DUAS colunas NUMÉRICAS, use `plot_scatterplot`.
            - Para comparar uma coluna NUMÉRICA entre as categorias de uma coluna CATEGÓRICA, use `plot_boxplot`.
            - Para ver a tendência de um valor NUMÉRICO ao longo do TEMPO/sequência, use `plot_lineplot`.
        5.  **Análise Multivariada (+2 variáveis):**
            - Para visualizar a correlação entre TODAS as colunas numéricas, use `plot_correlation_heatmap`.
        
        Responda em português, de forma clara. Ao gerar um gráfico, avise o usuário e informe o nome do arquivo. Forneça um parágrafo de 'Conclusão' após análises complexas."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state and "GOOGLE_API_KEY" in st.session_state:
    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.4, google_api_key=st.session_state.GOOGLE_API_KEY, convert_system_message_to_human=True)
        agent = create_tool_calling_agent(llm, available_tools, prompt_template)
        st.session_state.agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
        st.session_state.vector_store = initialize_rag(st.session_state.GOOGLE_API_KEY)
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
            with st.spinner("O agente está pensando e analisando, dependendo do tipo e quantidade de dados analisados pode demorar até alguns minutos..."):
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
else:
    st.info("Bem-vindo! Por favor, carregue um arquivo CSV e insira sua API Key na barra lateral para iniciar a análise.")