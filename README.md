# I2A2
Curso de agentes inteligentes de IA
Passo 1:
As Dependências do Projeto
	streamlit
	pandas
	matplotlib
	seaborn
	numpy
	langchain
	langchain-google-genai
	langchain-chroma
	tabulate
	python-dotenv

Passo 2: Execute a Aplicação no seu Computador
Abra o Terminal:
No Linux, procure por "Terminal".
Instale o Python: Se ainda não o tiver, baixe e instale o Python a partir de python.org.
	Bash -->
		sudo apt update
		sudo apt install python

	

Passo 3: Crie uma Pasta para o Projeto: Navegue pelo terminal até onde deseja salvar o projeto e crie uma pasta.
	Bash -->
		mkdir meu_agente_eda
		cd meu_agente_eda

Passo 4: Crie um Ambiente Virtual (Recomendado): Isso isola as bibliotecas do seu projeto.
	Bash -->
		python -m venv venv
		ou no meu caso
			python3 -m venv venv

Para ativá-lo:
	Bash -->
		source venv/bin/activate


Passo 5: Instale as Bibliotecas: Com os arquivos app.py e requirements.txt salvos na pasta, execute:
	Bash -->
		pip install -r requirements.txt

Passo 6: Execute o Agente: Agora, basta rodar o seguinte comando:
	Bash -->
		streamlit run app.py
