import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def load_document(file):
	name, extension = os.path.splitext(file)
	if extension == '.pdf':
		from langchain.document_loaders import PyPDFLoader
		print(f'loading pdf {file}')
		loader = PyPDFLoader(file)
	elif extension == '.docx':
		from langchain.document_loaders import Docx2txtLoader
		print(f'loading docx {file}')
		loader = Docx2txtLoader(file)
	elif extension == '.txt':
		from langchain.document_loaders import TextLoader
		loader = TextLoader(file)
	else:
		print('document format not supported')
	data = loader.load()
	return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
	from langchain.text_splitter import RecursiveCharacterTextSplitter
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	chunks = text_splitter.split_documents(data)
	return (chunks)

# class OpenAIEmbeddings(Chroma.EmbeddingFunction):

#     def __init__(self):
#         # Initialize the OpenAI embeddings API
#         pass

    # def __call__(self, input):
	# 	embeddinggs = OpenAIEmbeddings()
    #     return embeddinggs

# def create_embeddings(chunks):
#   	 embeddinggs = OpenAIEmbeddings() 
#     vector_store = Chroma.from_documents(chunks, embeddinggs)
#     return vector_store

def create_embeddings(chunks):
	embeddinggs = OpenAIEmbeddings()
	vector_store = Chroma.from_documents(chunks, embeddinggs)
	return vector_store

def ask_and_get_answer(vector_store, q, k=3):
	from langchain.chains import RetrievalQA
	from langchain.chat_models import ChatOpenAI

	llm = ChatOpenAI(
		model='gpt-3.5-turbo',
		temperature=1
	)
	retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
	chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
	answer = chain.run(q)
	return answer

def calculate_embedding_cost(texts):
	import tiktoken
	enc = tiktoken.encoding_for_model('text-embedding-ada-002')
	total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
	# print (f'Total Tokens: {total_tokens}')
	# print(f'Embedding Cost in USD: {total_tokens/1000 *0.0004:.6f}')
	return total_tokens, total_tokens/1000 *0.0004

def clear_history():
	if 'history' in st.session_state:
		del st.session_state['history']


if __name__ == "__main__":
	import os
	from dotenv import load_dotenv, find_dotenv
	load_dotenv(find_dotenv(), override=True)

	# st.image('./docschat_webapp/img.jpg')
	st.subheader('LLM Document Chat')
	with st.sidebar:
		api_key = st.text_input('OpenAI API Key: ', type='password')
		if api_key:
			os.environ['OPENAI_API_KEY'] = api_key
		
		uploaded_file = st.file_uploader('Upload a file: ', type=['pdf', 'docx', 'txt'])
		chunk_size = st.number_input('Chunk size: ', min_value=100, max_value=2048, value=512, on_change=clear_history)
		k = st.number_input('k:', min_value=1, max_value=20, value=3, on_change=clear_history)
		add_data = st.button('Add Data', on_click=clear_history)

		if uploaded_file and add_data:
			with st.spinner('Reading, Chunking and Embedding file..'):
				bytes_data = uploaded_file.read()
				file_name = os.path.join('./', uploaded_file.name)
			with open(file_name, 'wb') as f:
				f.write(bytes_data)
			
				data = load_document(file_name)
				chunks = chunk_data(data, chunk_size=chunk_size)
				st.write(f'chunk size: {chunk_size}, Chunks: {len(chunks)}')

				tokens, embedding_cost = calculate_embedding_cost(chunks)
				st.write(f'Embedding cose: ${embedding_cost:.4f}')

				
				vector_store = create_embeddings(chunks)
				st.session_state.vs = vector_store
				st.success('File uploaded, chunked and embedded succesfully.')

	q = st.text_input('Ask a Question about the content of the file:')
	if q:
		if 'vs' in st.session_state:
			vector_store = st.session_state.vs
			st.write(f'k: {k}')
			answer = ask_and_get_answer(vector_store, q, k)
			st.text_area('LLM Answer: ', value=answer)
			
			st.divider()
			if 'history' not in st.session_state:
				st.session_state.history = ''
			value = f'q: {q} \nA: {answer}'
			st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
			h = st.session_state.history
			st.text_area(label='Chat History', value=h, key='history', height=400)


