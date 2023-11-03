from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from fastapi import FastAPI
import dotenv

dotenv.load_dotenv()

app = FastAPI()
@app.get("/helloworld")
async def helloworld():
    return {"message": "Hello World"}

@app.post("/loadAndStore")
async def load_and_store(url : str = 'https://lilianweng.github.io/posts/2023-06-23-agent', chunk_size : int = 1024, chunk_overlap : int = 0, collection_name : str = 'temp1'):
    loader = WebBaseLoader(url) 
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap) 
    all_splits = text_splitter.split_documents(data) 
    Chroma.from_documents(collection_name=collection_name,documents=all_splits, embedding=OpenAIEmbeddings(),persist_directory='./chroma_db') 
    return {"Message": "Completed"}

@app.post("/peekDocument")
async def peek_document(collection_name : str = 'temp1'):
    vectorstore = Chroma(collection_name=collection_name,persist_directory='./chroma_db',embedding_function = OpenAIEmbeddings()) #Load collection from disk.
    return {"Top documnet in collection": vectorstore._collection.peek()} #Return top document in collection.

@app.post("/search")
async def search(query : str = 'What are the approaches to Task Decomposition?', collection_name : str = 'temp1'):
    vectorstore = Chroma(collection_name=collection_name,persist_directory='./chroma_db',embedding_function = OpenAIEmbeddings())
    result = vectorstore.similarity_search(query, k=1)
    return {"Result": result}

@app.post("/queryWithRetrieval")
async def query_with_retrieval(query : str = 'What are the approaches to Task Decomposition?', collection_name : str = 'temp1'):
    vectorstore = Chroma(collection_name=collection_name,persist_directory='./chroma_db',embedding_function = OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=vectorstore.as_retriever(search_kwargs={"k": 1}))
    result = qa_chain({"question": query})
    return {"result": result}

@app.post("/queryWithOutRetrieval")
async def query_without_retrieval(query : str = 'What are the approaches to Task Decomposition?', collection_name : str = 'temp1'):
    vectorstore = Chroma(collection_name=collection_name,persist_directory='./chroma_db',embedding_function = OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    result = llm.predict(query)
    return {"result": result}
