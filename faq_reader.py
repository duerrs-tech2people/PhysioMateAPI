from langchain_community.document_loaders import TextLoader
import bs4
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from environment import Environment
from langchain_openai import ChatOpenAI


### PIPELINE

## INDEXING: LOAD
loader = TextLoader("./FAQ.txt")
docs = loader.load()

# llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

local_env = Environment()


## INDEXING: SPLIT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
# print(len(all_splits), all_splits)

## INDEXING: STORE
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_version="gpt-3.5-turbo-0125", openai_api_key=local_env.OPENAI_API_KEY))


## RETRIEVAL
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke("Will there be a mobile app for abilitate?")
print(retrieved_docs)