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

question = 'How do I plan my therapy?'
local_env = Environment()

### PIPELINE

## INDEXING: LOAD
loader = TextLoader("./FAQ.txt")
docs = loader.load()


## INDEXING: SPLIT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=20, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
# print(len(all_splits))

## INDEXING: STORE
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_version="gpt-3.5-turbo-0125", openai_api_key=local_env.OPENAI_API_KEY))


## RETRIEVAL
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke(question)

print(question)
print('-----------------')

for page_content in retrieved_docs:
    print(page_content)

print('-----------------')
## GENERATION
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=local_env.OPENAI_API_KEY)

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
print(example_messages[0].content)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)