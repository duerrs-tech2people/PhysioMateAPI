import time
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
from langchain_core.prompts import PromptTemplate

def answerFaqRelatedQuestion(question):
    local_env = Environment()
    timings = {}

    ### PIPELINE

    ## INDEXING: LOAD
    start_time = time.time()
    loader = TextLoader("./FAQ.txt")
    docs = loader.load()


    ## INDEXING: SPLIT
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=30, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)


    ## INDEXING: STORE
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_version="gpt-3.5-turbo-0125", openai_api_key=local_env.OPENAI_API_KEY))
    timings['indexing_time'] = time.time() - start_time

    ## RETRIEVAL
    start_time = time.time()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    timings['retrieval_time'] = time.time() - start_time

    ## GENERATION
    start_time = time.time()

    promptTemplate = """Use the following pieces of context to answer the question at the end.
    These pieces come from a FAQ page, try to only include the answer of the question you find most likely to fit. Provide the full answer.
    If you don't know the answer, just say 'I do not know.', do not try to make up an answer.

    {context}

    Question: {question}

    Answer:"""

    custom_faq_prompt = PromptTemplate.from_template(promptTemplate)

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=local_env.OPENAI_API_KEY)

    # prompt = hub.pull("rlm/rag-prompt")

    # example_messages = prompt.invoke(
    #     {"context": "filler context", "question": "filler question"}
    # ).to_messages()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_faq_prompt
        | llm
        | StrOutputParser()
    )

    full_response = ""
    for chunk in rag_chain.stream(question):
        full_response += chunk
    timings['generation_time'] = time.time() - start_time

    # Return the timings alongside the question and answer
    return {'question': question, 'answer': full_response, 'timings': timings}
