from flask import Flask, request, jsonify
import openai
from environment import Environment
import langchain as lc
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
import bs4
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./FAQ.txt")
docs = loader.load()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

app = Flask(__name__)
local_env = Environment()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
len(all_splits)

# Configure the OpenAI library with your API key
openai.api_key = local_env.OPENAI_API_KEY

# Initialize LangChain components
# Note: Adjust the parameters and configurations according to your requirements
document = lc.Document(text=faq_data)
retriever = lc.Retriever(document=document, retriever_type="regex", config={
    "question_regex": "Q: (.*?)\n",
    "answer_regex": "A: (.*?)\n\n"
})
llm = lc.OpenAICompletion(api_key=local_env.OPENAI_API_KEY, model="gpt-3.5-turbo-0125")

chat_component = lc.ChatComponent(
    components=[
        retriever,
        llm
    ]
)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    question = data.get('question', '')


    try:
        # Use LangChain to answer the question using the FAQ
        answer = chat_component.answer(question)
    except Exception as e:
        print(e)
        answer = 'Failed to connect to the API or process the response.'

    return jsonify({'answer': answer})
    # try:
    #     # Using the OpenAI Python client library for the request
    #     response = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo-0125",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": question
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": "Hello! How can I assist you today?"
    #             }
    #         ],
    #         temperature=0.5,
    #         max_tokens=100
    #     )
        
    #     # Extracting the text from the response
    #     # Note: Adjust the extraction logic based on the actual structure of the response
    #     # The following assumes you're looking for the last message in the 'choices' list
    #     answer = response['choices'][0]['message']['content'] if response['choices'] else 'Sorry, I could not fetch a response.'

    # except Exception as e:
    #     print(e)
    #     answer = 'Failed to connect to the API or process the response.'

    # return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

