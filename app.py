from flask import Flask, request, jsonify
import openai
from environment import Environment

app = Flask(__name__)
local_env = Environment()

# Configure the OpenAI library with your API key
openai.api_key = local_env.OPENAI_API_KEY

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    question = data.get('question', '')

    try:
        # Using the OpenAI Python client library for the request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?"
                }
            ],
            temperature=0.5,
            max_tokens=100
        )
        
        # Extracting the text from the response
        # Note: Adjust the extraction logic based on the actual structure of the response
        # The following assumes you're looking for the last message in the 'choices' list
        answer = response['choices'][0]['message']['content'] if response['choices'] else 'Sorry, I could not fetch a response.'

    except Exception as e:
        print(e)
        answer = 'Failed to connect to the API or process the response.'

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

