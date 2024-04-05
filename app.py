from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    question = data.get('question', '')
    # Here, you'd integrate with the ChatGPT API or process the question as needed
    # For demonstration, we'll just echo the question back
    response = {'answer': f'You asked: {question}'}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
