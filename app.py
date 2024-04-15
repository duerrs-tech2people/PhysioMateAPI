from faq_reader import answerFaqRelatedQuestion
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s - %(levelname)s - %(message)s')

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    if not data or 'question' not in data:
        app.logger.error('No question provided')
        return jsonify({"error": "No question provided"}), 400

    question = data['question']
    app.logger.info(f"Received question: {question}")
    try:
        response = answerFaqRelatedQuestion(question)
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error processing question: {e}")
        return jsonify({"error": "Error processing your question"}), 500

if __name__ == '__main__':
    app.run(debug=True)