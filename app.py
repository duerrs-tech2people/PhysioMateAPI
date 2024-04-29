from faq_reader import answerFaqRelatedQuestion
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)
clicks = []
timings_array = []

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
        timings_array.append([response['question'], response['timings']])
        app.logger.info(timings_array)
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error processing question: {e}")
        return jsonify({"error": "Error processing your question"}), 500
    
@app.route('/api/click', methods=['POST'])
def track_click():
    data = request.json
    if not data or not all(k in data for k in ['userId', 'location', 'date']):
        app.logger.error('Missing data in click tracking')
        return jsonify({"error": "Missing data"}), 400

    clicks.append(data)
    app.logger.info(f"Logged click: {data}")
    return jsonify({"success": "Click logged"}), 200

@app.route('/api/clicks', methods=['GET'])
def get_clicks():
    app.logger.info(clicks)
    return jsonify(clicks)


if __name__ == '__main__':
    app.run(debug=True)