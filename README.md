# PhysioMateAPI
This is the backend for the Chatbot PhysioMate, which handles API requests.

# Running the script
python app.py

# Sending a request
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/chatbot" -Method POST -ContentType "application/json" -Body '{"question":"What is the capital of England?"}'