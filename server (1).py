# /FCAPS/RAG/server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_agent import run_llm_agent

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/search', methods=['POST'])
def search_logs():
    try:
        data = request.json
        query = data.get("query", "")

        if not query.strip():
            return jsonify({"error": "No query provided"}), 400

        # Run full FCAPS-Aware RAG + Agent Pipeline
        response = run_llm_agent(query)

        # Return just the response (optionally, add intent/risk later)
        return jsonify({
            "query": query,
            "response": response
        })

    except Exception as e:
        print(f"Error in /search: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
