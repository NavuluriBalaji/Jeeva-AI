from flask import Flask, request, jsonify
import requests
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_API_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def get_disease_info(disease):
    """Placeholder function for local disease info retrieval"""
    return f"Basic information about {disease}."

@app.route('/v1/disease_info', methods=['POST'])
def disease_info():
    """API endpoint to fetch disease information from Gemini API"""
    payload = request.get_json()

    if not payload or 'disease' not in payload:
        return jsonify({'error': 'Invalid request. Please provide a disease name in the JSON payload.'}), 400

    disease = payload.get('disease')
    prompt = f"Provide detailed information about {disease}, including its description, symptoms, and treatment."

    headers = {'Content-Type': 'application/json'}
    gemini_payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        # Log the request payload
        logging.info(f"Request payload: {gemini_payload}")

        # Make request to Gemini API
        response = requests.post(GEMINI_API_BASE_URL, headers=headers, json=gemini_payload)
        response.raise_for_status()

        # Log the response from Gemini API
        logging.info(f"Response from Gemini API: {response.json()}")

        # Process Gemini API response
        gemini_response = response.json()
        gemini_info = gemini_response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No information available.')

        return jsonify({
            'disease': disease,
            'info': gemini_info,
            'local_info': get_disease_info(disease)  # Optional local information
        })

    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Gemini API: {e}")
        return jsonify({'error': f"Error calling Gemini API: {e}"}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)