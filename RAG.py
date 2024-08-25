import requests

class RAG:
    @staticmethod
    def get_answer(context, query):
        # Construct the prompt
        prompt = f"{context}\nQuestion: {query}\nAnswer:"

        # Set up the API parameters for the chat model endpoint
        api_url = "https://api.openai.com/v1/chat/completions"  # Endpoint for chat models
        headers = {
            "Authorization": "Bearer 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"
        }
        data = {
            "model": "gpt-3.5-turbo",  # Specify the model you are using
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "stop": ["\n", "Question:"]
        }

        # Make the API request
        response = requests.post(api_url, headers=headers, json=data)

        # Check for successful response
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return ""

        # Parse the response
        result = response.json()

        # Extract the answer
        try:
            answer = result['choices'][0]['message']['content'].strip() if result.get('choices') else ""
        except KeyError as e:
            print(f"KeyError: {e}, Response: {response.text}")
            answer = ""

        return answer


