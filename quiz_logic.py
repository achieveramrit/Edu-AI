import openai # type: ignore
import json

openai.api_key = "your_openai_api_key"  # Replace with your API key

def generate_quiz(text_content):
    """Generates a multiple-choice quiz using OpenAI GPT."""
    if not text_content:
        raise ValueError("No text provided")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Generate a multiple-choice quiz with options and correct answers based on the given text."},
            {"role": "user", "content": text_content}
        ]
    )
    quiz_data = response['choices'][0]['message']['content']
    return json.loads(quiz_data)