import openai # type: ignore

openai.api_key = "your_openai_api_key"  # Replace with your API key

def get_chatbot_response(user_query):
    """Generates AI-powered explanations using OpenAI GPT."""
    if not user_query:
        raise ValueError("No question provided")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI tutor that helps students understand quiz mistakes."},
            {"role": "user", "content": user_query}
        ]
    )
    return response["choices"][0]["message"]["content"]