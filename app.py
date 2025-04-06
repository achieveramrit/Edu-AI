from flask import Flask, request, render_template, jsonify # type: ignore
import json
import os
import spacy # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import openai # type: ignore

app = Flask(__name__)
QUIZ_FILE = "quizzes.json"
openai.api_key = "your_openai_api_key"  # Replace with your API key
nlp = spacy.load("en_core_web_sm")

# Ensure quiz file exists
def load_quizzes():
    if os.path.exists(QUIZ_FILE):
        with open(QUIZ_FILE, "r") as file:
            return json.load(file)
    return {}

def save_quizzes(quizzes):
    with open(QUIZ_FILE, "w") as file:
        json.dump(quizzes, file, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

# AI-powered quiz generation
@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    text_content = data.get("text")
    
    if not text_content:
        return jsonify({"error": "No text provided"}), 400
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Generate a multiple-choice quiz with options and correct answers based on the given text."},
                  {"role": "user", "content": text_content}]
    )
    
    quiz_data = response['choices'][0]['message']['content']
    save_quizzes(json.loads(quiz_data))
    return jsonify({"message": "Quiz generated successfully!", "quiz": quiz_data})

# AI grading for open-ended answers
@app.route('/grade_open_answers', methods=['POST'])
def grade_open_answers():
    data = request.json
    student_answers = data.get("answers")
    model_answers = data.get("model_answers")
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([*student_answers.values(), *model_answers.values()])
    similarity_matrix = cosine_similarity(tfidf_matrix[:len(student_answers)], tfidf_matrix[len(student_answers):])
    
    scores = {question: round(similarity_matrix[i].max(), 2) * 100 for i, question in enumerate(student_answers.keys())}
    return jsonify({"scores": scores})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify # type: ignore
import openai # type: ignore

app = Flask(__name__)

# Set up OpenAI API Key (Use your own key)
OPENAI_API_KEY = "your_openai_api_key"

def get_chatbot_response(user_query):
    """Generates AI-powered explanations using OpenAI GPT."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI tutor that helps students understand quiz mistakes."},
                {"role": "user", "content": user_query}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return "Sorry, I couldn't process your request."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    user_query = data.get("question", "")
    response = get_chatbot_response(user_query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify # type: ignore
from quiz_logic import generate_quiz

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz_route():
    try:
        data = request.json
        text_content = data.get("text")
        quiz_data = generate_quiz(text_content)
        return jsonify({"message": "Quiz generated successfully!", "quiz": quiz_data})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    from flask import Flask, request, jsonify
from grading_logic import grade_open_answers

@app.route('/grade_open_answers', methods=['POST'])
def grade_open_answers_route():
    try:
        data = request.json
        student_answers = data.get("answers")
        model_answers = data.get("model_answers")
        scores = grade_open_answers(student_answers, model_answers)
        return jsonify({"scores": scores})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    from flask import Flask, request, jsonify
from chatbot_logic import get_chatbot_response

@app.route("/chatbot", methods=["POST"])
def chatbot_route():
    try:
        data = request.json
        user_query = data.get("question", "")
        response = get_chatbot_response(user_query)
        return jsonify({"response": response})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500