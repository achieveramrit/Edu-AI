import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

def generate_feedback(grading_results):
    """
    Generates personalized feedback based on grading results
    """
    feedback = []
    
    for i in range(len(grading_results['students'])):
        student = grading_results['students'][i]
        score = grading_results['scores'][i]
        student_answer = grading_results['students'][i]
        correct_answer = grading_results['correct_answers'][i]
        
        # Basic feedback based on score
        if score >= 90:
            assessment = "Excellent work!"
        elif score >= 70:
            assessment = "Good job, but there's room for improvement."
        else:
            assessment = "Let's review this material again."
        
        # Specific feedback by comparing answers
        specific_feedback = compare_answers(student_answer, correct_answer)
        
        # Construct full feedback
        full_feedback = {
            'student': student,
            'score': score,
            'assessment': assessment,
            'specific_feedback': specific_feedback,
            'suggestions': get_suggestions(score)
        }
        
        feedback.append(full_feedback)
    
    return feedback

def compare_answers(student_answer, correct_answer):
    """Generates specific feedback by comparing answers"""
    # Tokenize and clean both answers
    student_words = clean_text(student_answer)
    correct_words = clean_text(correct_answer)
    
    # Find missing concepts
    missing = set(correct_words) - set(student_words)
    
    if not missing:
        return "Your answer covered all the key points."
    else:
        return f"Consider including these concepts: {', '.join(missing)}"

def clean_text(text):
    """Cleans text for comparison"""
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    return [word for word in tokens if word not in stop_words and word.isalpha()]

def get_suggestions(score):
    """Provides learning suggestions based on score"""
    if score >= 90:
        return ["Challenge yourself with advanced materials on this topic."]
    elif score >= 70:
        return ["Review the main concepts.", "Practice with additional examples."]
    else:
        return [
            "Revisit the core materials on this topic.",
            "you should Schedule a meeting with your instructor for clarification.",
            "Try some practice exercises before moving on."
        ]