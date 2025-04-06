import unittest
import pandas as pd # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
import numpy as np # type: ignore
from PyPDF2 import PdfReader # type: ignore
from docx import Document # type: ignore

def extract_text_from_file(file_path):
    """
    Extracts text from a PDF or DOCX file.
    """
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages)
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        text = " ".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are allowed.")
    return text

def auto_grade_pdf_docx(assignment_path, rubric_path):
    """
    Automatically grades assignments based on a rubric for PDF/DOCX files.
    """
    try:
        # Extract text from rubric and assignments
        rubric_text = extract_text_from_file(rubric_path)
        assignment_text = extract_text_from_file(assignment_path)

        # Split rubric into answers and points (assume rubric is structured as "answer:points")
        rubric_lines = rubric_text.split("\n")
        rubric = pd.DataFrame([line.split(":") for line in rubric_lines if ":" in line], columns=["answer", "points"])
        rubric["points"] = rubric["points"].astype(float)

        # Split assignments into student names and answers (assume "name:answer" format)
        assignment_lines = assignment_text.split("\n")
        submissions = pd.DataFrame([line.split(":") for line in assignment_lines if ":" in line], columns=["name", "answer"])

        # Validate required columns
        if 'answer' not in rubric.columns or 'points' not in rubric.columns:
            raise ValueError("Rubric file must contain 'answer' and 'points' columns.")
        if 'name' not in submissions.columns or 'answer' not in submissions.columns:
            raise ValueError("Assignment file must contain 'name' and 'answer' columns.")

        # Initialize results dictionary
        results = {
            'students': [],
            'scores': [],
            'correct_answers': [],
        }

        # Vectorize rubric answers
        vectorizer = TfidfVectorizer()
        rubric_vectors = vectorizer.fit_transform(rubric['answer'])

        # Grade each submission
        for _, row in submissions.iterrows():
            student_answer = row['answer']
            student_vector = vectorizer.transform([student_answer])

            # Calculate similarity with rubric answers
            similarities = cosine_similarity(student_vector, rubric_vectors)
            max_sim_idx = np.argmax(similarities)
            max_similarity = similarities.max()

            # Normalize score
            normalized_score = (max_similarity * rubric.iloc[max_sim_idx]['points']) / 1.0

            results['students'].append(row['name'])
            results['scores'].append(normalized_score)
            results['correct_answers'].append(rubric.iloc[max_sim_idx]['answer'])

        return results

    except Exception as e:
        raise Exception(f"Grading error: {str(e)}")
    import unittest
from grading_logic import auto_grade_pdf_docx

class TestGradingLogic(unittest.TestCase):
    def test_auto_grade_pdf_docx(self):
        assignment_path = "test_data/assignment.docx"
        rubric_path = "test_data/rubric.docx"
        results = auto_grade_pdf_docx(assignment_path, rubric_path)
        self.assertIn("students", results)
        self.assertIn("scores", results)
        self.assertIn("correct_answers", results)

if __name__ == '__main__':
    unittest.main()
    from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

def grade_open_answers(student_answers, model_answers):
    """Grades open-ended answers using cosine similarity."""
    if not student_answers or not model_answers:
        raise ValueError("Both student answers and model answers are required.")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([*student_answers.values(), *model_answers.values()])
    similarity_matrix = cosine_similarity(tfidf_matrix[:len(student_answers)], tfidf_matrix[len(student_answers):])

    scores = {question: round(similarity_matrix[i].max(), 2) * 100 for i, question in enumerate(student_answers.keys())}
    return scores
# tests/test_grading_logic.py
import unittest
from grading_logic import grade_open_answers

class TestGradingLogic(unittest.TestCase):
    def test_grade_open_answers(self):
        student_answers = {"Q1": "Answer 1", "Q2": "Answer 2"}
        model_answers = {"Q1": "Correct Answer 1", "Q2": "Correct Answer 2"}
        scores = grade_open_answers(student_answers, model_answers)
        self.assertIn("Q1", scores)
        self.assertIn("Q2", scores)

if __name__ == '__main__':
    unittest.main()