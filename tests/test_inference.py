import os
import sys
import unittest

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import generate_question


class TestModelAInference(unittest.TestCase):
    def test_generate_question_basic(self):
        article = "Ali went to Lahore in 2020. The weather was extremely hot. He visited the Badshahi Mosque."
        question = generate_question(article)

        self.assertIsInstance(question, str)
        self.assertTrue(question.endswith("?"))
        self.assertGreater(len(question), 10)

    def test_generate_question_single_sentence(self):
        article = "This is a very short article with only one sentence."
        question = generate_question(article)

        self.assertIsInstance(question, str)
        self.assertTrue(question.endswith("?"))


if __name__ == "__main__":
    unittest.main()
