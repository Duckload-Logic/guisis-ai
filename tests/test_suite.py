import sys
import os
import unittest
from fastapi.testclient import TestClient

# Add project root to path to allow src imports
sys.path.append(os.getcwd())
from src.main import app

class TestUrgencyClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def predict(self, text: str):
        """
        Calls the production API.
        The business rules are now integrated into the service layer.
        """
        response = self.client.post("/api/v1/classify", json={"text": text})
        self.assertEqual(response.status_code, 200, f"Failed for text: {text}")
        return response.json()

    def test_low_urgency(self):
        test_cases = [
            ("How to apply for a student ID?", "LOW"),
            ("Saan po pwede makita yung campus map?", "LOW"),
            ("What are the library hours for this week?", "LOW"),
            ("Pa-schedule po ng appointment.", "LOW"),
            ("Good morning, inquire lang po about counseling.", "LOW"),
        ]
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.predict(text)
                print(
                    f"[LOW] '{text[:40]}...' -> "
                    + f"{result['level']} ({result['confidence']:.2f})"
                )
                self.assertEqual(
                    result['level'],
                    expected,
                    f"Expected {expected}, got {result['level']} for: {text}"
                )

    def test_medium_urgency(self):
        test_cases = [
            ("I have a conflict in my class schedule for Chem 101.", "MEDIUM"),
            ("Hindi po ako maka-enroll sa elective ko, puno na raw.", "MEDIUM"),
            (
                "Professor is not showing up for our online sync sessions.",
                "MEDIUM"
            ),
            (
                "Medyo nahihirapan po ako sa Calculus, baka may tips kayo.",
                "MEDIUM"
            ),
            ("Kinakabahan po ako sa thesis defense.", "MEDIUM"),
        ]
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.predict(text)
                print(
                    f"[MEDIUM] '{text[:40]}...' -> "
                    + f"{result['level']} ({result['confidence']:.2f})"
                )
                self.assertEqual(
                    result['level'],
                    expected,
                    f"Expected {expected}, got {result['level']} for: {text}"
                )

    def test_high_urgency(self):
        test_cases = [
            (
                "My scholarship is about to be revoked because of an "
                + "encoding error.",
                "HIGH"
            ),
            (
                "Nawawala po yung grades ko sa SIS, kailangan ko na for "
                + "graduation.",
                "HIGH"
            ),
            (
                "I cannot pay the tuition fee before the deadline "
                + "tomorrow.",
                "HIGH"
            ),
            (
                "Sobrang stress na po ako. Parang gusto ko na lang mag-stop.",
                "HIGH"
            ),
            (
                "May nagbabanta po sakin sa GC. Natatakot po ako pumasok.",
                "HIGH"
            ),
        ]
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.predict(text)
                print(
                    f"[HIGH] '{text[:40]}...' -> "
                    + f"{result['level']} ({result['confidence']:.2f})"
                )
                self.assertEqual(result['level'], expected)

    def test_critical_urgency(self):
        test_cases = [
            (
                "I am feeling extremely overwhelmed and I don't know what to "
                + "do anymore. I need help.",
                "HIGH"
            ),
            ("May nakita po akong student na nahimatay sa corridor.", "HIGH"),
            ("Someone is threatening me in the campus, I am scared.", "HIGH"),
            ("Tulong.", "CRITICAL"),
            ("Di ko na kaya.", "CRITICAL"),
            ("I want to die.", "CRITICAL"),
            ("May suicidal thoughts po ako.", "CRITICAL"),
        ]
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.predict(text)
                print(
                    f"[CRITICAL] '{text[:40]}...' -> "
                    + f"{result['level']} ({result['confidence']:.2f})"
                )
                # Ensure it captures the urgency (HIGH or CRITICAL)
                self.assertIn(
                    result['level'],
                    ["HIGH", "CRITICAL"],
                    f"Expected HIGH/CRITICAL, got {result['level']} for: {text}"
                )

if __name__ == "__main__":
    unittest.main()