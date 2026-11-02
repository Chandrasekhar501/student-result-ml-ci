import json
import os
import unittest
import joblib
import pandas as pd
from train_model import FEATURES

class TestMLPipeline(unittest.TestCase):
    def test_dataset_created(self):
        self.assertTrue(os.path.exists("student_results.csv"))
    def test_model_created(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))
    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))
    def test_accuracy_is_valid(self):
        with open("metrics.json", encoding="utf-8") as file:
            accuracy = json.load(file)["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")
        sample = pd.DataFrame([[85, 75, 80, 78]], columns=FEATURES)
        self.assertIn(int(model.predict(sample)[0]), [0, 1])
    def test_high_performance_student(self):
        model = joblib.load("student_result_model.pkl")
        sample = pd.DataFrame([[90, 85, 88, 80]], columns=FEATURES)
        self.assertEqual(int(model.predict(sample)[0]), 1)
    def test_low_performance_student(self):
        model = joblib.load("student_result_model.pkl")
        sample = pd.DataFrame([[55, 30, 40, 35]], columns=FEATURES)
        self.assertEqual(int(model.predict(sample)[0]), 0)

if __name__ == "__main__":
    unittest.main()
