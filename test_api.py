import unittest
from fastapi.testclient import TestClient
from app import app

class TestStudentResultAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Request-ID", response.headers)
    def test_high_performance_prediction(self):
        response = self.client.post("/predict", json={"attendance":90,"internal_marks":85,"assignment_marks":88,"previous_score":80})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "PASS")
    def test_low_performance_prediction(self):
        response = self.client.post("/predict", json={"attendance":55,"internal_marks":30,"assignment_marks":40,"previous_score":35})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "FAIL")
    def test_validation_error(self):
        response = self.client.post("/predict", json={"attendance":"not-a-number","internal_marks":30})
        self.assertEqual(response.status_code, 422)
    def test_unknown_route(self):
        self.assertEqual(self.client.get("/not-a-real-route").status_code, 404)

if __name__ == "__main__":
    unittest.main()
