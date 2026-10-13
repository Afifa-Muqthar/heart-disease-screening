"""Focused tests for the Flask prediction endpoint."""

import json
import logging
import unittest
from unittest.mock import patch

import app as application
import numpy as np


class FakeModel:
    def __init__(self, probability=0.9, error=None):
        self.probability = probability
        self.error = error
        self.calls = 0
        self.last_frame = None

    def predict_proba(self, frame):
        self.calls += 1
        self.last_frame = frame
        if self.error:
            raise self.error
        return np.array([[1 - self.probability, self.probability]])


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        application.app.config.update(TESTING=True)
        self.client = application.app.test_client()
        self.model = FakeModel()
        self.model_patch = patch.object(application, "MODEL", self.model)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    @staticmethod
    def valid_payload(**updates):
        return {
            "age": 54,
            "sex": "Male",
            "cp": "asymptomatic",
            "exang": False,
            "trestbps": 130,
            **updates,
        }

    def post(self, payload):
        return self.client.post("/api/predict", json=payload)

    def test_valid_inputs_return_categorical_result_in_feature_order(self):
        response = self.post(self.valid_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["band"], "higher")
        self.assertEqual(response.json["band_name"], "Higher")
        self.assertNotIn("score", response.json)
        self.assertEqual(list(self.model.last_frame.columns), application.FEATURES)
        self.assertEqual(self.model.last_frame.loc[0, "age"], 54)
        self.assertEqual(self.model.last_frame.loc[0, "trestbps"], 130)

    def test_missing_fields_return_400_and_names(self):
        response = self.post({"age": 54})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["missing"], ["sex", "cp", "exang", "trestbps"])
        self.assertEqual(self.model.calls, 0)

    def test_invalid_categories_return_400(self):
        invalid_cases = (
            {"sex": "Other"},
            {"cp": "unknown"},
            {"exang": "yes"},
            {"exang": 1},
        )
        for updates in invalid_cases:
            with self.subTest(updates=updates):
                response = self.post(self.valid_payload(**updates))
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.model.calls, 0)

    def test_supported_numeric_boundaries_are_accepted(self):
        for age, blood_pressure in ((28, 80), (77, 200)):
            with self.subTest(age=age, blood_pressure=blood_pressure):
                response = self.post(self.valid_payload(age=age, trestbps=blood_pressure))
                self.assertEqual(response.status_code, 200)
                self.assertNotEqual(response.json["band"], "unavailable")

    def test_out_of_range_numeric_values_and_fractional_age_are_rejected(self):
        invalid_cases = (
            {"age": 17},
            {"age": 101},
            {"age": 54.5},
            {"trestbps": 79},
            {"trestbps": 201},
        )
        for updates in invalid_cases:
            with self.subTest(updates=updates):
                response = self.post(self.valid_payload(**updates))
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.model.calls, 0)

    def test_ages_outside_dataset_range_return_unavailable_without_inference(self):
        for age in (18, 100):
            with self.subTest(age=age):
                response = self.post(self.valid_payload(age=age))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["band"], "unavailable")
                self.assertEqual(response.json["band_name"], "Estimate unavailable")
        self.assertEqual(self.model.calls, 0)

    def test_unavailable_age_still_validates_every_category(self):
        invalid_cases = (
            {"sex": "Other"},
            {"cp": "unknown"},
            {"exang": "yes"},
            {"exang": 1},
        )
        for updates in invalid_cases:
            with self.subTest(updates=updates):
                response = self.post(self.valid_payload(age=18, **updates))
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)

    def test_malformed_requests_return_json_400(self):
        responses = (
            self.client.post("/api/predict", data="not json", content_type="application/json"),
            self.client.post("/api/predict", data="{}", content_type="text/plain"),
            self.client.post("/api/predict", data=json.dumps([1, 2]), content_type="application/json"),
        )
        for response in responses:
            with self.subTest(status=response.status_code):
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)
        self.assertEqual(self.model.calls, 0)

    def test_oversized_request_returns_json_413(self):
        response = self.client.post(
            "/api/predict",
            data=json.dumps({"padding": "x" * 5000}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.json)

    def test_inference_error_is_generic_and_logs_no_payload_or_exception_text(self):
        self.model.error = RuntimeError("private inference detail")
        payload = self.valid_payload()

        with self.assertLogs(application.app.logger, level=logging.ERROR) as captured:
            response = self.post(payload)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {"error": "The estimate could not be completed. Please try again later."})
        self.assertIn("RuntimeError", " ".join(captured.output))
        self.assertNotIn("private inference detail", " ".join(captured.output))
        self.assertNotIn(json.dumps(payload), " ".join(captured.output))


if __name__ == "__main__":
    unittest.main()
