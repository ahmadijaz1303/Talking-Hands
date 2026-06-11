#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI endpoints for digit and English alphabet hand-gesture classification.

POST /digit
POST /english_alphabet

Body: multipart/form-data with field `image` as a binary file.
Response: {"prediction_class": "<label>", "confidence": <float>}
"""

import csv
import itertools
import copy
from pathlib import Path
from typing import Tuple

import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# Model wrappers
# -----------------------------------------------------------------------------

class TFLiteKeypointClassifier:
    """Minimal wrapper that returns both label and confidence."""

    def __init__(self, model_path: str, label_path: str):
        self.model_path = Path(model_path)
        self.label_path = Path(label_path)

        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with self.label_path.open(encoding="utf-8-sig") as f:
            self.labels = [row[0] for row in csv.reader(f)]

    def predict(self, landmark_list: list) -> Tuple[str, float]:
        """Return predicted label and confidence."""
        input_idx = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_idx, np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()
        output_idx = self.output_details[0]["index"]
        scores = np.squeeze(self.interpreter.get_tensor(output_idx))

        # Softmax if model output isn't already normalized
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = self.labels[pred_idx] if pred_idx < len(self.labels) else "Unknown"
        return label, confidence


# -----------------------------------------------------------------------------
# Image/landmark helpers (pulled from app.py)
# -----------------------------------------------------------------------------

def calc_landmark_list(image: np.ndarray, landmarks) -> list:
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list: list) -> list:
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        return temp_landmark_list

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def extract_landmarks(image_bgr: np.ndarray, hands_detector) -> list:
    """Run MediaPipe Hands on an image and return a single preprocessed landmark list."""
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    if not results.multi_hand_landmarks:
        return []
    # Use the first detected hand
    landmark_list = calc_landmark_list(image_bgr, results.multi_hand_landmarks[0])
    return pre_process_landmark(landmark_list)


# -----------------------------------------------------------------------------
# FastAPI setup
# -----------------------------------------------------------------------------

app = FastAPI(title="Hand Gesture Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe Hands configured for static images
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5
)

# Load classifiers
digit_classifier = TFLiteKeypointClassifier(
    model_path="model/keypoint_classifier/keypoint_classifier.tflite",
    label_path="model/keypoint_classifier/keypoint_classifier_label.csv",
)

new_english_classifier = TFLiteKeypointClassifier(
    model_path="model/New_english_dataset/keypoint_classifier.tflite",
    label_path="model/New_english_dataset/keypoint_classifier_label.csv",
)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

async def _predict(file: UploadFile, classifier: TFLiteKeypointClassifier):
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    image = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")

    # Mirror the image horizontally to match the preprocessing used in app3.py 
    # where cv.flip(image, 1) was applied before extracting landmarks and training the model.
    image = cv.flip(image, 1)

    landmark = extract_landmarks(image, hands)
    if not landmark:
        return {"prediction_class": "no_hand", "confidence": 0.0, "hand_detected": False}

    label, confidence = classifier.predict(landmark)
    return {
        "prediction_class": label,
        "confidence": confidence,
        "hand_detected": True,
    }


@app.post("/digit")
async def predict_digit(image: UploadFile = File(...)):
    return await _predict(image, digit_classifier)


@app.post("/english_alphabet")
async def predict_english_alphabet(image: UploadFile = File(...)):
    return await _predict(image, new_english_classifier)


# -----------------------------------------------------------------------------
# For local debugging: uvicorn endpoint:app --host 0.0.0.0 --port 8080
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("endpoint:app", host="0.0.0.0", port=8080, reload=False)
