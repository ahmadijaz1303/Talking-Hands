# Talking Hands

A final-year AI project for Pakistan Sign Language communication, developed from September 2025 to July 2026. This repository contains gesture-recognition notebooks, model assets, and real-time Python inference scripts.

## Problem

Talking Hands explores a two-way communication workflow between recognized sign-language gestures and text or voice output.

## Included components

- Real-time gesture inference scripts for English, Urdu, digits, and greetings.
- OpenCV camera capture and MediaPipe hand-landmark extraction.
- TensorFlow/Keras model assets and experimentation notebooks.
- A Python endpoint prototype and dependency list.

## Pipeline

```text
Camera frame → MediaPipe landmarks → trained gesture model → predicted label → text/voice output
```

## Repository layout

- `Notebooks/`: experimentation and training notebooks.
- `model/`: model assets used by the inference scripts.
- `inference_english.py`, `inference_urdu.py`, `inference_digit.py`, `inference_greeting.py`: supported inference entry points.
- `endpoint.py`: Python endpoint prototype.
- `requirements.txt`: dependencies.

## Run locally

```bash
pip install -r requirements.txt
python inference_english.py
```

A webcam and the included model assets are required.

## Scope and limitations

This is an academic project, not a production sign-language translation service. Results depend on the gesture groups and model assets included here, and may vary with lighting, camera quality, hand position, signer variation, and unsupported gestures.

## Recognition

- 1st Place — Inter-University SPARK Business Idea Competition (2025)
- Most Innovative Idea Award — Techno Verse Think Tank (2025)

## Technologies

Python, TensorFlow/Keras, MediaPipe, OpenCV, and TensorFlow Lite.
