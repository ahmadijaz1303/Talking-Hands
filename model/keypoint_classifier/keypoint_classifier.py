#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = None
        self.keras_model = None
        self.input_details = None
        self.output_details = None

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                                   num_threads=num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except ValueError:
            keras_model_path = os.path.splitext(model_path)[0] + '.keras'
            if not os.path.exists(keras_model_path):
                raise
            # Fall back to the full Keras model when the local TFLite runtime
            # is older than the generated .tflite file.
            self.keras_model = tf.keras.models.load_model(keras_model_path)

    def __call__(
        self,
        landmark_list,
    ):
        result = self.predict(landmark_list)
        return result['class_id']

    def predict(
        self,
        landmark_list,
    ):
        if self.keras_model is not None:
            scores = np.squeeze(self.keras_model.predict(
                np.array([landmark_list], dtype=np.float32),
                verbose=0,
            ))
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            return {'class_id': class_id, 'confidence': confidence, 'scores': scores}

        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        scores = np.squeeze(
            self.interpreter.get_tensor(output_details_tensor_index))
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        return {'class_id': class_id, 'confidence': confidence, 'scores': scores}
