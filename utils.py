#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility helpers for FPS calculation."""
import time
from collections import deque


class CvFpsCalc:
    """Lightweight FPS calculator modeled after MediaPipe demo utils."""

    def __init__(self, buffer_len=10):
        self._start_t = time.perf_counter()
        self._dq = deque(maxlen=buffer_len)

    def get(self):
        """Return smoothed frames-per-second value."""
        now = time.perf_counter()
        self._dq.append(now)
        if len(self._dq) >= 2:
            fps = (len(self._dq) - 1) / (self._dq[-1] - self._dq[0])
        else:
            fps = 0.0
        return round(fps, 2)
