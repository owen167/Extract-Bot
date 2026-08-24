"""Shape heuristics for comic bubbles detected by the RT-DETR model."""

from __future__ import annotations

import cv2
import numpy as np


def _best_contour(crop_bgr: np.ndarray) -> np.ndarray | None:
    if crop_bgr.size == 0:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 35, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    height, width = gray.shape[:2]
    min_area = max(24.0, height * width * 0.015)
    usable = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    return max(usable or contours, key=cv2.contourArea)


def classify_bubble_shape(crop_bgr: np.ndarray) -> str:
    """Return a conservative semantic class from a bubble crop.

    The detector knows that a region is a bubble, while this lightweight stage
    inspects its border geometry. Ambiguous shapes intentionally return SPEECH.
    """
    contour = _best_contour(crop_bgr)
    if contour is None:
        return "SPEECH"

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return "SPEECH"

    x, y, width, height = cv2.boundingRect(contour)
    # A contour touching the crop edge is usually an incomplete panel/bubble
    # border caused by a detector crop. It cannot reliably prove a starburst.
    crop_height, crop_width = crop_bgr.shape[:2]
    if x <= 2 or y <= 2 or x + width >= crop_width - 2 or y + height >= crop_height - 2:
        return "SPEECH"
    bbox_area = float(max(1, width * height))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 120)
    edge_density = float(np.mean(edge_map > 0))
    rectangularity = area / bbox_area
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    epsilon = max(1.0, perimeter * 0.025)
    vertices = len(cv2.approxPolyDP(contour, epsilon, True))
    convexity = area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)

    # Clean rectangular caption/square borders have high rectangularity and
    # very few vertices after approximation.
    if vertices <= 6 and rectangularity >= 0.72:
        return "SQUARE"

    # Starburst borders have a very low convexity because of their deep points.
    # This check must happen before the thought heuristic.
    if convexity < 0.20:
        return "SHOUT"

    # Thought balloons have a dense ring of radial strokes. A smooth speech
    # balloon has substantially fewer edge pixels in the same crop.
    if edge_density >= 0.075 and convexity < 0.85:
        return "THOUGHT"

    return "SPEECH"
