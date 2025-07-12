import os
from pathlib import Path
from PIL import Image
import re

def total_dist_doc_to_visual(doc):
    """
    Extract visual input from the document.

    Args:
        doc (dict): A dictionary containing document data.

    Returns:
        List[Image.Image]: A list of PIL Image objects.
    """
    images = []
    for image_path in doc["image"]:
        full_path = Path(image_path)
        if full_path.exists():
            images.append(Image.open(full_path).convert("RGB"))
    return images

def total_dist_doc_to_text(doc):
    """
    Format the question and context as text input.

    Args:
        doc (dict): A dictionary containing document data.

    Returns:
        str: The formatted text input.
    """
    return doc["text"]

def rep_to_distance_valuse(pred_text):
    """
    Convert the prediction text to a numerical value.

    Args:
        pred_text (str): The prediction text from the model.

    Returns:
        float: The numerical value extracted from the prediction text.
    """
    match = re.search(r"[-+]?\d*\.\d+|\d+", pred_text)
    return float(match.group()) if match else 0.0


def total_dist_process_results(doc, results):
    """
    Process prediction results and return metrics.

    Args:
        doc (dict): A dictionary containing document data.
        results (list): A list of model predictions.

    Returns:
        dict: Processed results for evaluation.
    """
    pred = results[0] if results else 0.0
    pred_value = rep_to_distance_valuse(pred)
    gt_value = doc["gt_value"]
    return { "total distance":
            {
                "prediction": pred_value,
                "ground_truth": gt_value,
                "abs_error": abs(pred_value - gt_value),
            }
    }