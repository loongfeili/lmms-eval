import os
import re
from PIL import Image
from loguru import logger as eval_logger
from pathlib import Path
import yaml

# It's better to have a single source of truth for the dataset path.
# Reading it from the yaml config ensures consistency.
with open(Path(__file__).parent / "visual_correspondence.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # Remove lines with !function constructor
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

# The data file path is absolute, so we can get the directory from it.
# The images are located in an 'images' subdirectory within the same directory as the .jsonl file.
DATASET_DIR = Path(config['dataset_kwargs']['data_files']).parent
IMAGE_DIR = DATASET_DIR

def visual_correspondence_doc_to_visual(doc):
    """
    Load images from the document.
    The doc contains relative paths to two images.
    """
    image_paths = doc["image"]
    visuals = []
    for image_path in image_paths:
        # Construct the full path to the image
        # First attempt: relative to the data file's directory
        full_path = IMAGE_DIR / "images" / image_path
        if full_path.exists():
            visuals.append(Image.open(full_path).convert("RGB"))
            continue
        else:
            eval_logger.warning(f"Image not found at path: {full_path}")

    return visuals

def visual_correspondence_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Format the text prompt from the document.
    """
    # The 'text' field in the jsonl already contains the formatted question.
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        return f"{pre_prompt}{doc['text']}{post_prompt}"
    return doc["text"]

def parse_visual_correspondence_answer(pred_text):
    """
    Parse the model's prediction to extract the answer letter (A, B, C, or D).
    This function handles various common output formats.
    """
    if not isinstance(pred_text, str):
        return None
        
    pred_text = pred_text.strip().upper()

    # Pattern: "Answer: X" or "The correct answer is X" etc.
    match = re.search(r"(?:ANSWER\s*IS|ANSWER\s*:|ANSWER\s*[:]?)\s*([A-D])", pred_text)
    if match:
        return match.group(1)

    # Pattern: "(X)" or "[X]"
    match = re.search(r"[\(\[]([A-D])[\)\]]", pred_text)
    if match:
        return match.group(1)

    # Pattern: "X." at the beginning of a line or string
    match = re.search(r"^([A-D])\.", pred_text, re.MULTILINE)
    if match:
        return match.group(1)

    # Find the first occurrence of a letter A, B, C, or D in the string
    match = re.search(r"([A-D])", pred_text)
    if match:
        return match.group(1)

    # If no specific pattern is matched, check if the prediction is just a single letter
    if len(pred_text) == 1 and pred_text in "ABCD":
        return pred_text

    # Fallback if no answer is found
    return None

def visual_correspondence_process_results(doc, results):
    """
    Process the results of the model's prediction.
    """
    pred = results[0] if results else ""
    pred_letter = parse_visual_correspondence_answer(pred)
    gt_letter = doc["gt_value"]

    is_correct = pred_letter == gt_letter

    return {
        "visual_correspondence_detailed": {
            "id": doc["id"],
            "correct": is_correct,
            "pred": pred_letter,
            "target": gt_letter,
            "raw_pred": pred,
        }
    }

def visual_correspondence_aggregate_detailed_results(results):
    """
    Aggregate the detailed results to compute the final accuracy.
    """
    correct_count = sum(1 for i in results if i['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy
