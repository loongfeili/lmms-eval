import os
import re
import json
from collections import defaultdict
from PIL import Image
from loguru import logger as eval_logger
from pathlib import Path
import yaml

with open(Path(__file__).parent / "omnispatial.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
dataset_path = yaml.safe_load("".join(safe_data))['dataset_path']

def omnispatial_doc_to_visual(doc):
    """Extract visual input from document"""
    # Based on api_eval.py, images are stored as "task_type/{response_id.split('_')[0]}.png"
    response_id = doc["id"]
    task_type = doc["task_type"]
    
    # Construct image path
    img_filename = f"{response_id.split('_')[0]}.png"
    img_path = os.path.join(f"{dataset_path}", task_type, img_filename)
    if os.path.exists(img_path):
        return [Image.open(img_path).convert("RGB")]
    return []

def omnispatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format the question and options as text input"""
    question = doc["question"]
    options = doc["options"]
    
    # Format options as A, B, C, D...
    formatted_options = []
    for i, option in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D...
        formatted_options.append(f"{letter}. {option}")
    
    options_text = "\n".join(formatted_options)
    
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    
    full_text = f"{pre_prompt}{question}\n{options_text}{post_prompt}"
    return full_text


def omnispatial_doc_to_target(doc):
    """Extract target answer as letter (A, B, C, D)"""
    gt_idx = doc["answer"]  # This is the index of the correct answer
    gt_letter = chr(65 + gt_idx)  # Convert to A, B, C, D
    return gt_letter


def parse_omnispatial_answer(pred_text):
    """Parse model prediction to extract answer letter"""
    pred_text = pred_text.strip().upper()
    
    # Direct single letter answer
    if len(pred_text) == 1 and pred_text in "ABCD":
        return pred_text
    
    # Pattern: "Answer: X" or "Answer is X"
    answer_pattern = re.compile(r"(?:Answer(?:\s*is)?)\s*:\s*([A-D])", re.IGNORECASE)
    match = answer_pattern.search(pred_text)
    if match:
        return match.group(1).upper()
    
    # Pattern: "(X)" where X is A, B, C, or D
    paren_pattern = re.compile(r"\(([A-D])\)", re.IGNORECASE)
    match = paren_pattern.search(pred_text)
    if match:
        return match.group(1).upper()
    
    # Pattern: "X." at the beginning of a line
    line_pattern = re.compile(r"^([A-D])\.", re.IGNORECASE | re.MULTILINE)
    match = line_pattern.search(pred_text)
    if match:
        return match.group(1).upper()
    
    # Look for the first occurrence of A, B, C, or D
    letter_pattern = re.compile(r"([A-D])", re.IGNORECASE)
    match = letter_pattern.search(pred_text)
    if match:
        return match.group(1).upper()
    
    # Default fallback
    eval_logger.warning(f"Unrecognized prediction format: {pred_text}. Defaulting to 'A'.")
    return "A"


def omnispatial_process_results(doc, results):
    """Process prediction results and return metrics"""
    pred = results[0] if results else ""
    pred_letter = parse_omnispatial_answer(pred)
    gt_letter = omnispatial_doc_to_target(doc)

    # Basic accuracy
    is_correct = (pred_letter == gt_letter)
    
    # Extract metadata for detailed analysis
    task_type = doc["task_type"]
    sub_task = doc["sub_task_type"] 
    response_id = doc["id"]

    # Return results that can be aggregated
    return {
        "acc": is_correct,
        "omnispatial_detailed": {
            "task_type": task_type,
            "sub_task": sub_task,
            "response_id": response_id,
            "correct": is_correct,
            "pred": pred_letter,
            "target": gt_letter
        }
    }


def omnispatial_aggregate_detailed_results(results):
    """
    Aggregate results with detailed breakdown by task type and sub-task
    """
    # Initialize counters
    stats = {
        "Total": [],
        "Dynamic_Reasoning": {"Manipulation": [], "Motion_Analysis": [], "Total": []},
        "Spatial_Interaction": {"Traffic_Analysis": [], "Localization": [], "Geospatial_Strategy": [], "Total": []},
        "Complex_Logic": {"Pattern_Recognition": [], "Geometric_Reasoning": [], "Total": []}, 
        "Perspective_Taking": {"Egocentric": [], "Allocentric": [], "Hypothetical": [], "Total": []},
    }
    
    # Aggregate results
    for result in results:
        task_type = result["task_type"]
        sub_task = result["sub_task"]
        is_correct = result["correct"]
        
        stats["Total"].append(is_correct)
        if task_type in stats:
            if sub_task in stats[task_type]:
                stats[task_type][sub_task].append(is_correct)
            stats[task_type]["Total"].append(is_correct)
    
    # Calculate percentages
    results_summary = {}
    eps = 1e-6
    
    overall_acc = sum(stats["Total"]) / (len(stats["Total"]) + eps) * 100
    results_summary["Overall"] = overall_acc
    
    for task in [k for k in stats if k != "Total"]:
        if len(stats[task]["Total"]) > 0:
            task_acc = sum(stats[task]["Total"]) / len(stats[task]["Total"]) * 100
            results_summary[task] = task_acc
            
            for sub_task in stats[task]:
                if sub_task != "Total" and len(stats[task][sub_task]) > 0:
                    sub_acc = sum(stats[task][sub_task]) / len(stats[task][sub_task]) * 100
                    results_summary[f"{task}_{sub_task}"] = sub_acc
    
    eval_logger.info(f"OmniSpatial Overall: {overall_acc:.2f}%")
    for task in [k for k in stats if k != "Total"]:
        if len(stats[task]["Total"]) > 0:
            task_acc = sum(stats[task]["Total"]) / len(stats[task]["Total"]) * 100
            print(f"{task}: {task_acc:.2f}% Correct / ALL : {sum(stats[task]['Total'])}/{len(stats[task]['Total'])}")
            for sub_task in stats[task]:
                if sub_task != "Total" and len(stats[task][sub_task]) > 0:
                    sub_acc = sum(stats[task][sub_task]) / len(stats[task][sub_task]) * 100
                    eval_logger.info(f"    {sub_task}: {sub_acc:.2f}%")
    
    return overall_acc
