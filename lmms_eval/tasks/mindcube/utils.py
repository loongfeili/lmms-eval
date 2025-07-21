import os
import re
import json
from collections import defaultdict
from PIL import Image
from loguru import logger as eval_logger
from pathlib import Path
import yaml

# Try relative imports first, fall back to absolute imports
try:
    from .mindcube_util.result_init import _initialize_cogmap_results_structure, _update_similarity_metrics, _initialize_similarity_accumulators, _preserve_necessary_cogmap_fields, apply_filtering_to_results
    from .mindcube_util.parse_resps import(
        get_setting_from_id,
        determine_answer_fields,
        extract_answer,
        _extract_cognitive_map,
        _extract_grounded_cogmap,
    ) 
    from .src.evaluation.cogmap.cogmap_metrics import calculate_cogmap_similarity
except ImportError:
    # Fall back to absolute imports
    from lmms_eval.tasks.mindcube.mindcube_util.result_init import _initialize_cogmap_results_structure, _update_similarity_metrics, _initialize_similarity_accumulators, _preserve_necessary_cogmap_fields, apply_filtering_to_results
    from lmms_eval.tasks.mindcube.mindcube_util.parse_resps import(
        get_setting_from_id,
        determine_answer_fields,
        extract_answer,
        _extract_cognitive_map,
        _extract_grounded_cogmap,
    ) 
    from lmms_eval.tasks.mindcube.src.evaluation.cogmap.cogmap_metrics import calculate_cogmap_similarity


with open(Path(__file__).parent / "mindcube.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
dataset_path = yaml.safe_load("".join(safe_data))['dataset_path']
image_path = os.path.dirname(os.path.dirname(dataset_path))

def mindcube_doc_to_visual(doc):
    """Extract visual input from document"""
    # Based on api_eval.py, images are stored as "task_type/{response_id.split('_')[0]}.png"
    id = doc['id']
    for img_file in doc["images"]:
        img_full_path = os.path.join(image_path, img_file)
        if os.path.exists(img_full_path):
            return [Image.open(img_full_path).convert("RGB")]
    return []

def mindcube_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format the question and options as text input"""
    question = doc["input_prompt"]
    full_text = question
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        full_text = f"{pre_prompt}{question}{post_prompt}"
    return full_text

def mindcube_process_results(doc, results):
    """Process results to extract answers"""
    cogmap_results = _initialize_cogmap_results_structure()
    setting = get_setting_from_id(doc['id'])
    cogmap_results['settings'][setting]['total'] += 1
    include_in_overall = cogmap_results['settings'][setting].get('include_in_overall', True)

    cogmap_answer = results[0]  # Use the original results parameter
    extracted_answer = extract_answer(cogmap_answer)
    if not extracted_answer:
        eval_logger.warning(f"No valid answer found in response: {cogmap_answer}")
    is_correct = extracted_answer == doc['gt_answer']

    generated_cogmap = _extract_cognitive_map(cogmap_answer)
    grounded_cogmap = _extract_grounded_cogmap(doc)
    if generated_cogmap and grounded_cogmap:
        similarity = calculate_cogmap_similarity(generated_cogmap, grounded_cogmap)
        total_similarity_metrics = _initialize_similarity_accumulators()
        _update_similarity_metrics(similarity, cogmap_results, setting, total_similarity_metrics, include_in_overall)

    cogmap_results = _preserve_necessary_cogmap_fields(cogmap_results)
    return_results = {
        "accuracy": {
            "setting": setting, 
            "correct": True if cogmap_results['gen_cogmap_correct'] > 0 else False,
        },
        f"graph_metrics_{setting}":{
            "setting": setting,
            "include_in_overall": include_in_overall,
            "correct": True if cogmap_results['settings'][setting]['gen_cogmap_correct'] > 0 else False,
            "parsable_json": True if cogmap_results['settings'][setting]['cogmap_similarity']["parsable_json_count"] > 0 else False,
            "valid format": True if cogmap_results['settings'][setting]['cogmap_similarity']["valid_format_count"] > 0 else False,
            "valid graph": True if cogmap_results['settings'][setting]['cogmap_similarity']["total_valid"] > 0 else False,
            "rotation_invariant_isomorphic": True if cogmap_results["settings"][setting]['cogmap_similarity']["rotation_invariant_isomorphic_count"] > 0 else False,
            "relative_position_accuracy": cogmap_results['settings'][setting]['cogmap_similarity']["avg_relative_position_accuracy"],
            "facing_similarity": cogmap_results['settings'][setting]['cogmap_similarity']["avg_facing_similarity"],
            "directional_similarity": cogmap_results['settings'][setting]['cogmap_similarity']["avg_directional_similarity"],
            "overall_similarity": cogmap_results['settings'][setting]['cogmap_similarity']["avg_overall_similarity"],
            "best_rotation": cogmap_results['settings'][setting]['cogmap_similarity'].get("best_rotation", 0.0),
        },
        "graph_metrics_average":{
            "setting": setting, 
            "include_in_overall": include_in_overall,
            "correct": True if cogmap_results['settings'][setting]['gen_cogmap_correct'] > 0 else False,
            "parsable_json": True if cogmap_results['settings'][setting]['cogmap_similarity']["parsable_json_count"] > 0 else False,
            "valid format": True if cogmap_results['settings'][setting]['cogmap_similarity']["valid_format_count"] > 0 else False,
            "valid graph": True if cogmap_results['settings'][setting]['cogmap_similarity']["total_valid"] > 0 else False,
            "rotation_invariant_isomorphic": True if cogmap_results["settings"][setting]['cogmap_similarity']["rotation_invariant_isomorphic_count"] > 0 else False,
            "relative_position_accuracy": cogmap_results['settings'][setting]['cogmap_similarity']["avg_relative_position_accuracy"],
            "facing_similarity": cogmap_results['settings'][setting]['cogmap_similarity']["avg_facing_similarity"],
            "directional_similarity": cogmap_results['settings'][setting]['cogmap_similarity']["avg_directional_similarity"],
            "overall_similarity": cogmap_results['settings'][setting]['cogmap_similarity']["avg_overall_similarity"],
            "best_rotation": cogmap_results['settings'][setting]['cogmap_similarity'].get("best_rotation", 0.0),
        },
    }
    return return_results


def mindcube_aggregate_accuracy(results):
    """Aggregate accuracy results"""
    total_correct = sum(res['correct'] for res in results)
    total_count = len(results)
    
    if total_count == 0:
        return {"accuracy": 0.0, "total": 0}
    
    accuracy = total_correct / total_count
    return {"accuracy": accuracy, "total": total_count}

def mindcube_aggregate_graph_metrics_average(results):
    """Aggregate graph metrics results across all settings to compute overall averages"""
    if not results:
        return {
            "accuracy": 0.0,
            "parsable_json_rate": 0.0,
            "valid_format_rate": 0.0,
            "valid_graph_rate": 0.0,
            "rotation_invariant_isomorphic_rate": 0.0,
            "relative_position_accuracy": 0.0,
            "facing_similarity": 0.0,
            "directional_similarity": 0.0,
            "overall_similarity": 0.0,
            "best_rotation": 0.0,
            "total": 0
        }
    
    # 只考虑include_in_overall为True的样本
    filtered_results = [res for res in results if res.get('include_in_overall', True)]
    
    if not filtered_results:
        return {
            "accuracy": 0.0,
            "parsable_json_rate": 0.0,
            "valid_format_rate": 0.0,
            "valid_graph_rate": 0.0,
            "rotation_invariant_isomorphic_rate": 0.0,
            "relative_position_accuracy": 0.0,
            "facing_similarity": 0.0,
            "directional_similarity": 0.0,
            "overall_similarity": 0.0,
            "best_rotation": 0.0,
            "total": 0
        }
    
    total_count = len(filtered_results)
    
    # 计算布尔指标的比例
    correct_count = sum(1 for res in filtered_results if res.get('correct', False))
    parsable_json_count = sum(1 for res in filtered_results if res.get('parsable_json', False))
    valid_format_count = sum(1 for res in filtered_results if res.get('valid format', False))
    valid_graph_count = sum(1 for res in filtered_results if res.get('valid graph', False))
    rotation_invariant_isomorphic_count = sum(1 for res in filtered_results if res.get('rotation_invariant_isomorphic', False))
    
    # 计算数值指标的平均值
    relative_position_accuracy_sum = sum(res.get('relative_position_accuracy', 0.0) for res in filtered_results)
    facing_similarity_sum = sum(res.get('facing_similarity', 0.0) for res in filtered_results)
    directional_similarity_sum = sum(res.get('directional_similarity', 0.0) for res in filtered_results)
    overall_similarity_sum = sum(res.get('overall_similarity', 0.0) for res in filtered_results)
    best_rotation_sum = sum(res.get('best_rotation', 0.0) for res in filtered_results)
    
    return {
        "accuracy": correct_count / total_count,
        "parsable_json_rate": parsable_json_count / total_count,
        "valid_format_rate": valid_format_count / total_count,
        "valid_graph_rate": valid_graph_count / total_count,
        "rotation_invariant_isomorphic_rate": rotation_invariant_isomorphic_count / total_count,
        "relative_position_accuracy": relative_position_accuracy_sum / total_count,
        "facing_similarity": facing_similarity_sum / total_count,
        "directional_similarity": directional_similarity_sum / total_count,
        "overall_similarity": overall_similarity_sum / total_count,
        "best_rotation": best_rotation_sum / total_count,
        "total": total_count
    }


def mindcube_aggregate_graph_metrics(results):
    """Aggregate graph metrics results for a specific setting"""
    if not results:
        return {
            "accuracy": 0.0,
            "parsable_json_rate": 0.0,
            "valid_format_rate": 0.0,
            "valid_graph_rate": 0.0,
            "rotation_invariant_isomorphic_rate": 0.0,
            "relative_position_accuracy": 0.0,
            "facing_similarity": 0.0,
            "directional_similarity": 0.0,
            "overall_similarity": 0.0,
            "best_rotation": 0.0,
            "total": 0
        }
    
    total_count = len(results)
    
    # 计算布尔指标的比例
    correct_count = sum(1 for res in results if res.get('correct', False))
    parsable_json_count = sum(1 for res in results if res.get('parsable_json', False))
    valid_format_count = sum(1 for res in results if res.get('valid format', False))
    valid_graph_count = sum(1 for res in results if res.get('valid graph', False))
    rotation_invariant_isomorphic_count = sum(1 for res in results if res.get('rotation_invariant_isomorphic', False))
    
    # 计算数值指标的平均值
    relative_position_accuracy_sum = sum(res.get('relative_position_accuracy', 0.0) for res in results)
    facing_similarity_sum = sum(res.get('facing_similarity', 0.0) for res in results)
    directional_similarity_sum = sum(res.get('directional_similarity', 0.0) for res in results)
    overall_similarity_sum = sum(res.get('overall_similarity', 0.0) for res in results)
    best_rotation_sum = sum(res.get('best_rotation', 0.0) for res in results)
    
    return {
        "accuracy": correct_count / total_count,
        "parsable_json_rate": parsable_json_count / total_count,
        "valid_format_rate": valid_format_count / total_count,
        "valid_graph_rate": valid_graph_count / total_count,
        "rotation_invariant_isomorphic_rate": rotation_invariant_isomorphic_count / total_count,
        "relative_position_accuracy": relative_position_accuracy_sum / total_count,
        "facing_similarity": facing_similarity_sum / total_count,
        "directional_similarity": directional_similarity_sum / total_count,
        "overall_similarity": overall_similarity_sum / total_count,
        "best_rotation": best_rotation_sum / total_count,
        "total": total_count
    }
