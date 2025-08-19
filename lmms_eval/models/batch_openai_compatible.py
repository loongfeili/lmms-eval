""""
Batch inference calls to OpenAI-compatible APIs.
"""
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from dotenv import find_dotenv, load_dotenv
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from PIL import Image

load_dotenv(verbose=True)


@register_model("batch_openai_compatible")
class BatchOpenAICompatible(lmms):
    def __init__(
        self,
        model_version: str = "grok-2-latest",
        timeout: int = 10,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        azure_openai: bool = False,
        max_frames_num: int = 16,
        batch_size: int = 8,
        threads: int = 16,  # Threads to use for parallel processing
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb  # some models have a limit on the size of the image
        self.continual_mode = continual_mode
        self.max_frames_num = max_frames_num
        self.threads = threads
        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        self.client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
            if not azure_openai
            else AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
        )

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        max_size = self.max_size_in_mb * 1024 * 1024  # 20MB in bytes
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        # If image is too large, resize it while maintaining aspect ratio
        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def process_single_request(self, contexts, gen_kwargs, doc_to_visual, doc_id, task, split):
        """Process a single request and return payload"""
        # Check cache first if continual mode is enabled
        if self.continual_mode is True and self.cache_mode == "resume":
            doc_uuid = f"{task}___{split}___{doc_id}"
            if doc_uuid in self.response_cache:
                response_text = self.response_cache[doc_uuid]
                if response_text:
                    return None, response_text, doc_uuid  # Return cached result

        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
        if None in visuals:
            visuals = []
            imgs = []
        else:
            visuals = self.flatten(visuals)
            imgs = []  # multiple images or frames for video
            
            # Use thread pool for parallel visual encoding
            all_tasks = []
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                for visual in visuals:
                    if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                        all_tasks.append(executor.submit(self.encode_video, visual, self.max_frames_num))
                    elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                        all_tasks.append(executor.submit(self.encode_image, visual))
                    elif isinstance(visual, Image.Image):
                        all_tasks.append(executor.submit(self.encode_image, visual))

                for task in all_tasks:
                    result = task.result()
                    if isinstance(result, list):  # Video frames
                        imgs.extend(result)
                    else:  # Single image
                        imgs.append(result)

        payload = {"messages": []}
        payload["model"] = self.model_version

        # When there is no image token in the context, append the image to the text
        payload["messages"].append({"role": "user", "content": []})
        payload["messages"][0]["content"].append({"type": "text", "text": contexts})
        for img in imgs:
            payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if gen_kwargs["max_new_tokens"] > 4096:
            gen_kwargs["max_new_tokens"] = 4096
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        payload["max_tokens"] = gen_kwargs["max_new_tokens"]
        payload["temperature"] = gen_kwargs["temperature"]

        if "o1" in self.model_version or "o3" in self.model_version:
            del payload["temperature"]
            payload["reasoning_effort"] = "medium"
            payload["response_format"] = {"type": "text"}
            payload.pop("max_tokens")
            payload["max_completion_tokens"] = gen_kwargs["max_new_tokens"]

        doc_uuid = f"{task}___{split}___{doc_id}" if self.continual_mode else None
        return payload, None, doc_uuid

    def call_api_batch(self, payloads):
        """Call API for a batch of payloads concurrently"""
        responses = []
        
        def call_single_api(payload):
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(**payload)
                    return response.choices[0].message.content
                except Exception as e:
                    error_msg = str(e)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                    eval_logger.debug(f"Full error details: {type(e).__name__}: {e}")
                    eval_logger.debug(f"API base URL: {getattr(self.client, 'base_url', 'Not set')}")
                    
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                        return ""
                    else:
                        time.sleep(self.timeout)
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=min(len(payloads), 10)) as executor:
            future_to_index = {executor.submit(call_single_api, payload): i for i, payload in enumerate(payloads)}
            responses = [None] * len(payloads)
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    response_text = future.result()
                    responses[index] = response_text
                except Exception as e:
                    eval_logger.error(f"API call failed: {e}")
                    responses[index] = ""
        
        return responses

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        
        for batch_requests in batched_requests:
            payloads = []
            cached_responses = []
            doc_uuids = []
            
            # Process each request in the batch
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments
                payload, cached_response, doc_uuid = self.process_single_request(
                    contexts, gen_kwargs, doc_to_visual, doc_id, task, split
                )
                
                if cached_response is not None:
                    # Use cached response
                    cached_responses.append((idx, cached_response, doc_uuid))
                else:
                    # Need to call API
                    payloads.append((idx, payload, doc_uuid))
            
            # Call API for non-cached requests
            batch_responses = [None] * len(batch_requests)
            
            # Set cached responses
            for idx, cached_response, doc_uuid in cached_responses:
                batch_responses[idx] = cached_response
            
            # Call API for remaining requests
            if payloads:
                api_payloads = [p[1] for p in payloads]  # Extract payloads
                api_responses = self.call_api_batch(api_payloads)
                
                # Map responses back to original positions
                for (idx, payload, doc_uuid), response_text in zip(payloads, api_responses):
                    batch_responses[idx] = response_text
                    
                    # Cache the response if continual mode is enabled
                    if self.continual_mode is True and doc_uuid:
                        self.response_cache[doc_uuid] = response_text
                        with open(self.response_persistent_file, "w") as f:
                            json.dump(self.response_cache, f)
            
            res.extend(batch_responses)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for OpenAI compatible models")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for OpenAI compatible models")
