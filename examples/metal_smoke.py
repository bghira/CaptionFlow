"""Run real image captioning through CaptionFlow on the vLLM Metal backend."""

import argparse
import json
import os
import time
from importlib.metadata import version

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-2B-Instruct-4bit")
    args = parser.parse_args()

    os.environ["VLLM_METAL_BACKEND"] = "mlx"
    os.environ["VLLM_METAL_MULTIMODAL_MODE"] = "auto"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import mlx.core as mx
    from vllm.platforms import current_platform

    from caption_flow.workers.caption import CaptionWorker, ProcessingItem

    if not type(current_platform).__module__.startswith("vllm_metal"):
        raise RuntimeError(f"Expected vLLM Metal, got {type(current_platform)}")
    if mx.default_device() != mx.gpu:
        raise RuntimeError("MLX is not using the GPU")

    worker = CaptionWorker(
        {"server": "ws://127.0.0.1:8765", "token": "local-smoke", "batch_image_processing": False}
    )
    worker.vllm_config = {
        "model": args.model,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.1,
        "dtype": "float16",
        "inference_prompts": ["Describe the shapes and their colors in one short sentence."],
        "sampling": {"temperature": 0.0, "max_tokens": 80},
    }
    worker.stages = worker._parse_stages_config(worker.vllm_config)
    worker.stage_order = worker._topological_sort_stages(worker.stages)

    colors = [("red", "blue"), ("green", "yellow")]
    batch = []
    for index, (square, circle) in enumerate(colors):
        image = Image.new("RGB", (384, 256), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((32, 64, 160, 192), fill=square)
        draw.ellipse((224, 64, 352, 192), fill=circle)
        batch.append(
            ProcessingItem(
                unit_id="metal-smoke",
                job_id=f"metal-smoke:chunk:0:idx:{index}",
                chunk_id="0",
                item_key=str(index),
                item_index=index,
                image=image,
                image_data=b"",
                metadata={},
            )
        )

    started = time.perf_counter()
    try:
        worker._setup_vllm()
        results = worker._process_batch_multi_stage(batch)
        if len(results) != len(batch) or worker.items_failed:
            raise RuntimeError(f"Captioning failed: {worker.items_failed} failed items")
        captions = []
        for item, fields in results:
            caption = " ".join(fields["captions"])
            expected = (*colors[item.item_index], "square", "circle")
            if not all(word in caption.lower() for word in expected):
                raise RuntimeError(f"Image content check failed: {caption!r}; expected {expected}")
            captions.append(caption)
        print(
            json.dumps(
                {
                    "device": mx.device_info()["device_name"],
                    "platform": type(current_platform).__name__,
                    "versions": {
                        name: version(name)
                        for name in ("caption-flow", "vllm", "vllm-metal", "torch", "mlx")
                    },
                    "model": args.model,
                    "captions": captions,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                },
                indent=2,
            )
        )
    finally:
        if worker.model_manager is not None:
            worker.model_manager.cleanup()


if __name__ == "__main__":
    main()
