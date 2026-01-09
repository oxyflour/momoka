import asyncio
import json
import os
import sys
import time

import torch
import websockets
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def build_dummy_batch(policy: PI0Policy, task: str) -> dict:
    batch: dict[str, object] = {"task": task}
    image_features = policy.config.image_features
    if not image_features:
        raise ValueError("pi0 config has no image features; cannot build a dummy observation.")

    for key, feature in policy.config.input_features.items():
        shape = tuple(feature.shape)
        if feature.type is FeatureType.VISUAL:
            batch[key] = torch.zeros(shape, dtype=torch.float32)
        elif feature.type in (FeatureType.STATE, FeatureType.ENV):
            batch[key] = torch.zeros(shape, dtype=torch.float32)

    return batch


def infer_once(
    policy: PI0Policy,
    preprocessor,
    postprocessor,
    task: str,
) -> dict:
    batch = build_dummy_batch(policy, task)
    processed = preprocessor(batch)
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(processed)

    # Postprocess the first action in the chunk.
    first_action = action_chunk[:, 0, :]
    action = postprocessor(first_action).squeeze(0)
    return {
        "action": action.tolist(),
        "chunk_size": int(action_chunk.shape[1]),
        "action_dim": int(action.shape[0]),
    }


async def main() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    uri = os.environ.get("LEROBOT_WS_URI", "ws://127.0.0.1:8765")
    model_id = os.environ.get("LEROBOT_PI0_MODEL_ID", "lerobot/pi0_libero")

    task = os.environ.get("LEROBOT_TASK", "place the object in the box")
    policy = PI0Policy.from_pretrained(model_id)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=model_id,
        preprocessor_overrides={"device_processor": {"device": policy.config.device}},
        postprocessor_overrides={"device_processor": {"device": policy.config.device}},
    )

    async with websockets.connect(uri) as websocket:
        while True:
            result = infer_once(policy, preprocessor, postprocessor, task)
            payload = {
                "ts": time.time(),
                "model_id": model_id,
                "task": task,
                "result": result,
            }
            await websocket.send(json.dumps(payload, ensure_ascii=True))
            print("sent:", payload)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
