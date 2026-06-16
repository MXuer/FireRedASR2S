import argparse
import json
import os
import shutil
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "hunyuan-mt"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.6
    max_tokens: int = 512
    repetition_penalty: float = 1.05


def prepare_fp8_model_dir(source: str, destination: str) -> str:
    src = Path(source)
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists() or target.is_symlink():
            continue
        real = item.resolve()
        if item.name.endswith(".safetensors"):
            target.symlink_to(real)
        else:
            shutil.copyfile(real, target)

    config_path = dst / "config.json"
    with config_path.open(encoding="utf-8") as fin:
        config = json.load(fin)
    quant_config = config.get("quantization_config") or {}
    if "ignored_layers" in quant_config:
        quant_config["ignore"] = quant_config.pop("ignored_layers")
    for group in (quant_config.get("config_groups") or {}).values():
        if "ignored_layers" in group:
            group["ignore"] = group.pop("ignored_layers")
    with config_path.open("w", encoding="utf-8") as fout:
        json.dump(config, fout, ensure_ascii=False, indent=2)
    return str(dst)


def create_app(model_path: str, served_model_name: str = "hunyuan-mt", max_concurrent: int = 1) -> FastAPI:
    app = FastAPI(title="Hunyuan-MT OpenAI Compatible Server")
    app.state.served_model_name = served_model_name
    app.state.tokenizer = None
    app.state.model = None
    app.state.model_path = model_path
    app.state.generate_semaphore = threading.BoundedSemaphore(max(1, int(max_concurrent)))

    @app.on_event("startup")
    def load_model() -> None:
        app.state.tokenizer = PreTrainedTokenizerFast.from_pretrained(app.state.model_path)
        app.state.model = AutoModelForCausalLM.from_pretrained(
            app.state.model_path,
            device_map="auto",
            trust_remote_code=True,
        )
        app.state.model.eval()

    @app.get("/health")
    def health():
        return {"ok": app.state.model is not None}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        with app.state.generate_semaphore:
            prompt_messages = [message.model_dump() for message in request.messages]
            inputs = app.state.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(app.state.model.device)
            with torch.no_grad():
                outputs = app.state.model.generate(
                    inputs,
                    max_new_tokens=request.max_tokens,
                    top_p=request.top_p,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.temperature > 0,
                    eos_token_id=app.state.tokenizer.eos_token_id,
                )
        text = app.state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        created = int(time.time())
        return {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": app.state.served_model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runtime-model-path", default="")
    parser.add_argument("--served-model-name", default="hunyuan-mt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10087)
    parser.add_argument("--max-concurrent", type=int, default=int(os.environ.get("HUNYUAN_MT_MAX_CONCURRENT", "4")))
    args = parser.parse_args()

    model_path = args.model_path
    if args.runtime_model_path:
        model_path = prepare_fp8_model_dir(args.model_path, args.runtime_model_path)

    import uvicorn

    app = create_app(model_path=model_path, served_model_name=args.served_model_name, max_concurrent=args.max_concurrent)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
