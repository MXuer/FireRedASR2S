import argparse
import json
import logging
import multiprocessing as mp
import os
import queue as queue_module
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger("semantic_asr.run_batch")


@dataclass
class BatchItem:
    wav_path: str
    uttid: str


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_seconds", type=float, default=0)
    args = parser.parse_args()

    outputs = run_batch(
        config_path=args.config,
        wav_scp=args.wav_scp,
        outdir=args.outdir,
        num_workers=args.num_workers,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def run_batch(
    config_path: str,
    wav_scp: str,
    outdir: str,
    num_workers: int = 1,
    max_seconds: float = 0,
) -> list[dict]:
    items = read_wav_scp(wav_scp)
    if not items:
        return []
    devices = visible_cuda_devices()
    worker_count = effective_worker_count(num_workers, devices, len(items))
    worker_devices = assign_worker_devices(worker_count, devices)
    chunks = split_round_robin(items, worker_count)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    for worker_index, (chunk, device) in enumerate(zip(chunks, worker_devices)):
        process = ctx.Process(
            target=_worker_main,
            kwargs={
                "worker_index": worker_index,
                "device": device,
                "items": chunk,
                "config_path": config_path,
                "outdir": outdir,
                "max_seconds": max_seconds,
                "queue": queue,
            },
            name=f"semantic-asr-worker-{worker_index}",
        )
        process.start()
        processes.append(process)

    results = []
    pending = len(processes)
    while pending:
        try:
            message = queue.get(timeout=1.0)
        except queue_module.Empty:
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join()
                failed_indexes = [process.name for process in failed]
                raise RuntimeError(f"Batch worker process failed before reporting: failed workers={failed_indexes}")
            continue
        pending -= 1
        if message["ok"]:
            results.extend(message["outputs"])
        else:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise RuntimeError(message["error"])
    for process in processes:
        process.join()
    failed = [process for process in processes if process.exitcode != 0]
    if failed:
        failed_indexes = [process.name for process in failed]
        raise RuntimeError(f"Batch worker process failed: failed workers={failed_indexes}")
    return sorted(results, key=lambda item: item["uttid"])


def read_wav_scp(path: str) -> list[BatchItem]:
    items = []
    with open(path, encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                wav_path = parts[0]
                uttid = os.path.splitext(os.path.basename(wav_path))[0]
            else:
                uttid, wav_path = parts
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"{path}:{line_number}: wav path does not exist: {wav_path}")
            items.append(BatchItem(wav_path=wav_path, uttid=uttid))
    seen = set()
    for item in items:
        if item.uttid in seen:
            raise ValueError(f"{path}: duplicate uttid: {item.uttid}")
        seen.add(item.uttid)
    return items


def visible_cuda_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return [str(index) for index in range(8)]
    return [item.strip() for item in raw.split(",") if item.strip()]


def assign_worker_devices(worker_count: int, devices: list[str]) -> list[str]:
    if not devices:
        devices = [str(index) for index in range(8)]
    return [devices[index % len(devices)] for index in range(worker_count)]


def effective_worker_count(num_workers_per_device: int, devices: list[str], item_count: int) -> int:
    device_count = max(1, len(devices))
    requested = int(num_workers_per_device) * device_count
    return max(1, min(requested, item_count))


def split_round_robin(items: list[BatchItem], worker_count: int) -> list[list[BatchItem]]:
    chunks = [[] for _ in range(worker_count)]
    for index, item in enumerate(items):
        chunks[index % worker_count].append(item)
    return chunks


def _worker_main(
    worker_index: int,
    device: str,
    items: list[BatchItem],
    config_path: str,
    outdir: str,
    max_seconds: float,
    queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    try:
        outputs = run_batch_items(
            items=items,
            config_path=config_path,
            outdir=outdir,
            max_seconds=max_seconds,
            device=device,
            worker_index=worker_index,
        )
        queue.put({"ok": True, "outputs": outputs})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def run_batch_items(
    items: list[BatchItem],
    config_path: str,
    outdir: str,
    max_seconds: float,
    device: str,
    worker_index: int = 0,
    runner: Callable[..., dict] | None = None,
) -> list[dict]:
    if runner is None:
        from semantic_asr.api import SemanticASR

        sdk = SemanticASR.from_config(config_path)

        def runner(**kwargs):
            return sdk.transcribe(
                wav_path=kwargs["wav_path"],
                uttid=kwargs["uttid"],
                outdir=kwargs["outdir"],
                max_seconds=kwargs["max_seconds"],
            )["outputs"]

    outputs = []
    for item in items:
        logger.info("worker=%s device=%s uttid=%s wav=%s", worker_index, device, item.uttid, item.wav_path)
        try:
            result = runner(
                config_path=config_path,
                wav_path=item.wav_path,
                uttid=item.uttid,
                outdir=outdir,
                max_seconds=max_seconds,
            )
            outputs.append({"uttid": item.uttid, "device": device, "outputs": result, "ok": True})
        except Exception as exc:
            logger.exception("worker=%s item failed uttid=%s wav=%s", worker_index, item.uttid, item.wav_path)
            error_path = write_error_json(outdir, item, exc)
            outputs.append(
                {
                    "uttid": item.uttid,
                    "device": device,
                    "ok": False,
                    "error": repr(exc),
                    "error_json": error_path,
                }
            )
    return outputs


def write_error_json(outdir: str, item: BatchItem, exc: Exception) -> str:
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, f"{item.uttid}.error.json")
    payload = {
        "uttid": item.uttid,
        "wav_path": item.wav_path,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
    return output_path


if __name__ == "__main__":
    main()
