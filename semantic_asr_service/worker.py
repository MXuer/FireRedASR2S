import argparse
import os
import time
import traceback
from typing import Callable

from semantic_asr.api import SemanticASR
from semantic_asr_service.settings import ServiceSettings, load_settings
from semantic_asr_service.store import JobStore


Runner = Callable[[dict, ServiceSettings], None]


def run_worker_once(
    settings: ServiceSettings,
    store: JobStore,
    runner: Runner | None = None,
) -> dict | None:
    job = store.claim_next_job()
    if job is None:
        return None
    runner = runner or run_job
    try:
        runner(job, settings)
        store.mark_succeeded(job["job_id"])
    except Exception:
        store.mark_failed(job["job_id"], traceback.format_exc())
    return store.get_job(job["job_id"])


def run_job(job: dict, settings: ServiceSettings) -> None:
    config_path = settings.resolve_config_path(job["config"])
    sdk = SemanticASR.from_config(config_path)
    sdk.transcribe(
        wav_path=job["wav_path"],
        uttid=job["job_id"],
        outdir=job["outdir"],
        formats=job["formats"],
        use_cache=False,
    )


def worker_loop(settings: ServiceSettings, device: str | None = None) -> None:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    store = JobStore(settings.db_path)
    while True:
        job = run_worker_once(settings, store)
        if job is None:
            time.sleep(settings.poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    worker_loop(load_settings(), device=args.device)


if __name__ == "__main__":
    main()
