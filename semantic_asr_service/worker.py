import argparse
import os
import time
import traceback
from typing import Callable

from semantic_asr_service.settings import ServiceSettings, load_settings
from semantic_asr_service.store import JobStore


Runner = Callable[[dict, ServiceSettings], None]


def run_worker_once(
    settings: ServiceSettings,
    store: JobStore,
    runner: Runner | None = None,
) -> dict | None:
    job = store.claim_next_job(stale_after_s=settings.stale_running_seconds)
    if job is None:
        return None
    _log(f"claimed job_id={job['job_id']} config={job['config']} wav={job['wav_path']}")
    runner = runner or run_job
    try:
        runner(job, settings)
        progress = {"stage": "done"}
        try:
            _auto_translate(job, settings)
        except Exception:
            progress = {"stage": "done", "translation_error": traceback.format_exc()}
            _log(f"auto translation failed job_id={job['job_id']}\n{progress['translation_error']}")
        store.mark_succeeded(job["job_id"], progress=progress)
        _log(f"succeeded job_id={job['job_id']}")
    except Exception:
        error = traceback.format_exc()
        store.mark_failed(job["job_id"], error)
        _log(f"failed job_id={job['job_id']}\n{error}")
    return store.get_job(job["job_id"])


def run_job(job: dict, settings: ServiceSettings) -> None:
    from semantic_asr.api import SemanticASR

    config_path = settings.resolve_config_path(job["config"])
    _log(f"loading pipeline job_id={job['job_id']} config_path={config_path}")
    sdk = SemanticASR.from_config(config_path)
    _log(f"running pipeline job_id={job['job_id']}")
    sdk.transcribe(
        wav_path=job["wav_path"],
        uttid=job["job_id"],
        outdir=job["outdir"],
        formats=job["formats"],
        use_cache=True,
    )


def _auto_translate(job: dict, settings: ServiceSettings) -> None:
    targets = sorted(settings.auto_translate_targets)
    if not targets:
        return
    if not settings.translation_base_url:
        raise RuntimeError("Auto translation is enabled but SEMANTIC_ASR_TRANSLATION_BASE_URL is not configured")
    from semantic_asr_service.translation import translate_job_result

    for target in targets:
        if target not in settings.translation_targets:
            raise ValueError(f"Auto translation target is not allowed: {target}")
        _log(f"auto translating job_id={job['job_id']} target={target}")
        translate_job_result(job, settings, target)
        _log(f"auto translated job_id={job['job_id']} target={target}")


def worker_loop(settings: ServiceSettings, device: str | None = None) -> None:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    store = JobStore(settings.db_path)
    _log(
        f"worker started db={settings.db_path} device={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"poll_interval_s={settings.poll_interval_s}"
    )
    idle_logged = False
    while True:
        job = run_worker_once(settings, store)
        if job is None:
            if not idle_logged:
                _log("idle: no queued jobs")
                idle_logged = True
            time.sleep(settings.poll_interval_s)
        else:
            idle_logged = False


def _log(message: str) -> None:
    print(f"[semantic-asr-worker] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    worker_loop(load_settings(), device=args.device)


if __name__ == "__main__":
    main()
