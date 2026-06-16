import argparse
import os

from semantic_asr_service.settings import ServiceSettings, load_settings
from semantic_asr_service.store import JobStore
from semantic_asr_service.translation import translate_job_result, translation_cache_path


def find_missing_translation_jobs(
    settings: ServiceSettings,
    store: JobStore,
    target_language: str,
    limit: int = 0,
) -> list[dict]:
    missing = []
    offset = 0
    page_size = 500
    while True:
        jobs = store.list_jobs(
            user_id="",
            include_all=True,
            limit=page_size,
            offset=offset,
            include_internal=True,
        )
        if not jobs:
            break
        for job in jobs:
            if job["status"] != "succeeded":
                continue
            cache_path = translation_cache_path(job["outdir"], target_language)
            if os.path.exists(cache_path):
                continue
            result_path = os.path.join(job["outdir"], f"{job['job_id']}.json")
            if not os.path.exists(result_path):
                continue
            missing.append(job)
            if limit and len(missing) >= limit:
                return missing
        offset += page_size
    return missing


def backfill_translations(
    settings: ServiceSettings,
    target_language: str,
    limit: int = 0,
    dry_run: bool = False,
) -> tuple[int, int]:
    if target_language not in settings.translation_targets:
        raise ValueError(f"Translation target is not allowed: {target_language}")
    store = JobStore(settings.db_path)
    jobs = find_missing_translation_jobs(settings, store, target_language, limit=limit)
    if dry_run:
        for job in jobs:
            print(f"[backfill] missing job_id={job['job_id']} config={job['config']}")
        print(f"[backfill] dry_run missing={len(jobs)}")
        return len(jobs), 0

    succeeded = 0
    failed = 0
    for job in jobs:
        print(f"[backfill] translating job_id={job['job_id']} config={job['config']} target={target_language}", flush=True)
        try:
            translate_job_result(job, settings, target_language)
            succeeded += 1
            print(f"[backfill] done job_id={job['job_id']}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[backfill] failed job_id={job['job_id']} error={exc}", flush=True)
    print(f"[backfill] complete succeeded={succeeded} failed={failed} total={len(jobs)}", flush=True)
    return succeeded, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="zh_cn")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backfill_translations(
        load_settings(),
        target_language=args.target,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
