import os


ARTIFACT_PATHS = {
    "json": lambda outdir, job_id: os.path.join(outdir, f"{job_id}.json"),
    "srt": lambda outdir, job_id: os.path.join(outdir, "asr_srt", f"{job_id}.srt"),
    "csv": lambda outdir, job_id: os.path.join(outdir, "asr_csv", f"{job_id}.csv"),
    "textgrid": lambda outdir, job_id: os.path.join(outdir, "asr_tg", f"{job_id}.TextGrid"),
}


def artifact_path(outdir: str, job_id: str, artifact_format: str) -> str:
    try:
        builder = ARTIFACT_PATHS[artifact_format]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact format: {artifact_format}") from exc
    return builder(outdir, job_id)


def artifact_urls(job_id: str, formats: list[str]) -> dict[str, str]:
    return {
        item: f"/v1/jobs/{job_id}/artifacts/{item}"
        for item in formats
        if item in ARTIFACT_PATHS
    }

