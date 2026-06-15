import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
import soundfile as sf

from semantic_asr.api import list_models
from semantic_asr_service.artifacts import artifact_path, artifact_urls
from semantic_asr_service.schemas import JobCreateResponse, JobStatusResponse
from semantic_asr_service.settings import ServiceSettings, load_settings
from semantic_asr_service.store import JobStore


def create_app(
    settings: ServiceSettings | None = None,
    store: JobStore | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    store = store or JobStore(settings.db_path)
    os.makedirs(settings.upload_root, exist_ok=True)
    os.makedirs(settings.jobs_root, exist_ok=True)

    app = FastAPI(title="Semantic ASR Service")
    app.state.settings = settings
    app.state.store = store

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/jobs", response_model=JobCreateResponse)
    def create_job(
        audio: UploadFile = File(...),
        config: str = Form(...),
        formats: str = Form("json,srt,csv,textgrid"),
        user=Depends(_require_user),
    ):
        _require_allowed_config(settings, config)
        selected_formats = _parse_formats(formats)
        suffix = Path(audio.filename or "").suffix.lower()
        if suffix not in settings.allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {suffix}")

        job_id = _new_job_id()
        upload_dir = os.path.join(settings.upload_root, job_id)
        outdir = os.path.join(settings.jobs_root, job_id, "outputs")
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        wav_path = os.path.join(upload_dir, f"input{suffix}")
        _save_upload(audio, wav_path, settings.max_upload_bytes)
        _validate_audio_duration(wav_path, settings.max_audio_seconds)

        job = store.create_job(
            job_id=job_id,
            user_id=user["user_id"],
            config=config,
            wav_path=wav_path,
            outdir=outdir,
            formats=selected_formats,
        )
        return {"job_id": job["job_id"], "status": job["status"]}

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(job_id: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        return _job_response(job)

    @app.get("/v1/jobs/{job_id}/result")
    def get_result(job_id: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        path = artifact_path(job["outdir"], job_id, "json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result JSON is not available")
        return FileResponse(path, media_type="application/json", filename=f"{job_id}.json")

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_format}")
    def get_artifact(job_id: str, artifact_format: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        try:
            path = artifact_path(job["outdir"], job_id, artifact_format)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Artifact is not available")
        return FileResponse(path, filename=os.path.basename(path))

    @app.get("/v1/models")
    def get_models(language: str, role: str | None = None, user=Depends(_require_user)):
        return list_models(language, role=role)

    return app


def _require_user(request: Request, authorization: str | None = Header(default=None)):
    settings = request.app.state.settings
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    user_id = settings.api_keys.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return {"user_id": user_id, "token": token, "is_admin": token in settings.admin_tokens}


def _get_authorized_job(store: JobStore, job_id: str, user: dict) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not user["is_admin"] and job["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _job_response(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "config": job["config"],
        "progress": job["progress"],
        "error": job.get("error"),
        "artifacts": artifact_urls(job["job_id"], job["formats"]) if job["status"] == "succeeded" else {},
    }


def _require_allowed_config(settings: ServiceSettings, config: str) -> None:
    try:
        settings.resolve_config_path(config)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_formats(formats: str) -> list[str]:
    selected = [item.strip().lower() for item in formats.split(",") if item.strip()]
    allowed = {"json", "srt", "csv", "textgrid"}
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown formats: {', '.join(unknown)}")
    selected = selected or ["json"]
    if "json" not in selected:
        selected.insert(0, "json")
    return selected


def _save_upload(upload: UploadFile, output_path: str, max_bytes: int) -> None:
    written = 0
    with open(output_path, "wb") as fout:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                fout.close()
                os.unlink(output_path)
                raise HTTPException(status_code=413, detail="Uploaded file is too large")
            fout.write(chunk)


def _validate_audio_duration(wav_path: str, max_audio_seconds: float) -> None:
    if max_audio_seconds <= 0:
        return
    try:
        info = sf.info(wav_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read audio metadata: {exc}") from exc
    if info.duration > max_audio_seconds:
        os.unlink(wav_path)
        raise HTTPException(status_code=413, detail="Uploaded audio is too long")


def _new_job_id() -> str:
    return uuid.uuid4().hex


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("semantic_asr_service.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
