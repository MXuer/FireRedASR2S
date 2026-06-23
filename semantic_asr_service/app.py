import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import soundfile as sf

from semantic_asr.api import list_models
from semantic_asr_service.artifacts import artifact_path, artifact_urls
from semantic_asr_service.demo import DEMO_HTML
from semantic_asr_service.schemas import JobCreateResponse, JobStatusResponse, TranslationCreateRequest
from semantic_asr_service.settings import ServiceSettings, load_settings
from semantic_asr_service.store import JobStore
from semantic_asr_service.translation import (
    HunyuanMTClient,
    stream_translate_job_result,
    translate_job_result,
    translation_cache_path,
)


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
    app.state.translation_client = None

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/demo", response_class=HTMLResponse)
    def demo():
        return DEMO_HTML

    @app.get("/v1/configs")
    def get_configs(user=Depends(_require_user)):
        return {"configs": sorted(settings.allowed_configs)}

    @app.get("/v1/translation-targets")
    def get_translation_targets(user=Depends(_require_user)):
        return {"targets": sorted(settings.translation_targets)}

    @app.get("/v1/me")
    def get_me(user=Depends(_require_user)):
        return {"user_id": user["user_id"], "is_admin": user["is_admin"]}

    @app.get("/v1/jobs")
    def list_jobs(
        limit: int = 10,
        offset: int = 0,
        config: str | None = None,
        user_id: str | None = None,
        user=Depends(_require_user),
    ):
        config_filter = _clean_optional_filter(config)
        user_filter = _clean_optional_filter(user_id)
        jobs = store.list_jobs(
            user["user_id"],
            include_all=user["is_admin"],
            limit=limit,
            offset=offset,
            config=config_filter,
            filter_user_id=user_filter,
        )
        total = store.count_jobs(
            user["user_id"],
            include_all=user["is_admin"],
            config=config_filter,
            filter_user_id=user_filter,
        )
        return {
            "jobs": [_job_response(job) for job in jobs],
            "total": total,
            "limit": max(1, min(500, int(limit))),
            "offset": max(0, int(offset)),
        }

    @app.post("/v1/jobs", response_model=JobCreateResponse)
    def create_job(
        audio: UploadFile = File(...),
        config: str = Form(...),
        formats: str = Form("json,srt,csv,textgrid"),
        local_path: str = Form(""),
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
            filename=audio.filename or "",
            local_path=local_path,
        )
        return {"job_id": job["job_id"], "status": job["status"]}

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(job_id: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        return _job_response(job)

    @app.delete("/v1/jobs/{job_id}")
    def delete_job(job_id: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        if job["status"] == "running":
            raise HTTPException(status_code=409, detail="Running jobs cannot be deleted")
        _delete_job_files(job, settings)
        store.delete_job(job_id)
        return {"job_id": job_id, "deleted": True}

    @app.post("/v1/jobs/{job_id}/retry")
    def retry_job(job_id: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        if job["status"] not in {"failed", "canceled"}:
            raise HTTPException(status_code=409, detail="Only failed or canceled jobs can be retried")
        if not store.retry_job(job_id):
            raise HTTPException(status_code=409, detail="Job could not be retried")
        retried = store.get_job(job_id)
        return _job_response(retried)

    @app.get("/v1/jobs/{job_id}/result")
    def get_result(job_id: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        path = artifact_path(job["outdir"], job_id, "json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result JSON is not available")
        return FileResponse(path, media_type="application/json", filename=f"{job_id}.json")

    @app.get("/v1/jobs/{job_id}/audio")
    def get_audio(job_id: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        if not os.path.exists(job["wav_path"]):
            raise HTTPException(status_code=404, detail="Uploaded audio is not available")
        filename = job.get("filename") or os.path.basename(job["wav_path"])
        return FileResponse(job["wav_path"], filename=filename)

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

    @app.post("/v1/jobs/{job_id}/translations")
    def create_translation(job_id: str, request: TranslationCreateRequest, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        if job["status"] != "succeeded":
            raise HTTPException(status_code=400, detail="ASR job has not succeeded")
        try:
            client = app.state.translation_client or HunyuanMTClient.from_settings(settings)
            return translate_job_result(job, settings, request.target_language, translator=client)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/translations/stream")
    def stream_translation(job_id: str, request: TranslationCreateRequest, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        if job["status"] != "succeeded":
            raise HTTPException(status_code=400, detail="ASR job has not succeeded")
        try:
            client = app.state.translation_client or HunyuanMTClient.from_settings(settings)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        def events():
            try:
                for event in stream_translate_job_result(job, settings, request.target_language, translator=client):
                    yield f"{json_dumps(event)}\n"
            except ValueError as exc:
                yield f"{json_dumps({'type': 'error', 'error': str(exc)})}\n"
            except FileNotFoundError as exc:
                yield f"{json_dumps({'type': 'error', 'error': str(exc)})}\n"
            except Exception as exc:
                yield f"{json_dumps({'type': 'error', 'error': str(exc)})}\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.get("/v1/jobs/{job_id}/translations/{target_language}")
    def get_translation(job_id: str, target_language: str, user=Depends(_require_user)):
        job = _get_authorized_job(store, job_id, user)
        path = translation_cache_path(job["outdir"], target_language, job_id=job["job_id"])
        if not os.path.exists(path):
            path = translation_cache_path(job["outdir"], target_language)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Translation is not available")
        return FileResponse(path, media_type="application/json", filename=os.path.basename(path))

    @app.get("/v1/models")
    def get_models(language: str, role: str | None = None, user=Depends(_require_user)):
        return list_models(language, role=role)

    return app


def _require_user(
    request: Request,
    authorization: str | None = Header(default=None),
    x_semantic_asr_user: str | None = Header(default=None),
):
    settings = request.app.state.settings
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    user_id = settings.api_keys.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    is_admin = token in settings.admin_tokens
    if settings.demo_user_header_enabled and isinstance(x_semantic_asr_user, str) and x_semantic_asr_user and not is_admin:
        user_id = _sanitize_demo_user(x_semantic_asr_user)
    return {"user_id": user_id, "token": token, "is_admin": is_admin}


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
        "filename": job.get("filename") or "",
        "local_path": job.get("local_path") or "",
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "progress": job["progress"],
        "error": job.get("error"),
        "artifacts": artifact_urls(job["job_id"], job["formats"]) if job["status"] == "succeeded" else {},
    }


def _delete_job_files(job: dict, settings: ServiceSettings) -> None:
    for path in _job_delete_paths(job, settings):
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def _job_delete_paths(job: dict, settings: ServiceSettings) -> list[str]:
    paths = []
    wav_path = job.get("wav_path") or ""
    outdir = job.get("outdir") or ""
    if wav_path:
        upload_dir = os.path.dirname(os.path.abspath(wav_path))
        if _is_relative_to(upload_dir, settings.upload_root):
            paths.append(upload_dir)
    if outdir:
        outputs_dir = os.path.abspath(outdir)
        job_dir = os.path.dirname(outputs_dir)
        if _is_relative_to(job_dir, settings.jobs_root):
            paths.append(job_dir)
    return [path for path in dict.fromkeys(paths) if path]


def _is_relative_to(path: str, root: str) -> bool:
    try:
        abs_path = os.path.abspath(path)
        abs_root = os.path.abspath(root)
        return abs_path != abs_root and os.path.commonpath([abs_path, abs_root]) == abs_root
    except ValueError:
        return False


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


def json_dumps(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False)


def _sanitize_demo_user(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "dev"
    return cleaned[:80]


def _clean_optional_filter(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


app = create_app()


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run("semantic_asr_service.app:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
