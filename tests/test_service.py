import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from semantic_asr_service.app import (
    create_app,
    _get_authorized_job,
    _job_response,
    _parse_formats,
    _require_allowed_config,
    _require_user,
    _save_upload,
    _validate_audio_duration,
)
from semantic_asr_service.artifacts import artifact_path
from semantic_asr_service.settings import ServiceSettings, _parse_api_keys, load_settings
from semantic_asr_service.store import JobStore
from semantic_asr_service.worker import run_worker_once


class SemanticAsrServiceTest(unittest.TestCase):
    def test_api_key_auth_accepts_user_and_rejects_missing_or_invalid(self):
        settings = self._settings(tempfile.mkdtemp())
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))

        user = _require_user(request, "Bearer token-a")

        self.assertEqual(user["user_id"], "alice")
        with self.assertRaises(HTTPException) as missing:
            _require_user(request, None)
        with self.assertRaises(HTTPException) as invalid:
            _require_user(request, "Bearer nope")
        self.assertEqual(missing.exception.status_code, 401)
        self.assertEqual(invalid.exception.status_code, 401)

    def test_parse_api_keys_marks_admin_tokens(self):
        api_keys, admin_tokens = _parse_api_keys("token-a:alice,admin-token:admin:admin")

        self.assertEqual(api_keys["token-a"], "alice")
        self.assertEqual(api_keys["admin-token"], "admin")
        self.assertIn("admin-token", admin_tokens)

    def test_service_port_defaults_to_10086_and_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {
                    "SEMANTIC_ASR_SERVICE_DATA_DIR": os.path.join(tmpdir, "service_data"),
                    "SEMANTIC_ASR_CONFIGS_DIR": os.path.join(tmpdir, "configs"),
                    "SEMANTIC_ASR_ALLOWED_CONFIGS": "zh_cn",
                    "SEMANTIC_ASR_API_KEYS": "token-a:alice",
                },
                clear=True,
            ):
                default_settings = load_settings()
            with mock.patch.dict(
                os.environ,
                {
                    "SEMANTIC_ASR_SERVICE_DATA_DIR": os.path.join(tmpdir, "service_data"),
                    "SEMANTIC_ASR_CONFIGS_DIR": os.path.join(tmpdir, "configs"),
                    "SEMANTIC_ASR_ALLOWED_CONFIGS": "zh_cn",
                    "SEMANTIC_ASR_API_KEYS": "token-a:alice",
                    "SEMANTIC_ASR_SERVICE_PORT": "12345",
                },
                clear=True,
            ):
                overridden_settings = load_settings()

        self.assertEqual(default_settings.port, 10086)
        self.assertEqual(overridden_settings.port, 12345)

    def test_web_demo_route_uses_existing_job_api_and_multi_file_upload(self):
        settings = self._settings(tempfile.mkdtemp())
        app = create_app(settings=settings, store=JobStore(settings.db_path))
        demo = self._route_endpoint(app, "/demo")()

        self.assertIn('id="audio" type="file" multiple', demo)
        self.assertIn('apiFetch("/v1/jobs"', demo)
        self.assertIn("`/v1/jobs/${item.job_id}`", demo)
        self.assertIn("/v1/configs", demo)
        self.assertIn("downloadArtifact(button.dataset.downloadJob", demo)
        self.assertIn('data-download-format="${escapeHtml(name)}"', demo)
        self.assertNotIn('target="_blank" rel="noopener"', demo)
        self.assertIn('id="waveform"', demo)
        self.assertIn("decodeAudioFile(job.fileObject)", demo)
        self.assertIn("normalizeSegments(result.sentences || [])", demo)
        self.assertIn("drawWaveform()", demo)
        self.assertIn("seekSegment(button.dataset.segmentIndex)", demo)
        self.assertIn('id="wave-zoom"', demo)
        self.assertIn('grid-template-columns: minmax(0, 1fr);', demo)
        self.assertIn('waveWrap.addEventListener("wheel"', demo)
        self.assertIn('waveWrap.addEventListener("pointerdown"', demo)
        self.assertIn("setWaveZoom(Number(waveZoom.value)", demo)

    def test_configs_route_returns_allowed_profiles(self):
        settings = self._settings(tempfile.mkdtemp())
        settings.allowed_configs = {"vi_vn", "zh_cn"}
        app = create_app(settings=settings, store=JobStore(settings.db_path))
        response = self._route_endpoint(app, "/v1/configs")(user={"user_id": "alice"})

        self.assertEqual(response, {"configs": ["vi_vn", "zh_cn"]})

    def test_submit_job_core_creates_queued_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            store = JobStore(settings.db_path)
            wav_path = self._write_wav(tmpdir)
            formats = _parse_formats("srt,csv")
            _require_allowed_config(settings, "zh_cn")

            job = store.create_job(
                "job1",
                "alice",
                "zh_cn",
                wav_path,
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                formats,
            )

        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["formats"], ["json", "srt", "csv"])
        self.assertEqual(job["progress"], {"stage": "queued"})

    def test_config_must_be_allowlisted(self):
        settings = self._settings(tempfile.mkdtemp())

        with self.assertRaises(HTTPException) as ctx:
            _require_allowed_config(settings, "not_allowed")

        self.assertEqual(ctx.exception.status_code, 400)

    def test_user_cannot_read_other_users_job_but_admin_can(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            store = JobStore(settings.db_path)
            job = store.create_job(
                "job1",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json"],
            )

            with self.assertRaises(HTTPException) as denied:
                _get_authorized_job(store, job["job_id"], {"user_id": "bob", "is_admin": False})
            allowed = _get_authorized_job(store, job["job_id"], {"user_id": "admin", "is_admin": True})

        self.assertEqual(denied.exception.status_code, 404)
        self.assertEqual(allowed["job_id"], "job1")

    def test_job_response_artifacts_only_after_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            store = JobStore(settings.db_path)
            job = store.create_job(
                "job1",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json", "srt"],
            )
            queued_response = _job_response(job)
            store.mark_succeeded("job1")
            succeeded_response = _job_response(store.get_job("job1"))

        self.assertEqual(queued_response["artifacts"], {})
        self.assertIn("json", succeeded_response["artifacts"])
        self.assertIn("srt", succeeded_response["artifacts"])

    def test_artifact_path_and_missing_file_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = artifact_path(tmpdir, "job1", "json")

        self.assertTrue(path.endswith("job1.json"))
        self.assertFalse(os.path.exists(path))

    def test_upload_helpers_enforce_size_and_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            upload = SimpleNamespace(file=io.BytesIO(b"123456"))
            output_path = os.path.join(tmpdir, "input.wav")

            with self.assertRaises(HTTPException) as too_large:
                _save_upload(upload, output_path, max_bytes=3)

            wav_path = self._write_wav(tmpdir, samples=32000)
            with self.assertRaises(HTTPException) as too_long:
                _validate_audio_duration(wav_path, max_audio_seconds=0.1)

        self.assertEqual(too_large.exception.status_code, 413)
        self.assertEqual(too_long.exception.status_code, 413)

    def test_worker_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            store = JobStore(settings.db_path)
            success = store.create_job(
                "success",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "success", "outputs"),
                ["json"],
            )
            failed = store.create_job(
                "failed",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "failed", "outputs"),
                ["json"],
            )

            def ok_runner(job, _settings):
                os.makedirs(job["outdir"], exist_ok=True)
                with open(artifact_path(job["outdir"], job["job_id"], "json"), "w", encoding="utf-8") as fout:
                    json.dump({"job_id": job["job_id"]}, fout)

            success_result = run_worker_once(settings, store, runner=ok_runner)

            def fail_runner(job, _settings):
                raise RuntimeError("boom")

            failed_result = run_worker_once(settings, store, runner=fail_runner)

        self.assertEqual(success_result["job_id"], success["job_id"])
        self.assertEqual(success_result["status"], "succeeded")
        self.assertEqual(failed_result["job_id"], failed["job_id"])
        self.assertEqual(failed_result["status"], "failed")
        self.assertIn("boom", failed_result["error"])

    @staticmethod
    def _route_endpoint(app, path: str):
        for route in app.routes:
            if getattr(route, "path", None) == path:
                return route.endpoint
        raise AssertionError(f"Route not found: {path}")

    @staticmethod
    def _write_wav(tmpdir: str, samples: int = 160) -> str:
        path = os.path.join(tmpdir, f"{len(os.listdir(tmpdir))}.wav")
        sf.write(path, np.zeros(samples, dtype=np.int16), 16000)
        return path

    @staticmethod
    def _settings(tmpdir: str) -> ServiceSettings:
        configs_dir = os.path.join(tmpdir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        with open(os.path.join(configs_dir, "zh_cn.json"), "w", encoding="utf-8") as fout:
            json.dump({"name": "dummy"}, fout)
        return ServiceSettings(
            data_dir=os.path.join(tmpdir, "service_data"),
            configs_dir=configs_dir,
            allowed_configs={"zh_cn"},
            api_keys={"token-a": "alice", "token-b": "bob", "admin-token": "admin"},
            admin_tokens={"admin-token"},
            max_upload_mb=1,
            max_audio_seconds=0,
        )


if __name__ == "__main__":
    unittest.main()
