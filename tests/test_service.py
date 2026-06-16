import io
import json
import os
import re
import threading
import time
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from semantic_asr_service.app import (
    create_app,
    _delete_job_files,
    _get_authorized_job,
    _job_delete_paths,
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
from semantic_asr_service.translation import (
    stream_translate_job_result,
    translate_job_result,
    translate_sentences_batched,
    translation_cache_path,
)
from semantic_asr_service.worker import run_worker_once


class SemanticAsrServiceTest(unittest.TestCase):
    def test_api_key_auth_accepts_user_and_rejects_missing_or_invalid(self):
        settings = self._settings(tempfile.mkdtemp())
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))

        user = _require_user(request, "Bearer token-a")
        demo_user = _require_user(request, "Bearer token-a", "pm-alice")
        admin = _require_user(request, "Bearer admin-token", "pm-alice")

        self.assertEqual(user["user_id"], "alice")
        self.assertEqual(demo_user["user_id"], "pm-alice")
        self.assertEqual(admin["user_id"], "admin")
        self.assertTrue(admin["is_admin"])
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

    def test_demo_startup_scripts_default_to_requested_gpu_layout(self):
        with open("scripts/start_demo_service.sh", encoding="utf-8") as fin:
            demo_script = fin.read()
        with open("scripts/start_hunyuan_mt_service.sh", encoding="utf-8") as fin:
            translation_script = fin.read()

        self.assertIn('WORKER_DEVICES="${SEMANTIC_ASR_DEMO_WORKER_DEVICES:-6,7}"', demo_script)
        self.assertIn('WORKERS_PER_DEVICE="${SEMANTIC_ASR_DEMO_WORKERS_PER_DEVICE:-2}"', demo_script)
        self.assertIn('DEVICE="${HUNYUAN_MT_DEVICE:-5}"', translation_script)
        self.assertIn('MODEL_ID="${HUNYUAN_MT_MODEL_ID:-Tencent-Hunyuan/HY-MT1.5-1.8B-FP8}"', translation_script)
        self.assertIn('MAX_CONCURRENT="${HUNYUAN_MT_MAX_CONCURRENT:-4}"', translation_script)

    def test_web_demo_route_uses_existing_job_api_and_multi_file_upload(self):
        settings = self._settings(tempfile.mkdtemp())
        app = create_app(settings=settings, store=JobStore(settings.db_path))
        demo = self._route_endpoint(app, "/demo")()

        self.assertIn('id="audio" type="file" multiple', demo)
        self.assertIn('apiFetch("/v1/jobs"', demo)
        self.assertIn("`/v1/jobs/${item.job_id}`", demo)
        self.assertIn("/v1/configs", demo)
        self.assertIn('id="user-name"', demo)
        self.assertIn("apiFetch(jobsUrl())", demo)
        self.assertIn("downloadArtifact(button.dataset.downloadJob", demo)
        self.assertIn('data-download-format="${escapeHtml(name)}"', demo)
        self.assertNotIn('target="_blank" rel="noopener"', demo)
        self.assertIn('id="waveform"', demo)
        self.assertIn("audioFileForJob(job)", demo)
        self.assertIn("`/v1/jobs/${job.job_id}/audio`", demo)
        self.assertIn("normalizeSegments(result.sentences || [])", demo)
        self.assertIn("drawWaveform()", demo)
        self.assertIn("seekSegment(button.dataset.segmentIndex)", demo)
        self.assertNotIn('id="wave-zoom"', demo)
        self.assertIn('grid-template-columns: minmax(0, 1fr);', demo)
        self.assertIn('waveWrap.addEventListener("wheel"', demo)
        self.assertIn('waveWrap.addEventListener("pointerdown"', demo)
        self.assertIn("DEFAULT_TRANSLATION_TARGET = \"zh_cn\"", demo)
        self.assertIn("DEFAULT_DISPLAY_MODE = \"bilingual\"", demo)
        self.assertIn("loadCachedTranslation(jobId, DEFAULT_TRANSLATION_TARGET)", demo)
        self.assertNotIn('id="translation-target"', demo)
        self.assertNotIn('id="text-mode"', demo)
        self.assertNotIn('id="translate"', demo)
        self.assertNotIn('apiFetch(`/v1/jobs/${jobId}/translations/stream`', demo)
        self.assertIn('form.append("local_path", localPath)', demo)
        self.assertIn("displayJobName(job)", demo)
        self.assertIn('"X-Semantic-ASR-User"', demo)
        self.assertIn('id="prev-page"', demo)
        self.assertIn('id="next-page"', demo)
        self.assertIn('id="delete-selected"', demo)
        self.assertIn('id="select-all-jobs"', demo)
        self.assertIn("deleteSelectedJobs", demo)
        self.assertIn("retryJob(button.dataset.retryJob)", demo)
        self.assertIn("showJobError(button.dataset.errorJob)", demo)
        self.assertIn("truncateText(fullName, 42)", demo)
        self.assertIn("<th style=\"width: 13%\">Language</th>", demo)
        self.assertNotIn("<th style=\"width: 17%\">Stage</th>", demo)
        self.assertIn('id="filter-config"', demo)
        self.assertIn('id="filter-user"', demo)
        self.assertIn("jobsUrl()", demo)
        self.assertIn('params.set("config", state.filters.config)', demo)
        self.assertIn('params.set("user_id", state.filters.userId)', demo)
        self.assertIn('class="jobs-table-wrap"', demo)
        self.assertIn("--top-panel-height", demo)
        self.assertIn("playUntilMs", demo)
        self.assertIn("handleAudioTimeUpdate", demo)
        self.assertIn("review.hidden = true", demo)
        self.assertNotIn("updateTranslationUi()", demo)
        self.assertIn("segmentTextHtml(segment)", demo)

    def test_configs_route_returns_allowed_profiles(self):
        settings = self._settings(tempfile.mkdtemp())
        settings.allowed_configs = {"vi_vn", "zh_cn"}
        app = create_app(settings=settings, store=JobStore(settings.db_path))
        response = self._route_endpoint(app, "/v1/configs")(user={"user_id": "alice"})

        self.assertEqual(response, {"configs": ["vi_vn", "zh_cn"]})

    def test_translation_targets_route_returns_allowlist(self):
        settings = self._settings(tempfile.mkdtemp())
        settings.translation_targets = {"zh_cn", "en_us"}
        app = create_app(settings=settings, store=JobStore(settings.db_path))
        response = self._route_endpoint(app, "/v1/translation-targets")(user={"user_id": "alice"})

        self.assertEqual(response, {"targets": ["en_us", "zh_cn"]})

    def test_me_route_returns_effective_demo_user(self):
        settings = self._settings(tempfile.mkdtemp())
        app = create_app(settings=settings, store=JobStore(settings.db_path))

        response = self._route_endpoint(app, "/v1/me")(user={"user_id": "pm-alice", "is_admin": False})

        self.assertEqual(response, {"user_id": "pm-alice", "is_admin": False})

    def test_jobs_route_lists_only_current_user_unless_admin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            app = create_app(settings=settings, store=JobStore(settings.db_path))
            store = app.state.store
            store.create_job(
                "alice-job",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "alice-job", "outputs"),
                ["json"],
                filename="alice.wav",
                local_path="/audio/alice.wav",
            )
            store.create_job(
                "bob-job",
                "bob",
                "vi_vn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "bob-job", "outputs"),
                ["json"],
                filename="bob.wav",
            )
            store.create_job(
                "translation_smoke_hidden",
                "alice",
                "ko_kr",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "translation_smoke_hidden", "outputs"),
                ["json"],
            )
            store.create_job(
                "alice-job-2",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "alice-job-2", "outputs"),
                ["json"],
                filename="alice2.wav",
            )

            alice_response = self._route_endpoint(app, "/v1/jobs")(
                limit=1,
                offset=0,
                user={"user_id": "alice", "is_admin": False},
            )
            alice_page_2 = self._route_endpoint(app, "/v1/jobs")(
                limit=1,
                offset=1,
                user={"user_id": "alice", "is_admin": False},
            )
            admin_response = self._route_endpoint(app, "/v1/jobs")(
                user={"user_id": "admin", "is_admin": True},
            )
            admin_bob_response = self._route_endpoint(app, "/v1/jobs")(
                config="vi_vn",
                user_id="bob",
                user={"user_id": "admin", "is_admin": True},
            )
            alice_vi_response = self._route_endpoint(app, "/v1/jobs")(
                config="vi_vn",
                user={"user_id": "alice", "is_admin": False},
            )
            alice_bob_response = self._route_endpoint(app, "/v1/jobs")(
                user_id="bob",
                user={"user_id": "alice", "is_admin": False},
            )

        self.assertEqual(len(alice_response["jobs"]), 1)
        self.assertEqual(len(alice_page_2["jobs"]), 1)
        self.assertEqual({alice_response["jobs"][0]["job_id"], alice_page_2["jobs"][0]["job_id"]}, {"alice-job", "alice-job-2"})
        self.assertEqual({job["job_id"] for job in admin_response["jobs"]}, {"alice-job", "alice-job-2", "bob-job"})
        self.assertEqual([job["job_id"] for job in admin_bob_response["jobs"]], ["bob-job"])
        self.assertEqual(alice_vi_response["jobs"], [])
        self.assertEqual(alice_bob_response["jobs"], [])
        first_alice = next(job for job in [*alice_response["jobs"], *alice_page_2["jobs"]] if job["job_id"] == "alice-job")
        self.assertEqual(first_alice["filename"], "alice.wav")
        self.assertEqual(first_alice["local_path"], "/audio/alice.wav")
        self.assertEqual(alice_response["total"], 2)
        self.assertEqual(alice_response["limit"], 1)
        self.assertEqual(alice_response["offset"], 0)

    def test_delete_job_removes_owned_terminal_job_and_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            app = create_app(settings=settings, store=JobStore(settings.db_path))
            store = app.state.store
            upload_dir = os.path.join(settings.upload_root, "job1")
            outdir = os.path.join(settings.jobs_root, "job1", "outputs")
            os.makedirs(upload_dir, exist_ok=True)
            os.makedirs(outdir, exist_ok=True)
            wav_path = os.path.join(upload_dir, "input.wav")
            sf.write(wav_path, np.zeros(160, dtype=np.int16), 16000)
            job = store.create_job("job1", "alice", "zh_cn", wav_path, outdir, ["json"])
            store.mark_succeeded(job["job_id"])

            response = self._route_endpoint(app, "/v1/jobs/{job_id}", method="DELETE")(
                "job1",
                user={"user_id": "alice", "is_admin": False},
            )

            self.assertEqual(response, {"job_id": "job1", "deleted": True})
            self.assertIsNone(store.get_job("job1"))
            self.assertFalse(os.path.exists(upload_dir))
            self.assertFalse(os.path.exists(os.path.dirname(outdir)))

    def test_delete_job_rejects_running_or_other_user_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            app = create_app(settings=settings, store=JobStore(settings.db_path))
            store = app.state.store
            running = store.create_job(
                "running",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "running", "outputs"),
                ["json"],
            )
            other = store.create_job(
                "other",
                "bob",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "other", "outputs"),
                ["json"],
            )
            store.claim_next_job()

            with self.assertRaises(HTTPException) as running_error:
                self._route_endpoint(app, "/v1/jobs/{job_id}", method="DELETE")(
                    running["job_id"],
                    user={"user_id": "alice", "is_admin": False},
                )
            with self.assertRaises(HTTPException) as other_error:
                self._route_endpoint(app, "/v1/jobs/{job_id}", method="DELETE")(
                    other["job_id"],
                    user={"user_id": "alice", "is_admin": False},
                )

        self.assertEqual(running_error.exception.status_code, 409)
        self.assertEqual(other_error.exception.status_code, 404)

    def test_job_delete_paths_are_limited_to_service_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            upload_dir = os.path.join(settings.upload_root, "job1")
            job_dir = os.path.join(settings.jobs_root, "job1")
            outdir = os.path.join(job_dir, "outputs")
            external_dir = os.path.join(tmpdir, "external")
            for path in (upload_dir, outdir, external_dir):
                os.makedirs(path, exist_ok=True)
            job = {
                "wav_path": os.path.join(upload_dir, "input.wav"),
                "outdir": outdir,
            }
            external_job = {
                "wav_path": os.path.join(external_dir, "input.wav"),
                "outdir": os.path.join(external_dir, "outputs"),
            }

            self.assertEqual(set(_job_delete_paths(job, settings)), {upload_dir, job_dir})
            self.assertEqual(_job_delete_paths(external_job, settings), [])
            _delete_job_files(external_job, settings)
            self.assertTrue(os.path.exists(external_dir))

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

    def test_retry_failed_job_route_requeues_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            app = create_app(settings=settings, store=JobStore(settings.db_path))
            store = app.state.store
            job = store.create_job(
                "job1",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json"],
            )
            store.mark_failed(job["job_id"], "boom")

            response = self._route_endpoint(app, "/v1/jobs/{job_id}/retry", method="POST")(
                job["job_id"],
                user={"user_id": "alice", "is_admin": False},
            )
            retried = store.get_job(job["job_id"])

        self.assertEqual(response["status"], "queued")
        self.assertEqual(retried["status"], "queued")
        self.assertIsNone(retried["error"])
        self.assertIsNone(retried["started_at"])
        self.assertIsNone(retried["finished_at"])

    def test_retry_route_rejects_running_or_other_user_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            app = create_app(settings=settings, store=JobStore(settings.db_path))
            store = app.state.store
            running = store.create_job(
                "running",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "running", "outputs"),
                ["json"],
            )
            other = store.create_job(
                "other",
                "bob",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "other", "outputs"),
                ["json"],
            )
            store.claim_next_job()

            with self.assertRaises(HTTPException) as running_error:
                self._route_endpoint(app, "/v1/jobs/{job_id}/retry", method="POST")(
                    running["job_id"],
                    user={"user_id": "alice", "is_admin": False},
                )
            with self.assertRaises(HTTPException) as other_error:
                self._route_endpoint(app, "/v1/jobs/{job_id}/retry", method="POST")(
                    other["job_id"],
                    user={"user_id": "alice", "is_admin": False},
                )

        self.assertEqual(running_error.exception.status_code, 409)
        self.assertEqual(other_error.exception.status_code, 404)

    def test_audio_route_returns_uploaded_audio_for_authorized_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            app = create_app(settings=settings, store=JobStore(settings.db_path))
            wav_path = self._write_wav(tmpdir)
            job = app.state.store.create_job(
                "job1",
                "alice",
                "zh_cn",
                wav_path,
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json"],
                filename="demo.wav",
            )

            response = self._route_endpoint(app, "/v1/jobs/{job_id}/audio")(
                job["job_id"],
                user={"user_id": "alice", "is_admin": False},
            )

        self.assertEqual(response.path, wav_path)
        self.assertEqual(response.filename, "demo.wav")

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
                self.assertEqual(store.get_job(job["job_id"])["status"], "running")
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

    def test_worker_translates_before_marking_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            settings.translation_base_url = "http://translation.local"
            settings.translation_targets = {"zh_cn", "en_us"}
            settings.auto_translate_targets = {"zh_cn"}
            store = JobStore(settings.db_path)
            job = store.create_job(
                "job1",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json"],
            )

            def ok_runner(job, _settings):
                os.makedirs(job["outdir"], exist_ok=True)
                with open(artifact_path(job["outdir"], job["job_id"], "json"), "w", encoding="utf-8") as fout:
                    json.dump({"job_id": job["job_id"], "sentences": []}, fout)

            with mock.patch("semantic_asr_service.translation.translate_job_result") as translate_mock:
                result = run_worker_once(settings, store, runner=ok_runner)

        self.assertEqual(result["status"], "succeeded")
        translate_mock.assert_called_once()
        called_job, called_settings, called_target = translate_mock.call_args.args
        self.assertEqual(called_job["job_id"], job["job_id"])
        self.assertIs(called_settings, settings)
        self.assertEqual(called_target, "zh_cn")

    def test_worker_marks_failed_when_required_auto_translation_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            settings.translation_base_url = "http://translation.local"
            settings.translation_targets = {"zh_cn"}
            settings.auto_translate_targets = {"zh_cn"}
            store = JobStore(settings.db_path)
            job = store.create_job(
                "job1",
                "alice",
                "zh_cn",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json"],
            )

            def ok_runner(job, _settings):
                os.makedirs(job["outdir"], exist_ok=True)
                with open(artifact_path(job["outdir"], job["job_id"], "json"), "w", encoding="utf-8") as fout:
                    json.dump({"job_id": job["job_id"], "sentences": []}, fout)

            with mock.patch("semantic_asr_service.translation.translate_job_result", side_effect=RuntimeError("translate boom")):
                result = run_worker_once(settings, store, runner=ok_runner)

        self.assertEqual(result["status"], "failed")
        self.assertIn("translate boom", result["error"])

    def test_translate_job_result_writes_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            settings.translation_targets = {"zh_cn"}
            store = JobStore(settings.db_path)
            job = store.create_job(
                "job1",
                "alice",
                "ko_kr",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json"],
            )
            store.mark_succeeded("job1")
            job = store.get_job("job1")
            os.makedirs(job["outdir"], exist_ok=True)
            with open(os.path.join(job["outdir"], "job1.json"), "w", encoding="utf-8") as fout:
                json.dump({
                    "sentences": [
                        {"start_ms": 100, "end_ms": 300, "cut_start_ms": 120, "cut_end_ms": 280, "text": "감회가 새롭습니다."}
                    ]
                }, fout)

            translator = self.FakeTranslator()
            first = translate_job_result(job, settings, "zh_cn", translator=translator)
            second = translate_job_result(job, settings, "zh_cn", translator=translator)

        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(first["sentences"][0]["translation"], "Chinese::감회가 새롭습니다.")
        self.assertEqual(second["sentences"][0]["translation"], "Chinese::감회가 새롭습니다.")
        self.assertEqual(translator.calls, 1)

    def test_translate_sentences_batched_accepts_structured_batch_output(self):
        sentences = [
            {"index": 3, "text": "Hallo Welt."},
            {"index": 4, "text": "Guten Morgen."},
        ]
        translator = self.FakeTranslator()

        translated = translate_sentences_batched(sentences, translator, "German", "Chinese", batch_size=16)

        self.assertEqual([item["translation"] for item in translated], ["Chinese::Hallo Welt.", "Chinese::Guten Morgen."])
        self.assertEqual(translator.calls, 1)
        self.assertEqual(translator.fallback_calls, 0)

    def test_translate_sentences_batched_falls_back_on_malformed_batch_output(self):
        sentences = [
            {"index": 1, "text": "مرحبا"},
            {"index": 2, "text": "كيف الحال؟"},
        ]
        translator = self.BadBatchTranslator()

        translated = translate_sentences_batched(sentences, translator, "Arabic", "Chinese", batch_size=16)

        self.assertEqual([item["translation"] for item in translated], ["Chinese::مرحبا", "Chinese::كيف الحال؟"])
        self.assertEqual(translator.batch_calls, 1)
        self.assertEqual(translator.fallback_calls, 2)

    def test_translate_sentences_batched_supports_concurrent_single_mode(self):
        sentences = [
            {"index": 1, "text": "hello"},
            {"index": 2, "text": "world"},
        ]
        translator = self.FakeTranslator()

        translated = translate_sentences_batched(
            sentences,
            translator,
            "English",
            "Chinese",
            batch_size=2,
            request_mode="concurrent_single",
        )

        self.assertEqual([item["translation"] for item in translated], ["Chinese::hello", "Chinese::world"])
        self.assertEqual(translator.calls, 0)
        self.assertEqual(translator.fallback_calls, 2)

    def test_concurrent_single_mode_respects_max_concurrency(self):
        sentences = [{"index": index, "text": f"text {index}"} for index in range(4)]
        translator = self.CountingTranslator()

        translated = translate_sentences_batched(
            sentences,
            translator,
            "English",
            "Chinese",
            batch_size=4,
            request_mode="concurrent_single",
            max_concurrency=2,
        )

        self.assertEqual([item["translation"] for item in translated], [f"Chinese::text {index}" for index in range(4)])
        self.assertLessEqual(translator.max_active, 2)

    def test_stream_translate_job_result_yields_sentences_and_writes_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            settings.translation_targets = {"zh_cn"}
            settings.translation_batch_size = 2
            store = JobStore(settings.db_path)
            job = store.create_job(
                "job1",
                "alice",
                "ko_kr",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json"],
            )
            store.mark_succeeded("job1")
            job = store.get_job("job1")
            os.makedirs(job["outdir"], exist_ok=True)
            with open(os.path.join(job["outdir"], "job1.json"), "w", encoding="utf-8") as fout:
                json.dump({
                    "sentences": [
                        {"start_ms": 0, "end_ms": 100, "text": "hello"},
                        {"start_ms": 100, "end_ms": 200, "text": "world"},
                    ]
                }, fout)

            events = list(stream_translate_job_result(job, settings, "zh_cn", translator=self.FakeTranslator()))

            sentence_events = [event for event in events if event["type"] == "sentence"]
            self.assertEqual(len(sentence_events), 2)
            self.assertEqual(events[-1]["type"], "done")
            self.assertTrue(os.path.exists(translation_cache_path(job["outdir"], "zh_cn")))

    def test_translate_job_result_rejects_disallowed_target(self):
        settings = self._settings(tempfile.mkdtemp())
        settings.translation_targets = {"zh_cn"}

        with self.assertRaises(ValueError):
            translate_job_result({"outdir": "", "job_id": "job1", "config": "ko_kr"}, settings, "en_us", self.FakeTranslator())

    def test_translation_routes_require_configured_service_or_fake_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self._settings(tmpdir)
            settings.translation_targets = {"zh_cn"}
            app = create_app(settings=settings, store=JobStore(settings.db_path))
            store = app.state.store
            job = store.create_job(
                "job1",
                "alice",
                "ko_kr",
                self._write_wav(tmpdir),
                os.path.join(tmpdir, "jobs", "job1", "outputs"),
                ["json"],
            )
            store.mark_succeeded(job["job_id"])
            job = store.get_job(job["job_id"])
            os.makedirs(job["outdir"], exist_ok=True)
            with open(os.path.join(job["outdir"], "job1.json"), "w", encoding="utf-8") as fout:
                json.dump({"sentences": [{"start_ms": 0, "end_ms": 100, "text": "hello"}]}, fout)
            request = SimpleNamespace(target_language="zh_cn")

            with self.assertRaises(HTTPException) as missing_service:
                self._route_endpoint(app, "/v1/jobs/{job_id}/translations")(
                    "job1",
                    request,
                    user={"user_id": "alice", "is_admin": False},
                )

            app.state.translation_client = self.FakeTranslator()
            translated = self._route_endpoint(app, "/v1/jobs/{job_id}/translations")(
                "job1",
                request,
                user={"user_id": "alice", "is_admin": False},
            )

        self.assertEqual(missing_service.exception.status_code, 503)
        self.assertEqual(translated["sentences"][0]["translation"], "Chinese::hello")

    def test_translation_cache_path_is_under_output_dir(self):
        path = translation_cache_path("/tmp/out", "zh_cn")

        self.assertEqual(path, "/tmp/out/translations/zh_cn.json")

    @staticmethod
    def _route_endpoint(app, path: str, method: str | None = None):
        for route in app.routes:
            if getattr(route, "path", None) != path:
                continue
            if method and method.upper() not in getattr(route, "methods", set()):
                continue
            if getattr(route, "path", None) == path:
                return route.endpoint
        raise AssertionError(f"Route not found: {path}")

    class FakeTranslator:
        def __init__(self):
            self.calls = 0
            self.fallback_calls = 0

        def translate(self, text: str, _source_language: str, target_language: str) -> str:
            self.fallback_calls += 1
            return f"{target_language}::{text}"

        def complete(self, prompt: str) -> str:
            self.calls += 1
            match = re.search(r"\[[\s\S]*\]", prompt)
            if not match:
                return "[]"
            items = json.loads(match.group(0))
            translated = [
                {"index": int(item["index"]), "translation": f"Chinese::{item['text']}"}
                for item in items
            ]
            return json.dumps(translated, ensure_ascii=False)

    class BadBatchTranslator:
        def __init__(self):
            self.batch_calls = 0
            self.fallback_calls = 0

        def translate(self, text: str, _source_language: str, target_language: str) -> str:
            self.fallback_calls += 1
            return f"{target_language}::{text}"

        def complete(self, _prompt: str) -> str:
            self.batch_calls += 1
            return "not json"

    class CountingTranslator:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def translate(self, text: str, _source_language: str, target_language: str) -> str:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            with self.lock:
                self.active -= 1
            return f"{target_language}::{text}"

        def complete(self, _prompt: str) -> str:
            raise AssertionError("concurrent_single should not call complete()")

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
