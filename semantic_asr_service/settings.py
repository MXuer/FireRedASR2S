from dataclasses import dataclass, field
import glob
import os
from pathlib import Path


@dataclass
class ServiceSettings:
    data_dir: str = "service_data"
    configs_dir: str = "configs"
    allowed_configs: set[str] = field(default_factory=set)
    api_keys: dict[str, str] = field(default_factory=lambda: {"dev-token": "dev"})
    admin_tokens: set[str] = field(default_factory=set)
    max_upload_mb: int = 2048
    max_audio_seconds: float = 0
    host: str = "0.0.0.0"
    port: int = 10086
    allowed_extensions: set[str] = field(default_factory=lambda: {".wav", ".flac", ".mp3", ".m4a", ".ogg"})
    poll_interval_s: float = 2.0

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "jobs.sqlite3")

    @property
    def upload_root(self) -> str:
        return os.path.join(self.data_dir, "uploads")

    @property
    def jobs_root(self) -> str:
        return os.path.join(self.data_dir, "jobs")

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024

    def resolve_config_path(self, config: str) -> str:
        if config not in self.allowed_configs:
            raise ValueError(f"Config is not allowed: {config}")
        path = os.path.join(self.configs_dir, f"{config}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file does not exist: {path}")
        return path


def load_settings() -> ServiceSettings:
    data_dir = os.environ.get("SEMANTIC_ASR_SERVICE_DATA_DIR", "service_data")
    configs_dir = os.environ.get("SEMANTIC_ASR_CONFIGS_DIR", "configs")
    allowed_configs = _env_set("SEMANTIC_ASR_ALLOWED_CONFIGS") or _discover_configs(configs_dir)
    api_keys, admin_tokens = _parse_api_keys(os.environ.get("SEMANTIC_ASR_API_KEYS", "dev-token:dev"))
    return ServiceSettings(
        data_dir=data_dir,
        configs_dir=configs_dir,
        allowed_configs=allowed_configs,
        api_keys=api_keys,
        admin_tokens=admin_tokens,
        max_upload_mb=int(os.environ.get("SEMANTIC_ASR_MAX_UPLOAD_MB", "2048")),
        max_audio_seconds=float(os.environ.get("SEMANTIC_ASR_MAX_AUDIO_SECONDS", "0")),
        host=os.environ.get("SEMANTIC_ASR_SERVICE_HOST", "0.0.0.0"),
        port=int(os.environ.get("SEMANTIC_ASR_SERVICE_PORT", "10086")),
        allowed_extensions=_env_set("SEMANTIC_ASR_ALLOWED_EXTENSIONS") or {".wav", ".flac", ".mp3", ".m4a", ".ogg"},
        poll_interval_s=float(os.environ.get("SEMANTIC_ASR_POLL_INTERVAL_S", "2.0")),
    )


def _discover_configs(configs_dir: str) -> set[str]:
    return {Path(path).stem for path in glob.glob(os.path.join(configs_dir, "*.json"))}


def _env_set(name: str) -> set[str]:
    raw = os.environ.get(name, "").strip()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _parse_api_keys(raw: str) -> tuple[dict[str, str], set[str]]:
    api_keys = {}
    admin_tokens = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        token = parts[0].strip()
        user_id = parts[1].strip() if len(parts) > 1 and parts[1].strip() else token
        role = parts[2].strip().lower() if len(parts) > 2 else "user"
        api_keys[token] = user_id
        if role == "admin":
            admin_tokens.add(token)
    return api_keys, admin_tokens
