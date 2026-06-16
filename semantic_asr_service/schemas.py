from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    config: str
    filename: str | None = None
    local_path: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict
    error: str | None = None
    artifacts: dict[str, str] = {}


class TranslationCreateRequest(BaseModel):
    target_language: str = "zh_cn"
