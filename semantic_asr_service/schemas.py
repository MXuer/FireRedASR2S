from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    config: str
    progress: dict
    error: str | None = None
    artifacts: dict[str, str] = {}

