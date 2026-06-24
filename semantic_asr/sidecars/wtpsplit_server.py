from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from typing import Any, Sequence


_SAT = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve wtpsplit/SaT sentence boundary predictions.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11001)
    parser.add_argument("--model", default=os.environ.get("WTPSPLIT_MODEL", "sat-12l-sm"))
    args = parser.parse_args()
    Handler.model_name = args.model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"wtpsplit boundary server listening on {args.host}:{args.port} model={args.model}", flush=True)
    server.serve_forever()


class Handler(BaseHTTPRequestHandler):
    model_name = "sat-12l-sm"

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self.send_error(404)
            return
        self._write_json({"ok": True})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/boundaries":
            self.send_error(404)
            return
        try:
            request = self._read_json()
            response = predict_boundaries(request, self.model_name)
            self._write_json(response)
        except Exception as exc:  # noqa: BLE001 - service boundary.
            self._write_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A003,N802 - BaseHTTPRequestHandler API
        return


def predict_boundaries(request: dict[str, Any], model_name: str) -> dict[str, Any]:
    tokens = [str(token) for token in request.get("tokens") or [] if str(token).strip()]
    if not tokens:
        return {"token_count": 0, "end_indices": [], "segments": []}
    text = str(request.get("text") or _join_tokens(tokens))
    max_length = request.get("max_length")
    sat = _load_sat(model_name)
    if max_length:
        segments = sat.split(text, max_length=int(max_length))
    else:
        segments = sat.split(text)
    segments = [str(segment).strip() for segment in segments if str(segment).strip()]
    end_indices = segments_to_end_indices(tokens, segments)
    return {
        "language": request.get("language", "und"),
        "token_count": len(tokens),
        "end_indices": end_indices,
        "segments": segments,
    }


def _load_sat(model_name: str):
    global _SAT
    if _SAT is None:
        from wtpsplit import SaT

        _SAT = SaT(model_name, from_pretrained_kwargs={"local_files_only": True})
    return _SAT


def segments_to_end_indices(tokens: Sequence[str], segments: Sequence[str]) -> list[int]:
    if not tokens:
        return []
    cumulative_token_lengths = []
    total = 0
    for token in tokens:
        total += len(_norm(token))
        cumulative_token_lengths.append(total)

    end_indices = []
    target = 0
    for segment in segments:
        target += len(_norm(segment))
        for index, token_length in enumerate(cumulative_token_lengths):
            if token_length >= target:
                if not end_indices or index > end_indices[-1]:
                    end_indices.append(index)
                break
    if not end_indices or end_indices[-1] != len(tokens) - 1:
        end_indices.append(len(tokens) - 1)
    return end_indices


def _join_tokens(tokens: Sequence[str]) -> str:
    text = ""
    previous = ""
    for token in tokens:
        if text and previous.isascii() and token.isascii():
            text += " "
        text += token
        previous = token
    return text


def _norm(text: str) -> str:
    return "".join(char.lower() for char in text if not char.isspace())


if __name__ == "__main__":
    main()
