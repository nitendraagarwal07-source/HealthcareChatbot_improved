import os, time, socket
from typing import Any, Dict
import orjson
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag_guardrails.jsonl"

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

def log_event(event: Dict[str, Any]) -> None:
    event = {"ts": _now_iso(), "host": socket.gethostname(), **event}
    with open(LOG_FILE, "ab") as f:
        f.write(orjson.dumps(event) + b"\n")

def log_query(**kwargs):
    log_event({"type": "query", **kwargs})

def log_response(**kwargs):
    log_event({"type": "response", **kwargs})
