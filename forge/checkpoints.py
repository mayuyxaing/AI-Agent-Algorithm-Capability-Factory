"""Local response checkpoints; validation is always executed again."""
import json
import uuid
from pathlib import Path
from .llm import OllamaLLM
from .util import digest, read_json, write_json


class ResponseCheckpoints:
    def __init__(self, folder: Path, identity: dict):
        self.folder, self.identity = folder, identity
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hits = []

    def path(self, role, context):
        key = digest(json.dumps({"identity": self.identity, "role_prompt": OllamaLLM.prompts()[role],
                                 "role": role, "context": context}, sort_keys=True, ensure_ascii=False))
        return self.folder / f"{key}.json"

    def lookup(self, role, context):
        path = self.path(role, context)
        if not path.exists():
            return None
        item = read_json(path)
        if digest(json.dumps(item["output"], sort_keys=True, ensure_ascii=False)) != item["output_sha256"]:
            raise ValueError("Local response checkpoint is corrupted")
        self.hits.append({"role": role, "source_run_id": item["source_run_id"], "checkpoint": path.name})
        return item["output"]

    def save(self, role, context, output, run_id):
        path = self.path(role, context)
        if path.exists():
            return
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        write_json(temporary, {"source_run_id": run_id, "role": role, "output": output,
                              "output_sha256": digest(json.dumps(output, sort_keys=True, ensure_ascii=False))})
        temporary.replace(path)
