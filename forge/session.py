"""Per-run model exchange and append-only event ownership."""
from copy import deepcopy
from pathlib import Path
from .checkpoints import ResponseCheckpoints
from .contracts import JsonObject, LanguageModel
from .llm import structured_call
from .util import write_json


class RunSession:
    def __init__(self, folder: Path, run_id: str, llm: LanguageModel,
                 checkpoints: ResponseCheckpoints | None = None):
        self.folder = folder
        self.run_id = run_id
        self.llm = llm
        self.checkpoints = checkpoints
        self.events: list[JsonObject] = []

    def ask(self, role: str, context: JsonObject) -> JsonObject:
        cached = self.checkpoints.lookup(role, context) if self.checkpoints else None
        if cached is not None:
            print(f"复用已保存的 {role} 模型响应；后续验证仍会执行。", flush=True)
            self.record(role + "_response", {"context": context, "source": "checkpoint"}, cached)
            return cached
        print(f"正在执行 {role}……", flush=True)
        # A provider cannot mutate the host's plan, evidence or retry context.
        result = structured_call(self.llm, role, deepcopy(context))
        self.record(role + "_response", {"context": context,
                    "source": getattr(self.llm, "response_source", "live")}, result)
        return result

    def record(self, role: str, incoming: JsonObject, outgoing: JsonObject) -> None:
        event = deepcopy(dict(step=len(self.events), role=role, input=incoming, output=outgoing))
        self.events.append(event)
        write_json(self.folder / "events.json", self.events)
        if self.checkpoints and role in {"understand", "plan", "generate", "repair"}:
            if role not in {"generate", "repair"} or outgoing.get("code"):
                self.checkpoints.save(role, incoming, outgoing, self.run_id)
