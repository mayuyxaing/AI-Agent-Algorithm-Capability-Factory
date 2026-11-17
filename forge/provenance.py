"""Shared execution and provider identities for experiments and response caches."""
import json
import importlib.metadata
import platform
from pathlib import Path
from .llm import plugins
from .util import digest


def execution_identity():
    """Reject resume when validators, templates, plugins or dependencies changed."""
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix(): digest(p.read_text(encoding="utf-8"))
             for p in sorted(root.rglob("*")) if p.suffix in {".py", ".json", ".txt"}}
    return {"source_sha256": digest(json.dumps(files, sort_keys=True)), "plugins": plugins(),
            "python": platform.python_version(),
            "distributions": sorted([d.metadata["Name"], d.version] for d in importlib.metadata.distributions())}


def provider_identity(llm):
    result = {k: getattr(llm, k, None) for k in ("mode", "model", "timeout", "max_output_tokens", "max_attempts")}
    # Endpoints can contain credentials; bind their digest, never their raw URL.
    result["endpoint_sha256"] = digest(str(getattr(llm, "endpoint", getattr(llm, "base_url", ""))))
    return result


