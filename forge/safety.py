"""Conservative linting only. NOT a security boundary or hostile-code sandbox."""
import ast

ALLOWED_IMPORTS = {"numpy", "pandas", "sklearn"}
DENIED_NAMES = {"eval", "exec", "compile", "open", "input", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "breakpoint", "exit", "quit"}


def inspect_code(code: str):
    errors = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not {"build_model", "predict"} <= functions:
        errors.append("Required top-level functions: build_model, predict")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name.split(".")[0] not in ALLOWED_IMPORTS for n in node.names):
                errors.append("Import outside numpy/pandas/sklearn allowlist")
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS):
            errors.append("Import outside numpy/pandas/sklearn allowlist")
        if isinstance(node, ast.Name) and (node.id in DENIED_NAMES or node.id.startswith("__")):
            errors.append(f"Prohibited name: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in {"to_csv", "to_pickle", "read_csv", "load", "dump", "save", "read_pickle", "to_sql"}):
            errors.append(f"Prohibited attribute: {node.attr}")
    return sorted(set(errors))
