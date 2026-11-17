"""Pure source parsing; no database writes and no language-model calls."""
import ast


def parse_source(suffix: str, source: str) -> list[dict]:
    facts = []
    if suffix == ".py":
        tree = ast.parse(source)
        dependencies = sorted({n.names[0].name if isinstance(n, ast.Import) else n.module or ""
                               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))})
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                facts.append(dict(title=n.name, text=ast.get_docstring(n) or n.name,
                                  line=n.lineno, evidence_quote=ast.get_source_segment(source, n),
                                  dependencies=dependencies, extraction="python_ast"))
    else:
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("## "):
                end = next((j for j in range(index + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
                quote = "\n".join(lines[index:end]).strip()
                facts.append(dict(title=line[3:], text=quote, line=index + 1,
                                  evidence_quote=quote, extraction="markdown_sections"))
    return facts
