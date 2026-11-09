"""Explicit host-side fault injection for a repair demonstration, never hidden."""
import ast


def inject_unknown_category_fault(code):
    tree = ast.parse(code)
    edits = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == "OneHotEncoder") or
                                        (isinstance(node.func, ast.Attribute) and node.func.attr == "OneHotEncoder")):
            for keyword in node.keywords:
                if keyword.arg == "handle_unknown":
                    keyword.value = ast.Constant("error")
                    break
            else:
                node.keywords.append(ast.keyword(arg="handle_unknown", value=ast.Constant("error")))
            edits += 1
    if not edits:
        raise ValueError("Repair demo requires OneHotEncoder; no fault was injected")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
