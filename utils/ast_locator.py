from __future__ import annotations
import ast as py_ast
from pathlib import Path
from typing import Optional, Tuple

from tree_sitter import Parser, Node
from tree_sitter_languages import get_language

_TS_LANG = get_language("typescript")    
_JS_LANG = get_language("javascript")    
_LANG_BY_EXT = {
    ".ts":  _TS_LANG, ".tsx": _TS_LANG,
    ".js":  _JS_LANG, ".jsx": _JS_LANG, ".mjs": _JS_LANG, ".cjs": _JS_LANG,
}

_PARSER_CACHE: dict[str, Parser] = {}


def _get_parser(ext: str) -> Parser:
    if ext not in _PARSER_CACHE:
        parser = Parser()
        parser.set_language(_LANG_BY_EXT[ext])
        _PARSER_CACHE[ext] = parser
    return _PARSER_CACHE[ext]


def _ts_locator(root: Node, want_kind: str, want_name: str) -> Optional[Tuple[int, int]]:
    """Recursive DFS to find the first node matching kind+name."""
    FN_TYPES = {
        "function": {
            "function_declaration", "method_definition",
            "arrow_function", "generator_function",
        },
        "class": {"class_declaration", "class"},
    }
    if root.type in FN_TYPES.get(want_kind, set()):
        for child in root.children:
            if child.type == "identifier" and child.text.decode() == want_name:
                return child.start_point  
        if root.parent and root.parent.type == "variable_declarator":
            ident = root.parent.child_by_field_name("name")
            if ident and ident.text.decode() == want_name:
                return ident.start_point
    for ch in root.children:
        hit = _ts_locator(ch, want_kind, want_name)
        if hit:
            return hit
    return None


def ast_locator(path: str, name: str, kind: str) -> Optional[Tuple[int, int]]:
    file = Path(path)
    if not file.exists():
        return None

    ext = file.suffix.lower()

    if ext == ".py":
        try:
            src = file.read_text(encoding="utf-8", errors="ignore")
            tree = py_ast.parse(src, filename=path)
        except (py_ast.AST, OSError, SyntaxError):
            return None

        class _Visitor(py_ast.NodeVisitor):
            def __init__(self):
                self.loc = None

            def visit_FunctionDef(self, node):
                if kind == "function" and node.name == name and self.loc is None:
                    self.loc = (node.lineno - 1, node.col_offset)
                self.generic_visit(node)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                if kind == "class" and node.name == name and self.loc is None:
                    self.loc = (node.lineno - 1, node.col_offset)
                self.generic_visit(node)

        v = _Visitor()
        v.visit(tree)
        return v.loc

    if ext in _LANG_BY_EXT:
        try:
            parser = _get_parser(ext)
            bytes_src = file.read_bytes()
            tree = parser.parse(bytes_src)
            return _ts_locator(tree.root_node, kind, name)
        except Exception:
            return None

    return None
