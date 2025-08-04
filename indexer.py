from __future__ import annotations
import os, json, pickle, re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language  

load_dotenv()

REPO_PATH         = Path(os.getenv("REPO_PATH", ".")).resolve()
MODEL_NAME        = os.getenv("EMBEDDING_MODEL", "buelfhood/unixcoder-base-unimodal-ST")
VECTOR_INDEX_FILE = os.getenv("VECTOR_INDEX_FILE", "repo_index.faiss")
CHUNKS_FILE       = os.getenv("CHUNKS_FILE", "repo_chunks.pkl")
BM25_INDEX_FILE   = os.getenv("BM25_INDEX_FILE", "repo_bm25_index.pkl")
SKIP_DIRS         = set(os.getenv("SKIP_DIRS", "__pycache__,.git,venv,.env").split(","))

EXT_LANGUAGE = {
    ".py":  get_language("python"),
    ".js":  get_language("javascript"),
    ".jsx": get_language("javascript"),
    ".mjs": get_language("javascript"),
    ".cjs": get_language("javascript"),
    ".ts":  get_language("typescript"),
    ".tsx": get_language("typescript"),
}

PARSERS: Dict[str, Parser] = {}
def parser_for(ext: str) -> Parser:
    if ext not in PARSERS:
        lang = EXT_LANGUAGE[ext]
        p = Parser()
        p.set_language(lang)
        PARSERS[ext] = p
    return PARSERS[ext]

print(f"📂  Repository: {REPO_PATH}")
print(f"🔎  Embedding model: {MODEL_NAME}")


def node_text(src: bytes, n: Node) -> str:
    return src[n.start_byte : n.end_byte].decode("utf8", "ignore")

def identifier_child(n: Node) -> Node | None:
    return n.child_by_field_name("name")

PY_DOCSTRING_RE = re.compile(br'^[ru]*[f]?("""|\'\'\')([\s\S]*?)\1', re.M)

def first_docstring(src: bytes, n: Node, is_python: bool) -> str | None:
    if not is_python:
    
        jsdoc = re.search(rb"/\*\*([^*]|\*(?!/))*\*/", src[: n.start_byte])
        if jsdoc:
            line = jsdoc.group().decode("utf8").split("\n")[0]
            return line.strip("/* ").strip()
        return None

    body = n.child_by_field_name("body")
    if body and body.child_count:
        first = body.child(0)
        if first.type == "expression_statement":
            txt = node_text(src, first)
            m = PY_DOCSTRING_RE.match(txt.encode())
            if m:
                return m.group(2).decode("utf8").strip()
    return None

def get_imports_js(src_bytes: bytes) -> List[str]:
    text = src_bytes.decode("utf8", "ignore")
    out = re.findall(r"^\s*import[^\n]+|^\s*export[^\n]+|require\([^)]*\)", text, flags=re.M)
    return out[:20]

def get_imports_py(src_bytes: bytes) -> List[str]:
    text = src_bytes.decode("utf8", "ignore")
    out = re.findall(r"^\s*(from\s+[^\n]+|import\s+[^\n]+)", text, flags=re.M)
    return out[:20]

def interesting_nodes(node: Node, ext: str) -> List[Node]:
    js_ts_types = {
        "function_declaration", "method_definition",
        "arrow_function", "generator_function",
        "class_declaration", "interface_declaration", "enum_declaration",
    }
    py_types = {"function_definition", "class_definition"}
    target = js_ts_types if ext != ".py" else py_types
    acc = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in target:
            acc.append(n)
        stack.extend(n.children)
    return acc

def build_chunk(node: Node, src: bytes, path: Path, ext: str, file_imports: List[str]) -> Dict:
    is_py = ext == ".py"
    kind_map = {
        "function_definition": "function",
        "function_declaration": "function",
        "arrow_function": "function",
        "generator_function": "function",
        "method_definition": "method",
        "class_definition": "class",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
    }
    ctype = kind_map.get(node.type, node.type)
    name_node = identifier_child(node) or node
    name = node_text(src, name_node).split()[0] if name_node else "anonymous"

    parent_class = None
    if ctype in {"method"} and node.parent:
    
        p = node.parent
        while p and p.type not in ("class_definition", "class_declaration"):
            p = p.parent
        if p:
            parent_class = node_text(src, identifier_child(p)).split()[0]

    doc = first_docstring(src, node, is_py)
    params = []
    if ext == ".py" and ctype == "function":
    
        sig = node.child_by_field_name("parameters")
        if sig:
            for child in sig.children:
                if child.type == "identifier":
                    params.append(node_text(src, child))
    elif ext != ".py" and "function" in ctype:
    
        sig = node.child_by_field_name("parameters")
        if sig:
            ids = [node_text(src, c) for c in sig.children if c.type == "identifier"]
            params += ids

    return {
        "file_path": str(path),
        "type": ctype,
        "name": name,
        "parent_class": parent_class,
        "parameters": params,
        "code": node_text(src, node),
        "docstring": doc,
        "imports": file_imports,
        "metadata": {
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "full_name": f"{parent_class+'.' if parent_class else ''}{name}",
        },
    }

def create_embedding_text(chunk: Dict) -> str:
    pieces = [f"{chunk['type']} {chunk['name']}"]
    if chunk.get("parent_class"):
        pieces.append(f"in class {chunk['parent_class']}")
    if chunk.get("parameters"):
        pieces.append(f"parameters: {', '.join(chunk['parameters'])}")
    if chunk.get("docstring"):
        pieces.append(f"description: {chunk['docstring'][:150]}")
    if chunk.get("imports"):
        pieces.append(f"imports: {', '.join(chunk['imports'][:5])}")
    preview = " ".join(chunk["code"].splitlines()[:3]).strip()
    pieces.append(f"code: {preview[:100]}{'...' if len(preview)>100 else ''}")
    return " | ".join(pieces)

all_chunks: List[Dict] = []

for path in REPO_PATH.rglob("*"):
    if path.is_dir():
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        continue
    ext = path.suffix.lower()
    if ext not in EXT_LANGUAGE:
        continue
    if any(skip in path.parts for skip in SKIP_DIRS):
        continue

    try:
        src_bytes = path.read_bytes()
        prs = parser_for(ext)
        tree = prs.parse(src_bytes)

        file_imports = (
            get_imports_py(src_bytes) if ext == ".py" else get_imports_js(src_bytes)
        )

        for node in interesting_nodes(tree.root_node, ext):
            all_chunks.append(build_chunk(node, src_bytes, path, ext, file_imports))

    except Exception as e:
        print(f"⚠️  Skipped {path}: {e}")

print(f"✅  Extracted {len(all_chunks)} chunks from {len({c['file_path'] for c in all_chunks})} files")

print("💡  Generating embeddings …")
model = SentenceTransformer(MODEL_NAME)
emb_texts = [create_embedding_text(c) for c in all_chunks]
embeddings = model.encode(emb_texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)

faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings.astype("float32"))
faiss.write_index(faiss_index, VECTOR_INDEX_FILE)
print(f"💾  FAISS written → {VECTOR_INDEX_FILE}")

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(all_chunks, f)
print(f"💾  Chunks metadata → {CHUNKS_FILE}")

bm25_corpus = [
    " ".join(
        filter(
            None,
            [chunk["code"], chunk.get("docstring", ""), chunk["name"], chunk.get("parent_class", "")],
        )
    )
    for chunk in all_chunks
]
tokenized = [doc.split() for doc in bm25_corpus]
bm25 = BM25Okapi(tokenized)

with open(BM25_INDEX_FILE, "wb") as f:
    pickle.dump(bm25, f)
print(f"💾  BM25 index → {BM25_INDEX_FILE}")

stats = {
    "total_chunks": len(all_chunks),
    "classes": len([c for c in all_chunks if c["type"] == "class"]),
    "functions": len([c for c in all_chunks if c["type"] == "function"]),
    "methods": len([c for c in all_chunks if c["type"] == "method"]),
    "interfaces": len([c for c in all_chunks if c["type"] == "interface"]),
    "enums": len([c for c in all_chunks if c["type"] == "enum"]),
}
print(json.dumps(stats, indent=2))
print("🎉  INDEXING COMPLETE")
