import os
from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import json
import faiss
from rank_bm25 import BM25Okapi
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

REPO_PATH = Path(os.getenv('REPO_PATH'))
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'buelfhood/unixcoder-base-unimodal-ST')
VECTOR_INDEX_FILE = os.getenv('VECTOR_INDEX_FILE', 'repo_index.faiss')
CHUNKS_FILE = os.getenv('CHUNKS_FILE', 'repo_chunks.pkl')
BM25_INDEX_FILE = os.getenv('BM25_INDEX_FILE', 'repo_bm25_index.pkl')
SKIP_DIRS = os.getenv('SKIP_DIRS', '__pycache__,.git,venv,.env').split(',')


print(f"Loading {MODEL_NAME} model...")
model = SentenceTransformer(MODEL_NAME)

def get_node_name(node):
    name_node = node.child_by_field_name('name')
    if name_node:
        return name_node.text.decode('utf8')
    return "unknown_name"

def get_docstring(node):
    body = node.child_by_field_name('body')
    if body and body.child_count > 0:
        first_stmt = body.child(0)
        if first_stmt.type == 'expression_statement':
            expr = first_stmt.child(0)
            if expr and expr.type == 'string':
                return expr.text.decode('utf8').strip('"""').strip("'''").strip()
    return None

def get_function_params(node):
    params = []
    params_node = node.child_by_field_name('parameters')
    if params_node:
        for child in params_node.children:
            if child.type == 'identifier':
                params.append(child.text.decode('utf8'))
            elif child.type == 'typed_parameter':
                param_name = child.child(0)
                if param_name and param_name.type == 'identifier':
                    params.append(param_name.text.decode('utf8'))
    return params

def get_imports(tree):
    imports = []
    def traverse_imports(node):
        if node.type == "import_statement" or node.type == 'import_from_statement':
            imports.append(node.text.decode('utf8'))
        for child in node.children:
            traverse_imports(child)
    traverse_imports(tree.root_node)
    return imports

def extract_chunks_enhanced(node, file_path, parent_class=None, file_imports=None):
    chunks = []
    
    if node.type == 'class_definition':
        class_name = get_node_name(node)
        docstring = get_docstring(node)
        
        chunk = {
            "file_path": str(file_path),
            "type": "class",
            "name": class_name,
            "code": node.text.decode('utf8'),
            "docstring": docstring,
            "imports": file_imports or [],
            "metadata": {
                "line_start": node.start_point[0],
                "line_end": node.end_point[0],
                "full_name": class_name
            }
        }
        chunks.append(chunk)
        
        body_node = node.child_by_field_name('body')
        if body_node:
            method_chunks = extract_chunks_enhanced(body_node, file_path, 
                                                   parent_class=class_name, 
                                                   file_imports=file_imports)
            chunks.extend(method_chunks)
            
    elif node.type == 'function_definition':
        func_name = get_node_name(node)
        docstring = get_docstring(node)
        params = get_function_params(node)
        
        full_name = f"{parent_class}.{func_name}" if parent_class else func_name
        
        chunk = {
            "file_path": str(file_path),
            "type": "method" if parent_class else "function",
            "name": func_name,
            "code": node.text.decode('utf8'),
            "docstring": docstring,
            "parameters": params,
            "parent_class": parent_class,
            "imports": file_imports or [],
            "metadata": {
                "line_start": node.start_point[0],
                "line_end": node.end_point[0],
                "full_name": full_name,
                "is_method": parent_class is not None
            }
        }
        chunks.append(chunk)
    
    for child in node.children:
        chunks.extend(extract_chunks_enhanced(child, file_path, 
                                            parent_class=parent_class,
                                            file_imports=file_imports))
    
    return chunks

def create_code_embedding_text(chunk):
    parts = []
    
    chunk_type = chunk['type']
    parts.append(f"{chunk_type} {chunk['name']}")
    
    if chunk.get('parent_class'):
        parts.append(f"in class {chunk['parent_class']}")
    
    if chunk.get('parameters'):
        params_str = ", ".join(chunk['parameters'])
        parts.append(f"parameters: {params_str}")
    
    if chunk.get('docstring'):
        parts.append(f"description: {chunk['docstring'][:200]}")
    
    if chunk.get('imports'):
        relevant_imports = []
        code = chunk['code']
        for imp in chunk['imports']:
            if 'import' in imp:
                module_name = imp.split()[-1].split('.')[0]
                if module_name in code:
                    relevant_imports.append(imp)
        if relevant_imports:
            parts.append(f"uses: {', '.join(relevant_imports[:5])}")
    
    code_lines = chunk['code'].split('\n')
    code_preview = ' '.join(code_lines[:3]).strip()
    if len(code_preview) > 100:
        code_preview = code_preview[:100] + "..."
    parts.append(f"code: {code_preview}")
    
    return " | ".join(parts)

all_chunks = []
print(f"Starting enhanced indexing for repository at: {REPO_PATH}")

for file_path in REPO_PATH.rglob("*.py"):
    if any(skip in str(file_path) for skip in SKIP_DIRS):
        continue
        
    try:
        source_code = file_path.read_bytes()
        tree = parser.parse(source_code)
        
        file_imports = get_imports(tree)
        
        file_chunks = extract_chunks_enhanced(tree.root_node, file_path, 
                                            file_imports=file_imports)
        all_chunks.extend(file_chunks)
        
    except Exception as e:
        print(f"Could not process {file_path}: {e}")

print(f"\nFound {len(all_chunks)} chunks in total.")

if all_chunks:
    print("\nSample chunk:")
    print(json.dumps(all_chunks[0], indent=2))

print(f"\n\nBuilding Vector Index with {MODEL_NAME} embeddings...")

embedding_texts = [create_code_embedding_text(chunk) for chunk in all_chunks]
code_contents = [chunk['code'] for chunk in all_chunks]

print("Generating embeddings...")
embeddings = model.encode(
    embedding_texts, 
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

d = embeddings.shape[1]
vector_index = faiss.IndexFlatL2(d)
vector_index.add(embeddings.astype('float32'))

print("\nSaving indices...")
faiss.write_index(vector_index, VECTOR_INDEX_FILE)

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(all_chunks, f)


print(f"FAISS index created with {vector_index.ntotal} vectors (dimension: {d}).")

print("\nBuilding Keyword Index...")

bm25_texts = []
for chunk in all_chunks:
    text_parts = [
        chunk['code'],
        chunk.get('docstring', ''),
        chunk['name'],
        chunk.get('parent_class', '')
    ]
    combined_text = ' '.join(filter(None, text_parts))
    bm25_texts.append(combined_text)

tokenized_corpus = [doc.split() for doc in bm25_texts]
bm25_index = BM25Okapi(tokenized_corpus)

with open(BM25_INDEX_FILE, "wb") as f:
    pickle.dump(bm25_index, f)

print("Enhanced BM25 index created.")

print("\n" + "="*50)
print("INDEXING COMPLETE")
print("="*50)

stats = {
    "total_chunks": len(all_chunks),
    "classes": len([c for c in all_chunks if c['type'] == 'class']),
    "functions": len([c for c in all_chunks if c['type'] == 'function']),
    "methods": len([c for c in all_chunks if c['type'] == 'method']),
    "files_processed": len(set(c['file_path'] for c in all_chunks)),
    "chunks_with_docstrings": len([c for c in all_chunks if c.get('docstring')])
}

print(f"Total chunks: {stats['total_chunks']}")
print(f"  - Classes: {stats['classes']}")
print(f"  - Functions: {stats['functions']}")
print(f"  - Methods: {stats['methods']}")
print(f"Files processed: {stats['files_processed']}")
print(f"Chunks with docstrings: {stats['chunks_with_docstrings']}")

print("\nIndex files created:")
print(f"  - {VECTOR_INDEX_FILE} (vector embeddings)")
print(f"  - {CHUNKS_FILE} (chunk metadata)")
print(f"  - {BM25_INDEX_FILE} (keyword index)")