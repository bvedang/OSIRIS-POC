# POC 1
from __future__ import annotations

import io
import os
import re
import pickle
import faiss
from pathlib import Path 
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from lsp_spike import LspClient
from collections import defaultdict
from unidiff import PatchSet
import ast, tokenize

FUNC_RE   = re.compile(r'^\s*(async\s+)?def\s+([A-Za-z_]\w*)\s*\(')
CLASS_RE  = re.compile(r'^\s*class\s+([A-Za-z_]\w*)\s*[:\(]')
FROM_IMPORT_RE = re.compile(r'^\s*from\s+((?:\.[\.]*)?[A-Za-z_][\w.]*)\s+import')
IMPORT_RE = re.compile(r'^\s*import\s+([A-Za-z_][\w.]*)')


def ast_locator(path: str, name:str, kind):
    """
    Return (line, col) of a function/class definition using `ast`.
    Works with decorators, async defs, nested classes.
    """
    source = Path(path).read_text()
    tree = ast.parse(source, filename=path)


    class Finder(ast.NodeVisitor):
        def __init__(self):
            self.loc = None
        
        def visit_FunctionDef(self, node):
            if kind == "function" and node.name == name:
                self.loc = (node.lineno - 1, node.col_offset)
            self.generic_visit(node)
        def visit_AsyncFunctionDef(self, node): 
            if kind == "function" and node.name == name:
                self.loc = (node.lineno - 1, node.col_offset)
            self.generic_visit(node)
        def visit_ClassDef(self, node):
            if kind == "class" and node.name == name:
                self.loc = (node.lineno - 1, node.col_offset)
            self.generic_visit(node)

    finder = Finder(); finder.visit(tree)
    return finder.loc            


def extract_import_modules(line):
    """Extract module names from import statements"""
    modules = set()
    
    if m := FROM_IMPORT_RE.match(line):
        module = m.group(1)
        if not module.startswith('.'):  
            modules.add(module.split('.')[0])
    # Check 'import X, Y, Z'
    elif m := IMPORT_RE.match(line):
        import_list = line[m.start(1):]
        for item in re.split(r'\s*,\s*', import_list):
            item = item.strip().split()[0] 
            if item:
                modules.add(item.split('.')[0])
    
    return modules

def find_symbol_location(file_path, symbol_name, symbol_type='def'):
    """Finds the line and character of a symbol definition."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        pattern = f"{symbol_type} {symbol_name}"
        for i, line in enumerate(lines):
            if pattern in line:
                char_pos = line.find(symbol_name)
                return i, char_pos
    except FileNotFoundError:
        return None, None
    return None, None

def extract_identifiers_from_diff(diff_text: str) -> dict:
    """Enhanced version with more detailed tracking"""
    result = defaultdict(set)
    file_changes = defaultdict(lambda: {
        'funcs_added': set(), 'funcs_removed': set(),
        'classes_added': set(), 'classes_removed': set(),
        'imports_added': set(), 'imports_removed': set()
    })
    
    for pfile in PatchSet(io.StringIO(diff_text)):
        path = re.sub(r'^[ab]/', '', pfile.target_file or pfile.source_file or '')
        
        # Track file-level changes
        if pfile.is_added_file:
            result['files_added'].add(path)
        elif pfile.is_removed_file:
            result['files_deleted'].add(path)
        elif pfile.is_rename:
            old = re.sub(r'^[ab]/', '', pfile.source_file)
            result['files_renamed'].add((old, path))
        
        result['modified_files'].add(path)
        
        # Parse content changes
        for hunk in pfile:
            for line in hunk:
                txt = line.value.rstrip()
                
                # Determine operation
                if line.is_added:
                    op = 'added'
                elif line.is_removed:
                    op = 'removed'
                else:
                    continue
                
                # Extract identifiers
                if m := FUNC_RE.match(txt):
                    name = m.group(2)
                    result[f'functions_{op}'].add(name)
                    file_changes[path][f'funcs_{op}'].add(name)
                elif m := CLASS_RE.match(txt):
                    name = m.group(1)
                    result[f'classes_{op}'].add(name)
                    file_changes[path][f'classes_{op}'].add(name)
                else:
                    # Check for imports
                    for module in extract_import_modules(txt):
                        result[f'imports_{op}'].add(module)
                        file_changes[path][f'imports_{op}'].add(module)
    
    # Calculate modified (in both added and removed)
    for category in ['functions', 'classes', 'imports']:
        added = result[f'{category}_added']
        removed = result[f'{category}_removed']
        modified = added & removed
        result[f'{category}_modified'] = modified
        result[f'{category}_added'] -= modified
        result[f'{category}_removed'] -= modified
    
    result['file_changes'] = dict(file_changes)
    return dict(result)

def gather_references_for_symbols(lsp_client, identifiers, repo_path, all_chunks):
    """Gather references for functions and classes found in the diff."""
    all_references = defaultdict(list)
    
    # Get all functions
    all_functions = (identifiers.get('functions_removed', set()) | 
                    identifiers.get('functions_modified', set()) |
                    identifiers.get('functions_added', set()))
    
    # Get all classes
    all_classes = (identifiers.get('classes_removed', set()) |
                  identifiers.get('classes_added', set()))

    # Process functions
    for func_name in all_functions:
        print(f"Looking for references to function: {func_name}")
        
        # Find the function in our indexed chunks
        for chunk in all_chunks:
            if chunk['type'] == 'function' and chunk['name'] == func_name:
                file_path = chunk['file_path']
                line, char = find_symbol_location(file_path, func_name, 'def')
                
                if line is not None:
                    try:
                        refs = lsp_client.get_references(file_path, line, char)
                        if refs and refs.get('result'):
                            all_references[func_name].extend(refs['result'])
                    except Exception as e:
                        print(f"Error getting references for {func_name}: {e}")
    
    # Process classes  
    for class_name in all_classes:
        print(f"Looking for references to class: {class_name}")
        
        for chunk in all_chunks:
            if chunk['type'] == 'class' and chunk['name'] == class_name:
                file_path = chunk['file_path']
                line, char = find_symbol_location(file_path, class_name, 'class')
                
                if line is not None:
                    try:
                        refs = lsp_client.get_references(file_path, line, char)
                        if refs and refs.get('result'):
                            all_references[class_name].extend(refs['result'])
                    except Exception as e:
                        print(f"Error getting references for {class_name}: {e}")
    
    return all_references

def format_references_context(all_references, repo_path):
    """Format references into a readable context string."""
    if not all_references:
        return "No references found for any functions or classes in the diff."
    
    formatted_parts = []
    formatted_parts.append("## Code References Analysis:")
    
    for name, refs in all_references.items():
        if refs:
            # Remove duplicates based on file and line
            unique_refs = {}
            for ref in refs:
                key = (ref['uri'], ref['range']['start']['line'])
                unique_refs[key] = ref
            
            formatted_parts.append(f"\n### `{name}` is referenced in:")
            
            # Group by file
            refs_by_file = defaultdict(list)
            for ref in unique_refs.values():
                file_path = Path(ref['uri'].replace('file://', ''))
                try:
                    rel_path = file_path.relative_to(repo_path)
                except:
                    rel_path = file_path
                refs_by_file[str(rel_path)].append(ref['range']['start']['line'] + 1)
            
            # Format references
            for file_path, lines in refs_by_file.items():
                lines_str = ', '.join(f"L{line}" for line in sorted(lines))
                formatted_parts.append(f"- {file_path}: {lines_str}")
                
                # Show snippet for first 2 usages
                try:
                    full_path = repo_path / file_path
                    if full_path.exists():
                        with open(full_path, 'r') as f:
                            file_lines = f.readlines()
                        
                        for line_num in sorted(lines)[:2]:
                            if 0 < line_num <= len(file_lines):
                                code_line = file_lines[line_num - 1].strip()
                                formatted_parts.append(f"    {line_num}: {code_line}")
                except Exception:
                    pass  # Skip if can't read file
    
    return "\n".join(formatted_parts)

# --- Main Script ---
print("Configuring environment...")
load_dotenv()

# Setup API
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    exit()

# Load indexes
print("Loading indexes...")
vector_index = faiss.read_index("repo.faiss")
with open("repo_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)
with open("repo_bm25.pkl", "rb") as f:
    bm25_index = pickle.load(f)
    
model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("Indexes loaded.")

# Your PR diff
with open('diff.txt', 'r') as f:
    pr_diff = f.read()
    # print(diff_content)

# Semantic search
k = 3
query_embedding = model_encoder.encode([pr_diff])
distances, indices = vector_index.search(query_embedding, k)
vector_search_results = [all_chunks[i] for i in indices[0]]

# Keyword search
tokenized_query = pr_diff.split()
keyword_search_results = bm25_index.get_top_n(tokenized_query, all_chunks, n=k)

# Combine results
retrieved_context = {}
for result in vector_search_results + keyword_search_results:
    key = (result['file_path'], result['name'])
    retrieved_context[key] = result['code']

context_str = "\n---\n".join(retrieved_context.values())

# Extract identifiers from diff
identifiers = extract_identifiers_from_diff(pr_diff)
print(identifiers)
print(f"\nExtracted from diff:")
print(f"- Functions added: {identifiers.get('functions_added', set())}")
print(f"- Functions removed: {identifiers.get('functions_removed', set())}")
print(f"- Functions modified: {identifiers.get('functions_modified', set())}")
print(f"- Classes added: {identifiers.get('classes_added', set())}")
print(f"- Classes removed: {identifiers.get('classes_removed', set())}")
print(f"- Imports: {identifiers.get('imports', set())}")
print(f"- Modified files: {identifiers.get('modified_files', set())}")

# Initialize LSP and gather references
repo_path = Path("/Users/vedangbarhate/Desktop/github_projects/rag").resolve()
lsp_context_str = "LSP analysis could not be performed."

lsp = LspClient(repo_path)
try:
    lsp.start()
    lsp.initialize()
    
    # Gather references for diff symbols
    all_references = gather_references_for_symbols(lsp, identifiers, repo_path, all_chunks)
    
    # Also check semantic search results
    for chunk in vector_search_results + keyword_search_results:
        if chunk['type'] == 'function' and chunk['name'] not in all_references:
            func_name = chunk['name']
            file_path = chunk['file_path']
            line, char = find_symbol_location(file_path, func_name, 'def')
            
            if line is not None:
                try:
                    refs = lsp.get_references(file_path, line, char)
                    if refs and refs.get('result'):
                        all_references[func_name].extend(refs['result'])
                        print(f"Found references for semantic result: {func_name}")
                except Exception as e:
                    print(f"Error getting references for {func_name}: {e}")
    
    # Format LSP context
    lsp_context_str = format_references_context(all_references, repo_path)
    
except Exception as e:
    lsp_context_str = f"LSP analysis failed: {e}"
    import traceback
    traceback.print_exc()
finally:
    lsp.shutdown()

# Generate review with Gemini
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
print(lsp_context_str)
# Your existing prompt...
prompt = f"""
You are **CodeReview-GPT**, an expert senior software engineer who delivers fast, high-signal pull-request reviews.
Your mission: catch critical issues, improve long-term maintainability, and mentor developers—all without unnecessary noise.

## Persona
- Thorough and detail-oriented  
- Constructive, explaining the *why* behind recommendations  
- Pragmatic: balances best practices with real-world constraints  
- Encourages learning by giving concrete examples  

## Communication Style
- Clear and direct (no fluff)  
- Consistent Markdown formatting for easy scanning  
- Brief positive notes *only* when they provide learning value  
- Never speculate on unseen code—comment only on what you can observe in the diff or supplied context  

---

## Context Supplied to You
```text
1. Semantic search results from the repository (may include similar code):
<existing_code>
{context_str}
</existing_code>

2. Code-reference analysis (where functions/classes are used):
<lsp_analysis>
{lsp_context_str}
</lsp_analysis>

````

---

## Task

Review the following code change:

```diff
{pr_diff}
```

---

## Review Process

### 1. **Initial Triage**

| Decision             | When to choose                                                                                 |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| **\[APPROVED]**      | Change is purely cosmetic (formatting, comments, typos, variable rename with no logic impact). |
| **\[NEEDS\_REVIEW]** | Any change to logic, algorithms, data flow, API, tests, docs, performance, or security.        |

### 2. **Deep Review** (only if *NEEDS\_REVIEW*)

Assess each area **strictly on the diff & provided context**:

1. **Correctness** – logic errors, off-by-one, edge cases
2. **Safety** – error handling, resource / lifecycle management
3. **Security & Privacy** – input validation, auth, sensitive data exposure
4. **Breaking Changes** – public API or contract alterations
5. **Performance** – algorithmic complexity, large allocations, N+1 queries
6. **Tests** – adequate unit/integration tests for new or changed logic
7. **Documentation** – public APIs, complex algorithms, migration notes
8. **Code Quality** – DRY, SOLID where valuable, clear naming, structure
9. **Consistency** – conforms to *Team-Specific Conventions* above
10. **Cross-File Impact** – use `lsp_analysis` to verify all call-sites updated

*Skip style nits handled by auto-formatters.*

---

## Output Format (Markdown)

````markdown
## Triage Status
**[NEEDS_REVIEW]** or **[APPROVED]**

## Review

### 🚨 BLOCKING  
*Must fix: bugs, security flaws, data loss risks.*

#### [Issue Title]
- **Location**: `file_path:line_start-line_end`
- **Issue**: What is wrong
- **Impact**: Consequence if unfixed
- **Fix**:
  ```language
  // Concrete code example or approach
````

### ⚠️ IMPORTANT

*Should fix: maintainability, performance, substantial quality issues.*

#### \[Issue Title]

* **Location**: `file_path:line_start-line_end`
* **Issue**: …
* **Why fix**: Value gained
* **Suggestion**: Approach or snippet

### 💡 CONSIDER

*Nice-to-have improvements, refactors, docs.*

* **Location**: Brief suggestion (one bullet per idea)

---

**Summary**
One or two sentences on overall quality and key concerns.

(If no issues at all, respond with exactly:)
`# ✅ LGTM!`
`No issues found. Good use of <specific positive observation>.`

````

---

## Examples (shortened)

```markdown
### 🚨 BLOCKING
#### Null Dereference on Error Path
- **Location**: `user_service.py:42-48`
- **Issue**: `user` may be `None` if lookup fails, then `.email` dereferences null.
- **Impact**: Runtime crash under load.
- **Fix**:
  ```python
  user = repo.get(user_id)
  if user is None:
      raise NotFoundError(user_id)
````

### ⚠️ IMPORTANT

#### Missing Unit Tests for New Parser

* **Location**: `parser/` (entire module)
* **Issue**: New code lacks tests for malformed input paths.
* **Why fix**: Prevent regressions and undefined behavior.
* **Suggestion**: Add tests covering empty, exotic Unicode, and >1 MB inputs.

```

---

### Important Rules (reminders)

1. **Reference real lines**; don’t invent code.  
2. **Only comment on observed issues**; no speculation.  
3. **Keep it actionable**—issue, impact, fix.  
4. **Be concise**; skip boilerplate praise.  
5. **Follow output format exactly**; otherwise reviewers waste time re-parsing.

---

*End of system prompt.*
"""


print("\n" + "="*50)
print("CALLING GEMINI API...")
print("="*50)

try:
    response = gemini_model.generate_content(prompt)
    print("\n🤖 GEMINI REVIEW:")
    print("-" * 20)
    print(response.text)
    print("-" * 20)
except Exception as e:
    print(f"\n--- An error occurred while calling the Gemini API ---")
    print(f"Error: {e}")
    print("This could be due to an invalid API key, network issues, or content safety filters.")
