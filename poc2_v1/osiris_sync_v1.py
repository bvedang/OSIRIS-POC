#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context-Rich GitHub PR Reviewer (Production Ready)
================================================

A comprehensive PR review system that combines:
- O(1) symbol lookup for performance
- AST-based accurate symbol location
- Intelligent LSP integration with selective file priming
- Hybrid retrieval (BM25 + FAISS with RRF fusion)
- Token budget management
- LLM-powered review generation

Prerequisites:
- repo.faiss: FAISS index of code embeddings
- repo_chunks.pkl: List of code chunks with metadata
- repo_bm25.pkl: BM25 index for keyword search
- diff.txt: Unified diff of the PR to review
- .env file with GOOGLE_API_KEY

Dependencies:
pip install faiss-cpu sentence-transformers unidiff google-generativeai \
            python-lsp-server pylsp-jsonrpc tiktoken python-dotenv
"""

from __future__ import annotations

import ast
import io
import json
import os
import pickle
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from unidiff import PatchSet

from lsp_spike import LspClient

# ============================================================================
# CONFIGURATION
# ============================================================================

# Regex patterns for code parsing
FUNC_RE = re.compile(r"^\s*(async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[:\(]")
FROM_IMPORT_RE = re.compile(r"^\s*from\s+((?:\.[\.]*)?[A-Za-z_][\w.]*)\s+import")
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)")

# Model configuration
EMBEDDING_MODEL = 'microsoft/unixcoder-base'
LLM_MODEL = 'gemini-2.5-pro'
DEFAULT_TOKEN_BUDGET = 1800
MAX_SEARCH_RESULTS = 30
MAX_CONTEXT_SNIPPETS = 12

# ============================================================================
# DIFF PARSING AND IDENTIFIER EXTRACTION
# ============================================================================

def extract_import_modules(line: str) -> Set[str]:
    """Extract top-level module names from import statements."""
    modules: Set[str] = set()
    
    if m := FROM_IMPORT_RE.match(line):
        module = m.group(1)
        if not module.startswith('.'):  # Skip relative imports
            modules.add(module.split('.')[0])
    elif m := IMPORT_RE.match(line):
        import_segment = line[m.start(1):]
        for item in re.split(r'\s*,\s*', import_segment):
            item = item.strip().split()[0]
            if item:
                modules.add(item.split('.')[0])
    
    return modules


def extract_identifiers_from_diff(diff_text: str) -> Dict[str, Set[str]]:
    """
    Extract all identifiers (functions, classes, imports) from a unified diff.
    
    Returns a dict with keys like:
    - functions_added, functions_removed, functions_modified
    - classes_added, classes_removed, classes_modified
    - imports_added, imports_removed, imports_modified
    - files_added, files_deleted, files_renamed, modified_files
    - file_changes: granular per-file change tracking
    """
    result: Dict[str, Set[str]] = defaultdict(set)
    file_changes: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: {
            'funcs_added': set(), 'funcs_removed': set(),
            'classes_added': set(), 'classes_removed': set(),
            'imports_added': set(), 'imports_removed': set()
        }
    )
    
    for pfile in PatchSet(io.StringIO(diff_text)):
        path = re.sub(r'^[ab]/', '', pfile.target_file or pfile.source_file or '')
        
        # Track file-level changes
        if pfile.is_added_file:
            result['files_added'].add(path)
        elif pfile.is_removed_file:
            result['files_deleted'].add(path)
        elif pfile.is_rename:
            old = re.sub(r'^[ab]/', '', pfile.source_file)
            result['files_renamed'].add(f"{old}→{path}")
        
        result['modified_files'].add(path)
        
        # Parse content changes
        for hunk in pfile:
            for line in hunk:
                txt = line.value.rstrip('\n')
                
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
    
    # Calculate modified items (present in both added and removed)
    for category in ['functions', 'classes', 'imports']:
        added = result[f'{category}_added']
        removed = result[f'{category}_removed']
        modified = added & removed
        
        result[f'{category}_modified'] = modified
        result[f'{category}_added'] -= modified
        result[f'{category}_removed'] -= modified
    
    result['file_changes'] = dict(file_changes)
    return dict(result)

# ============================================================================
# AST-BASED SYMBOL LOCATION
# ============================================================================

def ast_locator(path: str, name: str, kind: str) -> Optional[Tuple[int, int]]:
    """
    Find the line and column of a symbol using AST parsing.
    Handles decorators, async functions, and nested classes correctly.
    
    Args:
        path: File path to search
        name: Symbol name to find
        kind: 'function' or 'class'
    
    Returns:
        (line, column) tuple or None if not found or file doesn't exist
    """
    # Check if file exists first
    if not Path(path).exists():
        return None
        
    try:
        src = Path(path).read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(src, filename=path)
    except (SyntaxError, FileNotFoundError, OSError):
        return None
    
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.loc = None  # Instance variable to avoid sharing
        
        def visit_FunctionDef(self, node):
            if kind == 'function' and node.name == name and self.loc is None:
                self.loc = (node.lineno - 1, node.col_offset)
            self.generic_visit(node)
        
        def visit_AsyncFunctionDef(self, node):
            if kind == 'function' and node.name == name and self.loc is None:
                self.loc = (node.lineno - 1, node.col_offset)
            self.generic_visit(node)
        
        def visit_ClassDef(self, node):
            if kind == 'class' and node.name == name and self.loc is None:
                self.loc = (node.lineno - 1, node.col_offset)
            self.generic_visit(node)
    
    v = Visitor()
    v.visit(tree)
    return v.loc

# ============================================================================
# SYMBOL MAP FOR O(1) LOOKUPS
# ============================================================================

def build_symbol_map(chunks: List[dict]) -> Dict[str, List[dict]]:
    """Build a map from symbol names to their chunk definitions for O(1) lookup."""
    symbol_map: Dict[str, List[dict]] = defaultdict(list)
    for chunk in chunks:
        if name := chunk.get('name'):
            symbol_map[name].append(chunk)
    return dict(symbol_map)

# ============================================================================
# LSP INTEGRATION
# ============================================================================

def prime_lsp(lsp: LspClient, paths: Iterable[str]) -> None:
    """
    Prime the LSP server by opening files.
    This ensures reliable reference/diagnostic results.
    Only opens files that actually exist.
    """
    for path in paths:
        # Skip non-existent files (e.g., new files in PR)
        if not Path(path).exists():
            print(f"Skipping non-existent file for LSP: {path}")
            continue
            
        try:
            lsp.open_file(path)
        except Exception as e:
            print(f"Warning: Failed to prime LSP for {path}: {e}")


def gather_references(
    lsp: LspClient,
    identifiers: Dict[str, Set[str]],
    symbol_map: Dict[str, List[dict]]
) -> Dict[str, list]:
    """
    Gather cross-references for all symbols using LSP.
    
    Returns a dict mapping symbol names to their reference locations.
    """
    refs: Dict[str, list] = defaultdict(list)
    
    # Get new files to skip (they don't exist in the repo yet)
    new_files = identifiers.get('files_added', set())
    
    # Collect all symbols to check
    all_functions = (
        identifiers.get('functions_modified', set()) |
        identifiers.get('functions_removed', set())
    )
    
    all_classes = (
        identifiers.get('classes_modified', set()) | 
        identifiers.get('classes_removed', set())
    )
    
    # Process functions
    for func_name in all_functions:
        for chunk in symbol_map.get(func_name, []):
            if chunk['type'] != 'function':
                continue
            
            # Skip if file is new (doesn't exist in repo)
            if chunk['file_path'] in new_files:
                print(f"Skipping {func_name} in new file: {chunk['file_path']}")
                continue
                
            # Check if file exists before trying to read it
            if not Path(chunk['file_path']).exists():
                print(f"Warning: File not found: {chunk['file_path']}")
                continue
                
            loc = ast_locator(chunk['file_path'], func_name, 'function')
            if loc:
                try:
                    line, col = loc
                    response = lsp.get_references(chunk['file_path'], line, col)
                    if response and response.get('result'):
                        refs[func_name].extend(response['result'])
                except Exception as e:
                    print(f"Warning: LSP error for {func_name}: {e}")
    
    # Process classes
    for class_name in all_classes:
        for chunk in symbol_map.get(class_name, []):
            if chunk['type'] != 'class':
                continue
            
            # Skip if file is new (doesn't exist in repo)
            if chunk['file_path'] in new_files:
                print(f"Skipping {class_name} in new file: {chunk['file_path']}")
                continue
                
            # Check if file exists before trying to read it
            if not Path(chunk['file_path']).exists():
                print(f"Warning: File not found: {chunk['file_path']}")
                continue
                
            loc = ast_locator(chunk['file_path'], class_name, 'class')
            if loc:
                try:
                    line, col = loc
                    response = lsp.get_references(chunk['file_path'], line, col)
                    if response and response.get('result'):
                        refs[class_name].extend(response['result'])
                except Exception as e:
                    print(f"Warning: LSP error for {class_name}: {e}")
    
    return dict(refs)

# ============================================================================
# RETRIEVAL AND RANKING
# ============================================================================

def build_queries(diff_text: str, identifiers: Dict[str, Set[str]]) -> List[str]:
    """
    Build search queries from diff and extracted identifiers.
    
    Returns:
        List of query strings, prioritizing identifier-based search
    """
    # Collect all relevant identifiers
    all_identifiers = set()
    for key in identifiers:
        if key.startswith(('functions', 'classes', 'imports')):
            all_identifiers.update(identifiers[key])
    
    # Query 1: Identifier bag (most important)
    identifier_query = " ".join(sorted(all_identifiers))
    
    # Query 2: Diff header/context
    # Extract meaningful words from diff, avoiding noise
    diff_words = [w for w in diff_text.split() if len(w) > 3][:200]
    header_query = " ".join(diff_words)
    
    queries = []
    if identifier_query.strip():
        queries.append(identifier_query)
    if header_query.strip():
        queries.append(header_query)
    
    return queries


def reciprocal_rank_fusion(
    rank_lists: List[List[int]], 
    k: int = 60
) -> List[int]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.
    
    Args:
        rank_lists: List of rankings, each ranking is a list of indices
        k: RRF parameter (default 60)
    
    Returns:
        Fused ranking as a list of indices
    """
    scores: Counter[int] = Counter()
    
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank)
    
    return [idx for idx, _ in scores.most_common()]

# ============================================================================
# TOKEN BUDGET MANAGEMENT
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count with tiktoken or fallback."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model('gpt-4')
        return len(enc.encode(text))
    except (ImportError, Exception):
        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4


def fit_to_token_budget(
    sections: List[str], 
    max_tokens: int = DEFAULT_TOKEN_BUDGET
) -> List[str]:
    """
    Trim content to fit within token budget.
    
    Args:
        sections: List of text sections
        max_tokens: Maximum token budget
    
    Returns:
        List of sections that fit within budget
    """
    kept = []
    total_tokens = 0
    
    for section in sections:
        tokens = estimate_tokens(section)
        if total_tokens + tokens > max_tokens:
            break
        kept.append(section)
        total_tokens += tokens
    
    return kept

# ============================================================================
# REFERENCE FORMATTING
# ============================================================================

def format_references_context(
    all_refs: Dict[str, list], 
    repo_root: Path
) -> str:
    """
    Format LSP references into a readable context string.
    
    Includes deduplication and path normalization.
    """
    if not all_refs:
        return "No cross-references found."
    
    output = ["## Code Reference Analysis:"]
    
    for symbol, refs in all_refs.items():
        if not refs:
            continue
            
        # Deduplicate by file and line
        unique_refs = {}
        for ref in refs:
            key = (ref['uri'], ref['range']['start']['line'])
            unique_refs[key] = ref
        
        output.append(f"\n### `{symbol}` is referenced in:")
        
        # Group by file
        refs_by_file: Dict[str, List[int]] = defaultdict(list)
        for ref in unique_refs.values():
            file_path = Path(ref['uri'].replace('file://', ''))
            try:
                rel_path = file_path.relative_to(repo_root)
            except ValueError:
                rel_path = file_path
            
            line_num = ref['range']['start']['line'] + 1
            refs_by_file[str(rel_path)].append(line_num)
        
        # Format references
        for file_path, lines in sorted(refs_by_file.items()):
            lines_str = ', '.join(f"L{line}" for line in sorted(lines)[:10])
            if len(lines) > 10:
                lines_str += f" (+{len(lines) - 10} more)"
            output.append(f"- {file_path}: {lines_str}")
    
    return "\n".join(output)

# ============================================================================
# PROMPT GENERATION
# ============================================================================

def build_review_prompt(
    diff_text: str,
    context_sections: List[str],
    lsp_context: str
) -> str:
    """Build the complete prompt for the LLM review."""
    
    context_blob = "\n---\n".join(context_sections) if context_sections else "No context found."
    
    return f"""You are **CodeReview-GPT**, an expert senior software engineer providing high-quality code reviews.
                Your goal: catch critical issues, improve code maintainability, and mentor the code author — **all in a friendly, concise manner**.

                ## Persona & Tone
                - **Approachable Mentor:** You review as a colleague who is supportive and constructive. Tone is friendly and casual-professional (like a helpful senior dev).
                - **Thorough & Detail-Oriented:** You don’t miss logic errors, edge cases, or bad practices.
                - **Constructive:** Explain *why* each issue matters, and *how* to improve it, in a straightforward way:contentReference[oaicite:15]{{index=15}}.
                - **Pragmatic:** Focus on important issues (logic, bugs, security, performance), not minor style nitpicks (assume linters handle formatting).
                - **Positive:** If you see something well-done or an opportunity to praise good practice, briefly acknowledge it genuinely:contentReference[oaicite:16]{{index=16}}. (E.g. “Nice job handling X here.”) This makes the review balanced and human.
                - **No AI-jargon or filler:** Write as you naturally would in conversation. Use simple language and contractions:contentReference[oaicite:17]{{index=17}}. It’s okay to say "we should..." or "let’s..." when suggesting changes. Avoid phrases like “As an AI, ...”.

                ## Communication Guidelines
                - **Clear and Direct:** Get to the point without waffling:contentReference[oaicite:18]{{index=18}}. Each comment should be 1-3 sentences, focusing on the key issue and fix.
                - **Conversational Style:** Use first-person (“I noticed…”, “We could…”) and address the author as “you” when needed. This keeps the tone personable. Example: *“I think this function might not handle empty input. We should add a check for that to avoid a crash.”*
                - **Empathetic and Professional:** No harsh language. Even for mistakes, the tone is *“This is an issue, here’s how to fix it”* rather than scolding. You are on the same team as the author.
                - **No Speculation on Unseen Code:** Only comment on the given diff and context. If something is unclear, you can pose it as a question or suggestion, not an assumption.
                - **Consistent Markdown Formatting:** Use Markdown for easy reading (lists, code blocks for examples, etc.). 
                - **Snippet (if helpful):** Include a tiny example or diff that illustrates the fix.  
                ```python
                # Before
                if user == None:
                    ...

                # After
                if user is None:
                    ...
                ```
                _(You can show either a “before/after” pair or just the corrected version; keep it under ~10 lines.)_
                ---

                ## Context (for your reference)
                <sup>*The following sections provide context. They may include search results or file content for reference. Use them to inform your review, but do not assume beyond them.*</sup>

                ### Search Results:
                ```text
                {context_blob}
                ```
                ### 2. Additional Code Context (LSP Analysis):
                ```text
                {lsp_context}
                ```
                **Use the above context to understand the code change and its implications across the codebase.**

                ---

                ## Task
                You will now review the following code diff. Identify any problems or improvements following the criteria below.

                ```diff
                {diff_text}
                ```
                Scope: Focus strictly on the diff and provided context.
                ---

                ## Review Criteria

                ### Determine the Triage Status first:
                - If this change is purely cosmetic (comments, formatting, typos, variable renames with no logic change), then it's [APPROVED].
                - Otherwise (any logic/behavior changes, test changes, etc.), it's [NEEDS_REVIEW] for deeper inspection.

                ### For NEEDS_REVIEW changes, consider:
                Assess each area **strictly on the diff & provided context**:

                1. **Correctness & Edge Cases** - Any logic errors, off-by-one issues, or missing edge case handling?
                2. **Safety & Error Handling** - Exceptions, resource leaks, proper cleanup, etc.
                3. **Security & Privacy** - Vulnerabilities, input validation, secrets exposed?
                4. **Breaking Changes** - Does it alter an API or contract in a problematic way?
                5. **Performance** - Any inefficiencies (N+1 queries, slow algorithms) that could matter?
                6. **Tests** - Are there changes to tests or missing tests for new logic?
                7. **Documentation** - Should any part of this be documented or commented for clarity?
                8. **Code Quality** - Is the code clear, maintainable, following good practices (DRY, SOLID, etc.)?
                9. **Consistency** - Does it follow the project’s conventions and patterns?
                10. **Cross-File Impact** - Given the context, are all usage sites or related code updated accordingly?

                ----------

                ## Output Format

                Your review will be in Markdown and include:
                -   **Triage Status**:
                    -   Begin with **NEEDSREVIEWNEEDS_REVIEW** or **APPROVEDAPPROVED** on its own line (with bold formatting).
                -   **Review Comments**: Only if **NEEDS_REVIEW**. For each issue or suggestion, write a bullet point or subheading with a short title and explanation:
                    -   **Location**: Indicate the file and line numbers in parentheses after the title, e.g. _(`utils.js:45-52`)_, so the author knows where it applies.
                    -   **Issue & Suggestion**: In one or two sentences, describe the problem and why it matters, then suggest how to resolve it. Be concise and specific. _For example:_
                        -   🚨 **Potential Memory Leak** (`server.py:88-95`) - The file handle opened here isn’t closed. This could exhaust OS file descriptors over time. **Suggestion:** use a context manager (`with open(...)`) so it closes automatically.
                        -   ⚠️ **Inefficient Loop** (`data.js:130`) - The nested loop will slow down with large input sizes. It’s not a bug, but consider using a lookup table to reduce the complexity from O(n^2) to O(n).
                        -   💡 **Clarify Magic Number** (`config.yml:22`) - The value `42` is used without explanation. It might help future readers to define a named constant (e.g., `MAX_RETRIES = 42`) so the purpose is clear.
                    -   Keep each comment self-contained and avoid overly formal subsections. Write in a natural, conversational tone as demonstrated above.
                -   **Summary**: End with a short summary section (just a couple of sentences) summing up the overall assessment. If there are major issues, emphasize fixing them before merge. If the code is generally good, you can say it looks good overall aside from the listed items.
                
                **Important:**
                -   _Reference line numbers from the diff when explaining issues._                    
                -   _Only comment on visible changes or context given (don’t speculate about unseen code)._
                -   _Be **actionable**: the goal is to help the author improve the code._
                -   _Skip trivial style nitpicks (assume an auto-formatter will handle those)._
                ----------

                If **no issues at all** are found (i.e., the change is perfect): respond with a simple approval, for example:

                ```markdown
                # ✅ LGTM! 
                No issues found - the code looks good to merge. *Great job on keeping it clean and well-documented!* 
                ```
            """

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Main entry point for the PR reviewer."""
    
    # Track total execution time
    total_start = time.time()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file")
        return
    
    genai.configure(api_key=api_key)
    
    print("=== Context-Rich PR Reviewer (Synchronous) ===")
    print("Loading indices...")
    
    # Time index loading
    load_start = time.time()
    try:
        # Load search indices (sequentially)
        vector_index = faiss.read_index('repo_enhanced.faiss')
        with open('repo_chunks_enhanced.pkl', 'rb') as f:
            chunks = pickle.load(f)
        with open('repo_bm25_enhanced.pkl', 'rb') as f:
            bm25_index = pickle.load(f)
        
        # Build O(1) symbol map
        symbol_map = build_symbol_map(chunks)
        
        # Initialize embedding model
        encoder = SentenceTransformer(EMBEDDING_MODEL)
        
        load_time = time.time() - load_start
        print(f"Loaded {len(chunks)} code chunks in {load_time:.2f}s")
        
    except Exception as e:
        print(f"Error loading indices: {e}")
        return
    
    # Load diff
    try:
        diff_text = Path('diff.txt').read_text()
        print(f"Loaded diff ({len(diff_text)} chars)")
    except Exception as e:
        print(f"Error loading diff.txt: {e}")
        return
    
    # Extract identifiers
    print("\nExtracting identifiers from diff...")
    extract_start = time.time()
    identifiers = extract_identifiers_from_diff(diff_text)
    extract_time = time.time() - extract_start
    
    print(f"Found (in {extract_time:.2f}s):")
    print(f"  - Functions: {len(identifiers.get('functions_added', set()))} added, "
          f"{len(identifiers.get('functions_modified', set()))} modified, "
          f"{len(identifiers.get('functions_removed', set()))} removed")
    print(f"  - Classes: {len(identifiers.get('classes_added', set()))} added, "
          f"{len(identifiers.get('classes_modified', set()))} modified, "
          f"{len(identifiers.get('classes_removed', set()))} removed")
    print(f"  - Files: {len(identifiers.get('modified_files', set()))} modified")
    
    # Build queries and search
    print("\nPerforming hybrid search (sequential)...")
    search_start = time.time()
    queries = build_queries(diff_text, identifiers)
    
    # BM25 search (sequential)
    bm25_start = time.time()
    bm25_results = []
    for query in queries:
        docs = bm25_index.get_top_n(
            query.split(), 
            chunks, 
            n=MAX_SEARCH_RESULTS
        )
        bm25_results.extend([chunks.index(doc) for doc in docs])
    bm25_time = time.time() - bm25_start
    
    # Vector search (after BM25 completes)
    vector_start = time.time()
    query_text = queries[0] if queries else diff_text[:500]
    query_embedding = encoder.encode([query_text])
    distances, indices = vector_index.search(query_embedding, MAX_SEARCH_RESULTS)
    vector_results = indices[0].tolist()
    vector_time = time.time() - vector_start
    
    # Rank fusion
    fused_indices = reciprocal_rank_fusion([bm25_results, vector_results])
    top_indices = fused_indices[:MAX_CONTEXT_SNIPPETS]
    
    search_time = time.time() - search_start
    print(f"Retrieved {len(top_indices)} relevant code snippets in {search_time:.2f}s")
    print(f"  - BM25 search: {bm25_time:.2f}s")
    print(f"  - Vector search: {vector_time:.2f}s")
    
    # Gather LSP references
    print("\nGathering code references via LSP...")
    lsp_start = time.time()
    repo_root = Path.cwd().resolve()
    lsp_context = "LSP analysis unavailable."
    
    lsp = LspClient(repo_root)
    try:
        lsp.start()
        lsp.initialize()
        
        # Prime only relevant files
        relevant_files = {chunks[i]['file_path'] for i in top_indices}
        relevant_files.update(identifiers.get('modified_files', set()))
        
        # Remove new files that don't exist yet
        new_files = identifiers.get('files_added', set())
        relevant_files -= new_files
        
        # Time file priming (sequential)
        prime_start = time.time()
        print(f"Priming LSP with {len(relevant_files)} existing files (sequential)...")
        prime_lsp(lsp, relevant_files)
        prime_time = time.time() - prime_start
        print(f"  - File priming: {prime_time:.2f}s")
        
        # Gather references (sequential)
        refs_start = time.time()
        refs_map = gather_references(lsp, identifiers, symbol_map)
        lsp_context = format_references_context(refs_map, repo_root)
        refs_time = time.time() - refs_start
        
        lsp_time = time.time() - lsp_start
        print(f"Found references for {len(refs_map)} symbols in {lsp_time:.2f}s total")
        print(f"  - Reference gathering: {refs_time:.2f}s")
        
    except Exception as e:
        print(f"LSP error: {e}")
        traceback.print_exc()
    finally:
        try:
            lsp.shutdown()
        except:
            pass
    
    # Prepare context with token budget
    print("\nPreparing context for LLM...")
    context_snippets = [chunks[i]['code'] for i in top_indices]
    context_trimmed = fit_to_token_budget(context_snippets, DEFAULT_TOKEN_BUDGET)
    
    print(f"Context: {len(context_trimmed)} snippets within token budget")
    
    # Build prompt
    prompt = build_review_prompt(diff_text, context_snippets, lsp_context)
    
    # Call LLM
    print(f"\nCalling {LLM_MODEL}...")
    llm_start = time.time()
    model = genai.GenerativeModel(LLM_MODEL)
    
    try:
        response = model.generate_content(prompt)
        llm_time = time.time() - llm_start
        
        print(f"LLM response received in {llm_time:.2f}s")
        print("\n" + "="*60)
        print("REVIEW RESULTS")
        print("="*60)
        print(response.text)
        print("="*60)
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        traceback.print_exc()
    
    # Total execution time
    total_time = time.time() - total_start
    print(f"\n=== Performance Summary (Synchronous) ===")
    print(f"Index loading: {load_time:.2f}s")
    print(f"Diff parsing: {extract_time:.2f}s")
    print(f"Search operations: {search_time:.2f}s")
    print(f"  - BM25: {bm25_time:.2f}s")
    print(f"  - Vector: {vector_time:.2f}s")
    print(f"LSP operations: {lsp_time:.2f}s")
    print(f"  - File priming: {prime_time:.2f}s")
    print(f"  - Reference gathering: {refs_time:.2f}s")
    print(f"LLM generation: {llm_time:.2f}s")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f}s")


if __name__ == "__main__":
    main()