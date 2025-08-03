#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multistep Context-Rich GitHub PR Reviewer
=========================================

A two-phase PR review system:
1. First LLM call: Analyze diff and identify focus areas
2. Targeted context retrieval based on analysis
3. Second LLM call: Detailed review with relevant context

This approach allows for more intelligent context selection and 
deeper analysis of complex changes.
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
from dataclasses import dataclass
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
# DATA STRUCTURES
# ============================================================================

@dataclass
class AnalysisResult:
    """Result from first LLM analysis pass."""
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    focus_areas: List[str]  # Key areas to investigate
    specific_symbols: List[str]  # Specific functions/classes to check
    search_queries: List[str]  # Additional search queries
    key_concerns: List[str]  # Main concerns identified
    file_patterns: List[str]  # File patterns to look for

# ============================================================================
# DIFF PARSING AND IDENTIFIER EXTRACTION (reuse from original)
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
# REUSE HELPER FUNCTIONS FROM ORIGINAL
# ============================================================================

def ast_locator(path: str, name: str, kind: str) -> Optional[Tuple[int, int]]:
    """Find the line and column of a symbol using AST parsing."""
    if not Path(path).exists():
        return None
        
    try:
        src = Path(path).read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(src, filename=path)
    except (SyntaxError, FileNotFoundError, OSError):
        return None
    
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.loc = None
        
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


def build_symbol_map(chunks: List[dict]) -> Dict[str, List[dict]]:
    """Build a map from symbol names to their chunk definitions for O(1) lookup."""
    symbol_map: Dict[str, List[dict]] = defaultdict(list)
    for chunk in chunks:
        if name := chunk.get('name'):
            symbol_map[name].append(chunk)
    return dict(symbol_map)


def prime_lsp(lsp: LspClient, paths: Iterable[str]) -> None:
    """Prime the LSP server by opening files."""
    for path in paths:
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
    """Gather cross-references for all symbols using LSP."""
    refs: Dict[str, list] = defaultdict(list)
    new_files = identifiers.get('files_added', set())
    
    all_functions = (
        identifiers.get('functions_modified', set()) |
        identifiers.get('functions_removed', set())
    )
    
    all_classes = (
        identifiers.get('classes_modified', set()) | 
        identifiers.get('classes_removed', set())
    )
    
    for func_name in all_functions:
        for chunk in symbol_map.get(func_name, []):
            if chunk['type'] != 'function':
                continue
            
            if chunk['file_path'] in new_files:
                print(f"Skipping {func_name} in new file: {chunk['file_path']}")
                continue
                
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
    
    for class_name in all_classes:
        for chunk in symbol_map.get(class_name, []):
            if chunk['type'] != 'class':
                continue
            
            if chunk['file_path'] in new_files:
                print(f"Skipping {class_name} in new file: {chunk['file_path']}")
                continue
                
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


def reciprocal_rank_fusion(
    rank_lists: List[List[int]], 
    k: int = 60
) -> List[int]:
    """Combine multiple rankings using Reciprocal Rank Fusion."""
    scores: Counter[int] = Counter()
    
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank)
    
    return [idx for idx, _ in scores.most_common()]


def estimate_tokens(text: str) -> int:
    """Estimate token count with tiktoken or fallback."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model('gpt-4')
        return len(enc.encode(text))
    except (ImportError, Exception):
        return len(text) // 4


def fit_to_token_budget(
    sections: List[str], 
    max_tokens: int = DEFAULT_TOKEN_BUDGET
) -> List[str]:
    """Trim content to fit within token budget."""
    kept = []
    total_tokens = 0
    
    for section in sections:
        tokens = estimate_tokens(section)
        if total_tokens + tokens > max_tokens:
            break
        kept.append(section)
        total_tokens += tokens
    
    return kept


def format_references_context(
    all_refs: Dict[str, list], 
    repo_root: Path
) -> str:
    """Format LSP references into a readable context string."""
    if not all_refs:
        return "No cross-references found."
    
    output = ["## Code Reference Analysis:"]
    
    for symbol, refs in all_refs.items():
        if not refs:
            continue
            
        unique_refs = {}
        for ref in refs:
            key = (ref['uri'], ref['range']['start']['line'])
            unique_refs[key] = ref
        
        output.append(f"\n### `{symbol}` is referenced in:")
        
        refs_by_file: Dict[str, List[int]] = defaultdict(list)
        for ref in unique_refs.values():
            file_path = Path(ref['uri'].replace('file://', ''))
            try:
                rel_path = file_path.relative_to(repo_root)
            except ValueError:
                rel_path = file_path
            
            line_num = ref['range']['start']['line'] + 1
            refs_by_file[str(rel_path)].append(line_num)
        
        for file_path, lines in sorted(refs_by_file.items()):
            lines_str = ', '.join(f"L{line}" for line in sorted(lines)[:10])
            if len(lines) > 10:
                lines_str += f" (+{len(lines) - 10} more)"
            output.append(f"- {file_path}: {lines_str}")
    
    return "\n".join(output)

# ============================================================================
# STEP 1: INITIAL ANALYSIS PROMPT
# ============================================================================

def build_analysis_prompt(diff_text: str, identifiers: Dict[str, Set[str]]) -> str:
    """Build prompt for first LLM call to analyze the PR."""
    
    # Summarize changes
    summary_parts = []
    
    if identifiers.get('files_added'):
        summary_parts.append(f"New files: {', '.join(list(identifiers['files_added'])[:5])}")
    if identifiers.get('files_deleted'):
        summary_parts.append(f"Deleted files: {', '.join(list(identifiers['files_deleted'])[:5])}")
    if identifiers.get('functions_added'):
        summary_parts.append(f"New functions: {', '.join(list(identifiers['functions_added'])[:5])}")
    if identifiers.get('functions_modified'):
        summary_parts.append(f"Modified functions: {', '.join(list(identifiers['functions_modified'])[:5])}")
    if identifiers.get('classes_added'):
        summary_parts.append(f"New classes: {', '.join(list(identifiers['classes_added'])[:5])}")
    if identifiers.get('classes_modified'):
        summary_parts.append(f"Modified classes: {', '.join(list(identifiers['classes_modified'])[:5])}")
    
    change_summary = "\n".join(summary_parts) if summary_parts else "Minor changes"
    
    return f"""You are a senior software engineer analyzing a pull request to identify key areas of concern.
Your goal is to quickly assess the risk level and determine what context would be most valuable for a thorough review.

## Summary of Changes:
{change_summary}

## The Diff:
```diff
{diff_text[:3000]}{"..." if len(diff_text) > 3000 else ""}
```

## Your Task:
Analyze this PR and provide a structured analysis in the following JSON format:

```json
{{
    "risk_level": "HIGH|MEDIUM|LOW",
    "focus_areas": [
        "Area 1 to investigate deeply",
        "Area 2 to investigate deeply"
    ],
    "specific_symbols": [
        "function_or_class_name_to_examine",
        "another_symbol"
    ],
    "search_queries": [
        "specific search query for context",
        "another search query"
    ],
    "key_concerns": [
        "Main concern about this change",
        "Another concern"
    ],
    "file_patterns": [
        "test_*.py",
        "*/config/*"
    ]
}}
```

## Guidelines:
- **HIGH risk**: Breaking changes, security issues, major architectural changes, database migrations
- **MEDIUM risk**: New features, significant refactoring, performance-sensitive changes
- **LOW risk**: Documentation, minor bug fixes, cosmetic changes

Focus on identifying:
1. What existing code might be affected by these changes
2. What patterns or anti-patterns you see
3. What additional context would help review this properly
4. Specific symbols (functions/classes) that need deep inspection

Be specific in your search queries and symbol names. The goal is to gather the most relevant context for a thorough review.

Respond ONLY with the JSON object, no additional text."""


def parse_analysis_result(response_text: str) -> AnalysisResult:
    """Parse the JSON response from the analysis LLM call."""
    try:
        # Extract JSON from response (in case LLM adds extra text)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
        
        return AnalysisResult(
            risk_level=data.get('risk_level', 'MEDIUM'),
            focus_areas=data.get('focus_areas', []),
            specific_symbols=data.get('specific_symbols', []),
            search_queries=data.get('search_queries', []),
            key_concerns=data.get('key_concerns', []),
            file_patterns=data.get('file_patterns', [])
        )
    except Exception as e:
        print(f"Error parsing analysis result: {e}")
        # Return default analysis if parsing fails
        return AnalysisResult(
            risk_level='MEDIUM',
            focus_areas=['General code quality', 'Error handling'],
            specific_symbols=[],
            search_queries=['error handling patterns', 'similar implementations'],
            key_concerns=['Unable to parse initial analysis'],
            file_patterns=[]
        )

# ============================================================================
# ENHANCED CONTEXT GATHERING
# ============================================================================

def targeted_search(
    analysis: AnalysisResult,
    identifiers: Dict[str, Set[str]],
    chunks: List[dict],
    bm25_index,
    vector_index,
    encoder: SentenceTransformer,
    symbol_map: Dict[str, List[dict]]
) -> List[int]:
    """
    Perform targeted search based on initial analysis results.
    
    Returns indices of most relevant chunks.
    """
    all_results = []
    
    # 1. Search for specific symbols mentioned in analysis
    for symbol in analysis.specific_symbols:
        if symbol in symbol_map:
            for chunk in symbol_map[symbol]:
                if chunk in chunks:
                    all_results.append(chunks.index(chunk))
    
    # 2. Execute custom search queries from analysis
    for query in analysis.search_queries:
        # BM25 search
        docs = bm25_index.get_top_n(
            query.split(), 
            chunks, 
            n=10
        )
        all_results.extend([chunks.index(doc) for doc in docs])
        
        # Vector search
        query_embedding = encoder.encode([query])
        distances, indices = vector_index.search(query_embedding, 10)
        all_results.extend(indices[0].tolist())
    
    # 3. Look for files matching patterns
    for pattern in analysis.file_patterns:
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        for i, chunk in enumerate(chunks):
            if re.match(pattern_re, chunk['file_path']):
                all_results.append(i)
    
    # 4. Add context based on focus areas
    focus_queries = []
    for area in analysis.focus_areas:
        # Convert focus area to search query
        focus_queries.append(area.lower())
    
    for query in focus_queries:
        docs = bm25_index.get_top_n(
            query.split(), 
            chunks, 
            n=5
        )
        all_results.extend([chunks.index(doc) for doc in docs])
    
    # Deduplicate and rank by frequency
    result_counts = Counter(all_results)
    ranked_indices = [idx for idx, _ in result_counts.most_common(MAX_CONTEXT_SNIPPETS * 2)]
    
    return ranked_indices[:MAX_CONTEXT_SNIPPETS]

# ============================================================================
# STEP 2: DETAILED REVIEW PROMPT
# ============================================================================

def build_detailed_review_prompt(
    diff_text: str,
    context_sections: List[str],
    lsp_context: str,
    analysis: AnalysisResult
) -> str:
    """Build prompt for second LLM call with targeted context."""
    
    context_blob = "\n---\n".join(context_sections) if context_sections else "No context found."
    
    # Format concerns and focus areas
    concerns_text = "\n".join(f"- {concern}" for concern in analysis.key_concerns)
    focus_text = "\n".join(f"- {area}" for area in analysis.focus_areas)
    
    return f"""You are **CodeReview-GPT**, an expert senior software engineer providing a detailed code review.
Based on initial analysis, this PR has been classified as **{analysis.risk_level} RISK**.

## Key Concerns Identified:
{concerns_text}

## Areas Requiring Deep Review:
{focus_text}

## Context (gathered based on initial analysis)
### 1. Relevant Code Context:
```text
{context_blob}
```

### 2. Cross-Reference Analysis (LSP):
```text
{lsp_context}
```

## The Complete Diff:
```diff
{diff_text}
```

## Your Task:
Provide a thorough code review focusing especially on the identified concerns and risk areas.

### Review Guidelines:
- **Tone**: Professional but friendly, like a helpful senior developer
- **Focus**: Pay special attention to the concerns and focus areas identified above
- **Depth**: Since this is {analysis.risk_level} risk, {"be extremely thorough" if analysis.risk_level == "HIGH" else "be thorough but efficient"}
- **Constructive**: Explain why issues matter and how to fix them
- **Actionable**: Every comment should help improve the code

### Check for:
1. **Critical Issues** related to the identified concerns
2. **Logic Errors & Edge Cases** - especially in the focus areas
3. **Security & Safety** - {("CRITICAL for this change" if analysis.risk_level == "HIGH" else "if applicable")}
4. **Breaking Changes** - API compatibility, backwards compatibility
5. **Performance** - especially if dealing with loops, queries, or large data
6. **Test Coverage** - are the risky areas properly tested?
7. **Error Handling** - proper exception handling and recovery
8. **Code Quality** - maintainability, clarity, following best practices
9. **Cross-cutting Concerns** - how changes affect other parts of the codebase

## Output Format:

Start with one of these statuses:
- **🚨 NEEDS_WORK**: Critical issues found that must be fixed
- **⚠️ NEEDS_REVIEW**: Issues found that should be addressed
- **✅ APPROVED**: Only minor or no issues found

Then provide your detailed review:

For each issue, use this format:
### 🔴 [Severity] Issue Title (`file:line`)
**What**: Brief description of the problem
**Why it matters**: Impact if not fixed  
**How to fix**: Concrete suggestion or code example

Severity levels:
- 🔴 **Critical**: Must fix before merge (security, data loss, crashes)
- 🟡 **Major**: Should fix (bugs, performance issues)
- 🔵 **Minor**: Consider fixing (code quality, maintainability)

End with a **Summary** section that:
1. Recaps the most important issues
2. Provides clear next steps
3. Acknowledges what was done well (if applicable)

Remember: Be thorough but constructive. The goal is to help ship better code, not to nitpick."""

# ============================================================================
# MAIN ORCHESTRATION - MULTISTEP VERSION
# ============================================================================

def main():
    """Main entry point for the multistep PR reviewer."""
    
    # Track total execution time
    total_start = time.time()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file")
        return
    
    genai.configure(api_key=api_key)
    
    print("=== Multistep Context-Rich PR Reviewer ===")
    print("Loading indices...")
    
    # Time index loading
    load_start = time.time()
    try:
        # Load search indices
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
          f"{len(identifiers.get('functions_modified', set()))} modified")
    print(f"  - Classes: {len(identifiers.get('classes_added', set()))} added, "
          f"{len(identifiers.get('classes_modified', set()))} modified")
    print(f"  - Files: {len(identifiers.get('modified_files', set()))} modified")
    
    # ========================================================================
    # STEP 1: Initial Analysis (First LLM Call)
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 1: Initial PR Analysis")
    print("="*60)
    
    analysis_prompt = build_analysis_prompt(diff_text, identifiers)
    
    print(f"Calling {LLM_MODEL} for initial analysis...")
    llm1_start = time.time()
    model = genai.GenerativeModel(LLM_MODEL)
    
    try:
        response = model.generate_content(
            analysis_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,  # Lower temp for structured output
                max_output_tokens=1024,
            )
        )
        llm1_time = time.time() - llm1_start
        
        print(f"Analysis completed in {llm1_time:.2f}s")
        
        # Parse analysis results
        analysis = parse_analysis_result(response.text)
        
        print(f"\nAnalysis Results:")
        print(f"  Risk Level: {analysis.risk_level}")
        print(f"  Key Concerns: {len(analysis.key_concerns)}")
        print(f"  Focus Areas: {', '.join(analysis.focus_areas[:3])}")
        print(f"  Specific Symbols to Check: {', '.join(analysis.specific_symbols[:5])}")
        
    except Exception as e:
        print(f"Error in initial analysis: {e}")
        traceback.print_exc()
        # Fallback to basic analysis
        analysis = AnalysisResult(
            risk_level='MEDIUM',
            focus_areas=['General code quality'],
            specific_symbols=[],
            search_queries=['related implementations'],
            key_concerns=['Failed to analyze, doing general review'],
            file_patterns=[]
        )
    
    # ========================================================================
    # STEP 2: Targeted Context Gathering
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: Targeted Context Gathering")
    print("="*60)
    
    # Perform targeted search based on analysis
    print("Performing targeted search based on analysis...")
    search_start = time.time()
    
    top_indices = targeted_search(
        analysis,
        identifiers,
        chunks,
        bm25_index,
        vector_index,
        encoder,
        symbol_map
    )
    
    search_time = time.time() - search_start
    print(f"Retrieved {len(top_indices)} targeted code snippets in {search_time:.2f}s")
    
    # Gather LSP references with focus on high-risk symbols
    print("\nGathering code references via LSP...")
    lsp_start = time.time()
    repo_root = Path.cwd().resolve()
    lsp_context = "LSP analysis unavailable."
    
    lsp = LspClient(repo_root)
    try:
        lsp.start()
        lsp.initialize()
        
        # Prime files based on targeted search results
        relevant_files = {chunks[i]['file_path'] for i in top_indices}
        relevant_files.update(identifiers.get('modified_files', set()))
        
        # Add files containing specific symbols from analysis
        for symbol in analysis.specific_symbols:
            for chunk in symbol_map.get(symbol, []):
                relevant_files.add(chunk['file_path'])
        
        # Remove new files that don't exist yet
        new_files = identifiers.get('files_added', set())
        relevant_files -= new_files
        
        # Prime LSP
        prime_start = time.time()
        print(f"Priming LSP with {len(relevant_files)} files...")
        prime_lsp(lsp, relevant_files)
        prime_time = time.time() - prime_start
        
        # Gather references
        refs_start = time.time()
        refs_map = gather_references(lsp, identifiers, symbol_map)
        lsp_context = format_references_context(refs_map, repo_root)
        refs_time = time.time() - refs_start
        
        lsp_time = time.time() - lsp_start
        print(f"LSP analysis completed in {lsp_time:.2f}s")
        
    except Exception as e:
        print(f"LSP error: {e}")
        traceback.print_exc()
    finally:
        try:
            lsp.shutdown()
        except:
            pass
    
    # Prepare context sections
    context_snippets = []
    
    # Add targeted snippets
    for i in top_indices:
        chunk = chunks[i]
        # Add header to explain why this chunk was selected
        header = f"## {chunk['file_path']} - {chunk.get('type', 'code')} {chunk.get('name', '')}"
        context_snippets.append(f"{header}\n{chunk['code']}")
    
    # Fit to token budget (leave room for the second prompt)
    context_trimmed = fit_to_token_budget(context_snippets, DEFAULT_TOKEN_BUDGET)
    
    print(f"\nContext prepared: {len(context_trimmed)} snippets")
    
    # ========================================================================
    # STEP 3: Detailed Review (Second LLM Call)
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: Detailed Code Review")
    print("="*60)
    
    # Build detailed review prompt
    review_prompt = build_detailed_review_prompt(
        diff_text,
        context_trimmed,
        lsp_context,
        analysis
    )
    
    print(f"Calling {LLM_MODEL} for detailed review...")
    llm2_start = time.time()
    
    try:
        response = model.generate_content(
            review_prompt,
        )
        llm2_time = time.time() - llm2_start
        
        print(f"Detailed review completed in {llm2_time:.2f}s")
        print("\n" + "="*60)
        print("FINAL REVIEW RESULTS")
        print("="*60)
        print(response.text)
        print("="*60)
        
    except Exception as e:
        print(f"Error in detailed review: {e}")
        traceback.print_exc()
    
    # ========================================================================
    # Performance Summary
    # ========================================================================
    total_time = time.time() - total_start
    print(f"\n=== Performance Summary (Multistep) ===")
    print(f"Index loading: {load_time:.2f}s")
    print(f"Diff parsing: {extract_time:.2f}s")
    print(f"Initial analysis (LLM 1): {llm1_time:.2f}s")
    print(f"Targeted search: {search_time:.2f}s")
    print(f"LSP operations: {lsp_time:.2f}s")
    print(f"Detailed review (LLM 2): {llm2_time:.2f}s")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f}s")
    
    # Show what the multistep approach achieved
    print(f"\n=== Multistep Benefits ===")
    print(f"Risk-based analysis: {analysis.risk_level}")
    print(f"Targeted {len(analysis.specific_symbols)} specific symbols")
    print(f"Custom searches: {len(analysis.search_queries)} queries")
    print(f"Focus areas: {len(analysis.focus_areas)} identified")


if __name__ == "__main__":
    main()