from __future__ import annotations

import sys
import ast
import io
import json
import os
import pickle
import re
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from unidiff import PatchSet

from lsp_spike import LspClient

load_dotenv()

FUNC_RE = re.compile(r"^\s*(async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[:\(]")
FROM_IMPORT_RE = re.compile(r"^\s*from\s+((?:\.[\.]*)?[A-Za-z_][\w.]*)\s+import")
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)")

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'buelfhood/unixcoder-base-unimodal-ST')
LLM_MODEL = os.getenv('PR_LLM_MODEL', 'gemini-2.5-pro')
DEFAULT_TOKEN_BUDGET = int(os.getenv('PR_TOKEN_BUDGET', '1800'))
MAX_SEARCH_RESULTS = int(os.getenv('PR_MAX_SEARCH_RESULTS', '30'))
MAX_CONTEXT_SNIPPETS = int(os.getenv('PR_MAX_CONTEXT_SNIPPETS', '12'))
LLM_TEMPERATURE = float(os.getenv('PR_LLM_TEMPERATURE', '0.3'))


VECTOR_INDEX_FILE = os.getenv('VECTOR_INDEX_FILE', 'repo_enhanced.faiss')
CHUNKS_FILE = os.getenv('CHUNKS_FILE', 'repo_chunks_enhanced.pkl')
BM25_INDEX_FILE = os.getenv('BM25_INDEX_FILE', 'repo_bm25_enhanced.pkl')
MODEL_INFO_FILE = os.getenv('MODEL_INFO_FILE', 'embedding_model.txt')


@dataclass
class AnalysisResult:
    risk_level: str
    focus_areas: List[str]
    specific_symbols: List[str]
    search_queries: List[str]
    key_concerns: List[str]
    file_patterns: List[str]

@dataclass
class CodeBlock:
    content: str
    file_path: str
    block_type: str
    start_line: int
    end_line: int
    language: str = 'python'
    context: str = ''

@dataclass 
class ExtractedDiffContent:
    added_blocks: List[CodeBlock] = field(default_factory=list)
    removed_blocks: List[CodeBlock] = field(default_factory=list)
    modified_before: List[CodeBlock] = field(default_factory=list)
    modified_after: List[CodeBlock] = field(default_factory=list)
    
    def all_blocks(self) -> List[CodeBlock]:
        return (self.added_blocks + self.removed_blocks + 
                self.modified_before + self.modified_after)

def extract_code_blocks_from_diff(diff_text: str) -> ExtractedDiffContent:
    result = ExtractedDiffContent()
    
    for pfile in PatchSet(io.StringIO(diff_text)):
        file_path = re.sub(r'^[ab]/', '', pfile.target_file or pfile.source_file or '')
        
        for hunk in pfile:
            current_block = []
            current_op = None
            start_line = None
            
            for line_obj in hunk:
                line = line_obj.value.rstrip('\n')
                
                if line_obj.is_added:
                    op = 'added'
                elif line_obj.is_removed:
                    op = 'removed'
                else:
                    op = 'context'
                
                if op != current_op and current_block and current_op != 'context':
                    block = CodeBlock(
                        content='\n'.join(current_block),
                        file_path=file_path,
                        block_type=current_op,
                        start_line=start_line or 0,
                        end_line=start_line + len(current_block) - 1 if start_line else 0
                    )
                    
                    if current_op == 'added':
                        result.added_blocks.append(block)
                    elif current_op == 'removed':
                        result.removed_blocks.append(block)
                    
                    current_block = []
                
                if op != 'context':
                    if not current_block:
                        start_line = line_obj.target_line_no if op == 'added' else line_obj.source_line_no
                    current_block.append(line)
                    current_op = op
            
            if current_block and current_op != 'context':
                block = CodeBlock(
                    content='\n'.join(current_block),
                    file_path=file_path,
                    block_type=current_op,
                    start_line=start_line or 0,
                    end_line=start_line + len(current_block) - 1 if start_line else 0
                )
                
                if current_op == 'added':
                    result.added_blocks.append(block)
                elif current_op == 'removed':
                    result.removed_blocks.append(block)
    
    _identify_modified_blocks(result)
    
    return result

def _identify_modified_blocks(result: ExtractedDiffContent):
    by_file = defaultdict(lambda: {'added': [], 'removed': []})
    
    for block in result.added_blocks:
        by_file[block.file_path]['added'].append(block)
    
    for block in result.removed_blocks:
        by_file[block.file_path]['removed'].append(block)
    
    for file_path, blocks in by_file.items():
        added = blocks['added']
        removed = blocks['removed']
        
        if added and removed and len(added) == len(removed):
            for add_block, rem_block in zip(added, removed):
                if _blocks_are_related(add_block, rem_block):
                    result.modified_before.append(rem_block)
                    result.modified_after.append(add_block)
                    result.added_blocks.remove(add_block)
                    result.removed_blocks.remove(rem_block)

def _blocks_are_related(block1: CodeBlock, block2: CodeBlock) -> bool:
    names1 = set()
    names2 = set()
    
    for line in block1.content.split('\n'):
        if m := FUNC_RE.match(line):
            names1.add(m.group(2))
        elif m := CLASS_RE.match(line):
            names1.add(m.group(1))
    
    for line in block2.content.split('\n'):
        if m := FUNC_RE.match(line):
            names2.add(m.group(2))
        elif m := CLASS_RE.match(line):
            names2.add(m.group(1))
    
    return bool(names1 & names2)

def create_code_block_embedding_text(block: CodeBlock) -> str:
    parts = []
    
    parts.append(f"file: {block.file_path}")
    parts.append(f"change_type: {block.block_type}")
    
    functions = []
    classes = []
    imports = []
    
    for line in block.content.split('\n'):
        if m := FUNC_RE.match(line):
            functions.append(m.group(2))
        elif m := CLASS_RE.match(line):
            classes.append(m.group(1))
        elif FROM_IMPORT_RE.match(line) or IMPORT_RE.match(line):
            imports.append(line.strip())
    
    if functions:
        parts.append(f"functions: {', '.join(functions)}")
    if classes:
        parts.append(f"classes: {', '.join(classes)}")
    if imports:
        parts.append(f"imports: {'; '.join(imports[:3])}")
    
    code_preview = ' '.join(block.content.split('\n')[:5])
    if len(code_preview) > 200:
        code_preview = code_preview[:200] + "..."
    parts.append(f"code: {code_preview}")
    
    return " | ".join(parts)

def extract_import_modules(line: str) -> Set[str]:
    modules: Set[str] = set()
    
    if m := FROM_IMPORT_RE.match(line):
        module = m.group(1)
        if not module.startswith('.'):
            modules.add(module.split('.')[0])
    elif m := IMPORT_RE.match(line):
        import_segment = line[m.start(1):]
        for item in re.split(r'\s*,\s*', import_segment):
            item = item.strip().split()[0]
            if item:
                modules.add(item.split('.')[0])
    
    return modules


def extract_identifiers_from_diff(diff_text: str) -> Dict[str, Set[str]]:
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
        
        if pfile.is_added_file:
            result['files_added'].add(path)
        elif pfile.is_removed_file:
            result['files_deleted'].add(path)
        elif pfile.is_rename:
            old = re.sub(r'^[ab]/', '', pfile.source_file)
            result['files_renamed'].add(f"{old}→{path}")
        
        result['modified_files'].add(path)
        
        for hunk in pfile:
            for line in hunk:
                txt = line.value.rstrip('\n')
                
                if line.is_added:
                    op = 'added'
                elif line.is_removed:
                    op = 'removed'
                else:
                    continue
                
                if m := FUNC_RE.match(txt):
                    name = m.group(2)
                    result[f'functions_{op}'].add(name)
                    file_changes[path][f'funcs_{op}'].add(name)
                elif m := CLASS_RE.match(txt):
                    name = m.group(1)
                    result[f'classes_{op}'].add(name)
                    file_changes[path][f'classes_{op}'].add(name)
                else:
                    for module in extract_import_modules(txt):
                        result[f'imports_{op}'].add(module)
                        file_changes[path][f'imports_{op}'].add(module)
    
    for category in ['functions', 'classes', 'imports']:
        added = result[f'{category}_added']
        removed = result[f'{category}_removed']
        modified = added & removed
        
        result[f'{category}_modified'] = modified
        result[f'{category}_added'] -= modified
        result[f'{category}_removed'] -= modified
    
    result['file_changes'] = dict(file_changes)
    return dict(result)

def ast_locator(path: str, name: str, kind: str) -> Optional[Tuple[int, int]]:
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
    symbol_map: Dict[str, List[dict]] = defaultdict(list)
    for chunk in chunks:
        if name := chunk.get('name'):
            symbol_map[name].append(chunk)
    return dict(symbol_map)


def prime_lsp(lsp: LspClient, paths: Iterable[str]) -> None:
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
    scores: Counter[int] = Counter()
    
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank)
    
    return [idx for idx, _ in scores.most_common()]


def estimate_tokens(text: str) -> int:
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

def build_analysis_prompt(diff_text: str, identifiers: Dict[str, Set[str]], 
                         extracted_content: ExtractedDiffContent) -> str:
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
    
    summary_parts.append(f"Code blocks: {len(extracted_content.added_blocks)} added, "
                        f"{len(extracted_content.removed_blocks)} removed, "
                        f"{len(extracted_content.modified_after)} modified")
    
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
    try:
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
        return AnalysisResult(
            risk_level='MEDIUM',
            focus_areas=['General code quality', 'Error handling'],
            specific_symbols=[],
            search_queries=['error handling patterns', 'similar implementations'],
            key_concerns=['Unable to parse initial analysis'],
            file_patterns=[]
        )

def targeted_search(
    analysis: AnalysisResult,
    identifiers: Dict[str, Set[str]],
    chunks: List[dict],
    bm25_index,
    vector_index,
    encoder: SentenceTransformer,
    symbol_map: Dict[str, List[dict]],
    extracted_content: ExtractedDiffContent
) -> List[int]:
    all_results = []
    search_rankings = []
    
    symbol_results = []
    for symbol in analysis.specific_symbols:
        if symbol in symbol_map:
            for chunk in symbol_map[symbol]:
                if chunk in chunks:
                    idx = chunks.index(chunk)
                    symbol_results.append(idx)
                    all_results.append(idx)
    if symbol_results:
        search_rankings.append(symbol_results)
    
    if extracted_content.added_blocks or extracted_content.modified_after:
        print(f"  Searching for similar code to {len(extracted_content.added_blocks)} added blocks...")
        added_results = []
        
        for block in (extracted_content.added_blocks + extracted_content.modified_after)[:5]:
            block_text = create_code_block_embedding_text(block)
            print(f"  Encoding block: {len(block_text)} chars")
            sys.stdout.flush()
            block_embedding = encoder.encode([block_text])
            
            distances, indices = vector_index.search(block_embedding, 10)
            added_results.extend(indices[0].tolist())
            all_results.extend(indices[0].tolist())
        
        if added_results:
            search_rankings.append(added_results)
    
    if extracted_content.removed_blocks or extracted_content.modified_before:
        print(f"  Searching for code similar to {len(extracted_content.removed_blocks)} removed blocks...")
        removed_results = []
        
        for block in (extracted_content.removed_blocks + extracted_content.modified_before)[:3]:
            block_text = create_code_block_embedding_text(block)
            print(f"  Encoding block: {len(block_text)} chars")
            sys.stdout.flush()
            block_embedding = encoder.encode([block_text])
            
            distances, indices = vector_index.search(block_embedding, 5)
            removed_results.extend(indices[0].tolist())
            all_results.extend(indices[0].tolist())
        
        if removed_results:
            search_rankings.append(removed_results)
    
    for query in analysis.search_queries:
        query_results = []
        
        docs = bm25_index.get_top_n(
            query.split(), 
            chunks, 
            n=10
        )
        for doc in docs:
            idx = chunks.index(doc)
            query_results.append(idx)
            all_results.append(idx)
        
        query_embedding = encoder.encode([query])
        distances, indices = vector_index.search(query_embedding, 10)
        query_results.extend(indices[0].tolist())
        all_results.extend(indices[0].tolist())
        
        if query_results:
            search_rankings.append(query_results)
    
    pattern_results = []
    for pattern in analysis.file_patterns:
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        for i, chunk in enumerate(chunks):
            if re.match(pattern_re, chunk['file_path']):
                pattern_results.append(i)
                all_results.append(i)
    if pattern_results:
        search_rankings.append(pattern_results)
    
    for area in analysis.focus_areas:
        area_results = []
        
        docs = bm25_index.get_top_n(
            area.lower().split(), 
            chunks, 
            n=5
        )
        for doc in docs:
            idx = chunks.index(doc)
            area_results.append(idx)
            all_results.append(idx)
        
        if area_results:
            search_rankings.append(area_results)
    
    if len(search_rankings) > 1:
        print(f"  Combining {len(search_rankings)} search strategies with RRF...")
        ranked_indices = reciprocal_rank_fusion(search_rankings)
        return ranked_indices[:MAX_CONTEXT_SNIPPETS]
    else:
        result_counts = Counter(all_results)
        ranked_indices = [idx for idx, _ in result_counts.most_common(MAX_CONTEXT_SNIPPETS)]
        return ranked_indices


def build_detailed_review_prompt(
    diff_text: str,
    context_sections: List[str],
    lsp_context: str,
    analysis: AnalysisResult,
    extracted_content: ExtractedDiffContent
) -> str:
    
    context_blob = "\n---\n".join(context_sections) if context_sections else "No context found."
    
    concerns_text = "\n".join(f"- {concern}" for concern in analysis.key_concerns)
    focus_text = "\n".join(f"- {area}" for area in analysis.focus_areas)
    
    change_summary = []
    if extracted_content.added_blocks:
        change_summary.append(f"- Adding {len(extracted_content.added_blocks)} new code blocks")
    if extracted_content.removed_blocks:
        change_summary.append(f"- Removing {len(extracted_content.removed_blocks)} code blocks")
    if extracted_content.modified_after:
        change_summary.append(f"- Modifying {len(extracted_content.modified_after)} existing code blocks")
    
    change_summary_text = "\n".join(change_summary) if change_summary else "- Minor code changes"
    
    return f"""You are **CodeReview-GPT**, an expert senior software engineer providing a detailed code review.
Based on initial analysis, this PR has been classified as **{analysis.risk_level} RISK**.

## Key Concerns Identified:
{concerns_text}

## Areas Requiring Deep Review:
{focus_text}

## Change Summary:
{change_summary_text}

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

def main():
    try:
        total_start = time.time()
        
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in .env file")
            return
        
        genai.configure(api_key=api_key)
        
        print("=== Enhanced Multistep Context-Rich PR Reviewer ===")
        print("Loading indices...")
        
        model_file = Path(MODEL_INFO_FILE)
        if model_file.exists():
            saved_model = model_file.read_text().strip()
            print(f"Found saved embedding model info: {saved_model}")
        
        try:
            vector_index = faiss.read_index(VECTOR_INDEX_FILE)
            with open(CHUNKS_FILE, 'rb') as f:
                chunks = pickle.load(f)
            with open(BM25_INDEX_FILE, 'rb') as f:
                bm25_index = pickle.load(f)
            print("Loaded enhanced indices")
        except:
            print(f"Error: Failed to load indices: {e}")
            print(f"Please ensure these files exist:")
            print(f"  - {VECTOR_INDEX_FILE}")
            print(f"  - {CHUNKS_FILE}")
            print(f"  - {BM25_INDEX_FILE}")
            print("Run the indexer first to create these files.")
            return
        
        print(f"\nDEBUG INFO:")
        print(f"  - Number of chunks loaded: {len(chunks)}")
        print(f"  - Vector index size: {vector_index.ntotal}")
        print(f"  - First chunk keys: {list(chunks[0].keys()) if chunks else 'No chunks'}")
        print(f"  - Embedding model to load: {EMBEDDING_MODEL}")
        
        sys.stdout.flush()
        
        print("\nCreating SentenceTransformer encoder...")
        sys.stdout.flush()
        
        try:
            encoder = SentenceTransformer(EMBEDDING_MODEL)
            print("✓ SentenceTransformer encoder created successfully")
                
        except Exception as e:
            print(f"Failed to create encoder: {e}")
            traceback.print_exc()
            return
        
        print("\nBuilding symbol map...")
        sys.stdout.flush()
        
        try:
            symbol_map_start = time.time()
            symbol_map = build_symbol_map(chunks)
            symbol_map_time = time.time() - symbol_map_start
            print(f"✓ Symbol map built with {len(symbol_map)} symbols in {symbol_map_time:.2f}s")
        except Exception as e:
            print(f"ERROR building symbol map: {e}")
            traceback.print_exc()
            return
        
        load_time = time.time() - total_start
        print(f"\nTotal loading completed in {load_time:.2f}s")
        print(f"Loaded {len(chunks)} code chunks")
        
    except Exception as e:
        print(f"Error loading indices: {e}")
        traceback.print_exc()
        return
    
    print("\nLoading diff file...")
    try:
        diff_path = Path('diff.txt')
        if not diff_path.exists():
            print(f"Error: diff.txt not found in current directory: {Path.cwd()}")
            return
        diff_text = diff_path.read_text()
        print(f"Loaded diff ({len(diff_text)} chars)")
    except Exception as e:
        print(f"Error loading diff.txt: {e}")
        traceback.print_exc()
        return
    
    print("\nExtracting identifiers from diff...")
    extract_start = time.time()
    identifiers = extract_identifiers_from_diff(diff_text)
    
    print("Extracting code blocks from diff...")
    extracted_content = extract_code_blocks_from_diff(diff_text)
    
    extract_time = time.time() - extract_start
    
    print(f"Found (in {extract_time:.2f}s):")
    print(f"  - Functions: {len(identifiers.get('functions_added', set()))} added, "
          f"{len(identifiers.get('functions_modified', set()))} modified")
    print(f"  - Classes: {len(identifiers.get('classes_added', set()))} added, "
          f"{len(identifiers.get('classes_modified', set()))} modified")
    print(f"  - Files: {len(identifiers.get('modified_files', set()))} modified")
    print(f"  - Code blocks: {len(extracted_content.added_blocks)} added, "
          f"{len(extracted_content.removed_blocks)} removed, "
          f"{len(extracted_content.modified_after)} modified")
    
    print("\n" + "="*60)
    print("STEP 1: Initial PR Analysis")
    print("="*60)
    
    analysis_prompt = build_analysis_prompt(diff_text, identifiers, extracted_content)
    
    print(f"Calling {LLM_MODEL} for initial analysis...")
    llm1_start = time.time()
    model = genai.GenerativeModel(LLM_MODEL)
    try:
        response = model.generate_content(
            analysis_prompt,
            generation_config=genai.GenerationConfig(
                temperature=LLM_TEMPERATURE,
            )
        )
        llm1_time = time.time() - llm1_start
        
        print(f"Analysis completed in {llm1_time:.2f}s")
        
        analysis = parse_analysis_result(response.text)
        
        print(f"\nAnalysis Results:")
        print(f"  Risk Level: {analysis.risk_level}")
        print(f"  Key Concerns: {len(analysis.key_concerns)}")
        print(f"  Focus Areas: {', '.join(analysis.focus_areas[:3])}")
        print(f"  Specific Symbols to Check: {', '.join(analysis.specific_symbols[:5])}")
        
    except Exception as e:
        print(f"Error in initial analysis: {e}")
        traceback.print_exc()
        analysis = AnalysisResult(
            risk_level='MEDIUM',
            focus_areas=['General code quality'],
            specific_symbols=[],
            search_queries=['related implementations'],
            key_concerns=['Failed to analyze, doing general review'],
            file_patterns=[]
        )
    
    print("\n" + "="*60)
    print("STEP 2: Enhanced Targeted Context Gathering")
    print("="*60)
    
    print("Performing targeted search based on analysis and extracted code...")
    search_start = time.time()
    
    top_indices = targeted_search(
        analysis,
        identifiers,
        chunks,
        bm25_index,
        vector_index,
        encoder,
        symbol_map,
        extracted_content
    )
    
    search_time = time.time() - search_start
    print(f"Retrieved {len(top_indices)} targeted code snippets in {search_time:.2f}s")
    
    print("\nGathering code references via LSP...")
    lsp_start = time.time()
    repo_root = Path.cwd().resolve()
    lsp_context = "LSP analysis unavailable."
    
    lsp = LspClient(repo_root)
    try:
        lsp.start()
        lsp.initialize()
        
        relevant_files = {chunks[i]['file_path'] for i in top_indices}
        relevant_files.update(identifiers.get('modified_files', set()))
        
        for symbol in analysis.specific_symbols:
            for chunk in symbol_map.get(symbol, []):
                relevant_files.add(chunk['file_path'])
        
        new_files = identifiers.get('files_added', set())
        relevant_files -= new_files
        
        prime_start = time.time()
        print(f"Priming LSP with {len(relevant_files)} files...")
        prime_lsp(lsp, relevant_files)
        prime_time = time.time() - prime_start
        
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
    
    context_snippets = []
    
    for i in top_indices:
        chunk = chunks[i]
        header = f"## {chunk['file_path']} - {chunk.get('type', 'code')} {chunk.get('name', '')}"
        context_snippets.append(f"{header}\n{chunk['code']}")
    
    context_trimmed = fit_to_token_budget(context_snippets, DEFAULT_TOKEN_BUDGET)
    
    print(f"\nContext prepared: {len(context_trimmed)} snippets")
    
    print("\n" + "="*60)
    print("STEP 3: Detailed Code Review")
    print("="*60)
    
    review_prompt = build_detailed_review_prompt(
        diff_text,
        context_trimmed,
        lsp_context,
        analysis,
        extracted_content
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
    
    total_time = time.time() - total_start
    print(f"\n=== Performance Summary (Enhanced Multistep) ===")
    print(f"Index loading: {load_time:.2f}s")
    print(f"Diff parsing & code extraction: {extract_time:.2f}s")
    print(f"Initial analysis (LLM 1): {llm1_time:.2f}s" if 'llm1_time' in locals() else "Initial analysis: N/A")
    print(f"Targeted search: {search_time:.2f}s" if 'search_time' in locals() else "Targeted search: N/A")
    print(f"LSP operations: {lsp_time:.2f}s" if 'lsp_time' in locals() else "LSP operations: N/A")
    print(f"Detailed review (LLM 2): {llm2_time:.2f}s" if 'llm2_time' in locals() else "Detailed review: N/A")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f}s")
    
    print(f"\n=== Enhanced Multistep Benefits ===")
    print(f"Risk-based analysis: {analysis.risk_level}")
    print(f"Targeted {len(analysis.specific_symbols)} specific symbols")
    print(f"Custom searches: {len(analysis.search_queries)} queries")
    print(f"Focus areas: {len(analysis.focus_areas)} identified")
    print(f"Code blocks analyzed: {len(extracted_content.all_blocks())}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

if __name__ == "__main__":
    main()