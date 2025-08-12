# OSIRIS - A Context-Rich PR Reviewer

from __future__ import annotations

import sys
import json
import os
import pickle
import re
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List,  Set

import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from unidiff import PatchSet

from parser import (
    ExtractedDiffContent,
    extract_code_blocks_from_diff,
    create_code_block_embedding_text,
    extract_identifiers_from_diff
)
from groq import Groq
from utils.ast_locator import ast_locator
from lsp_spike import LspClient

load_dotenv()

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
REPO_PATH = Path(os.getenv('REPO_PATH')).resolve()

ANALYSIS_SYSTEM_PROMPT = """You are OSIRIS, a static code analysis AI for pull requests.

<role>
Perform first-pass analysis to identify what needs human review. Extract measurable facts, risk patterns, and generate targeted search queries.
</role>

<planning_phase>
Before analysis:
1. Scan diff to determine file count and total line changes
2. Identify file types and change categories
3. Map modified symbols and their relationships
4. Determine appropriate analysis depth based on complexity
</planning_phase>

<complexity_classification>
Classify using these exact thresholds:
- **HIGH**: >10 files OR >500 lines OR core abstraction changes OR schema/API changes
- **MEDIUM**: 3-10 files OR 100-500 lines OR new features OR significant refactors
- **LOW**: <3 files AND <100 lines AND (bug fixes only OR docs/tests only)

Note: Use OR for HIGH/MEDIUM, AND for LOW to avoid ambiguity
</complexity_classification>

<analysis_rules>
1. **OBSERVABLE**: State only what you can directly see in the code
2. **PATTERN**: Identify known risk patterns from your training
3. **QUESTION**: Flag items needing domain knowledge (mark with ?)
4. **SEARCH_LIMITS**: Generate 2-5 search queries (scale with complexity)
5. **SYMBOL_LIMITS**: Focus on top 5-10 most important symbols
6. **ALWAYS**: Include at least 1 but maximum 5 reviewer questions
</analysis_rules>

<edge_case_handling>
- Empty diff → Return minimal structure with complexity_level: "LOW"
- Binary files → Note in focus_areas but skip symbol analysis
- Merge commits → Analyze only non-merge changes
- Generated files → Flag but don't analyze deeply
</edge_case_handling>

<output_schema>
{
  "complexity_level": "HIGH|MEDIUM|LOW",
  "complexity_rationale": "One sentence with specific counts (X files, Y lines)",
  "focus_areas": [
    {
      "area": "Specific code area or pattern",
      "reason": "Why it needs attention",
      "type": "OBSERVABLE|PATTERN|QUESTION",
      "priority": 1-3  // 1=highest
    }
  ],
  "specific_symbols": [
    {
      "symbol": "exact_function_or_class_name",
      "concern": "Specific concern",
      "files": ["exact/file/path.py"],
      "risk": "LOW|MEDIUM|HIGH"
    }
  ],
  "reviewer_questions": [
    "Question requiring domain knowledge (1-5 questions)"
  ],
  "search_queries": [
    {
      "query": "3-6 word search terms",
      "purpose": "What this reveals about the codebase"
    }
  ],
  "positive_observations": [
    "Good pattern observed (include 0-3 items)"
  ],
  "key_concerns": [
    "Primary issue identified (1-3 items)"
  ],
  "file_patterns": [
    "Pattern and why it matters (e.g., '*/test/* - 80% of changes are tests')"
  ]
}
</output_schema>

<quality_check>
Before outputting, verify:
- Complexity classification matches the criteria exactly
- Number of items in each array is within specified limits
- All concerns have clear, actionable reasoning
- Questions are truly unanswerable without domain knowledge
</quality_check>

Output ONLY valid JSON. No text outside the JSON structure."""


REVIEW_SYSTEM_PROMPT = """You are OSIRIS, a structured code review AI.

<mission>
Generate SPECIFIC, actionable feedback tied to exact code locations. Focus on issues that matter for code safety, correctness, and maintainability.
</mission>

<review_planning>
1. First pass: Identify all potential issues
2. Filter: Keep only issues worth commenting on
3. Prioritize: Order by severity and impact
4. Limit: Maximum 15 comments per review
5. Balance: Include 1-3 positive observations if exceptional patterns exist
</review_planning>

<merge_decision_criteria>
can_merge = true when ALL of:
- Zero critical issues
- Zero unhandled security concerns
- All major issues have fixes provided
- Code achieves stated purpose

can_merge = false when ANY of:
- Critical security/data loss risks exist
- Fundamental logic errors present
- Breaking changes without migration path
</merge_decision_criteria>

<comment_limits>
- Total comments: 1-15 (scale with PR size)
- Critical issues: No limit (all must be reported)
- Major issues: Maximum 7
- Minor/Info: Maximum 5 combined
- Positive observations: Maximum 3
</comment_limits>

<line_number_rules>
ALWAYS use line numbers from:
- NEW file (right side) for ADDITIONS and MODIFICATIONS
- OLD file (left side) for DELETIONS only
- Range if issue spans multiple lines (line_start, line_end)
</line_number_rules>

<output_schema>
{
  "summary": {
    "description": "2-3 sentences: what changed and why",
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "main_concerns": ["max 3 primary issues"],
    "can_merge": true/false,
    "merge_blocker_reason": "Required if can_merge=false"
  },
  "comments": [
    {
      "file": "exact/path/to/file.ext",
      "line_start": <integer>,
      "line_end": <integer>,
      "type": "OBSERVABLE|PATTERN|QUESTION|SUGGESTION",
      "severity": "critical|major|minor|info",
      "title": "Brief descriptive title (<50 chars)",
      "message": "Detailed explanation with specific impact",
      "snippet": "```language\\n3-15 lines of code\\n```",
      "fix": {
        "has_fix": true/false,
        "suggestion": "```diff\\n- old\\n+ new\\n```",
        "confidence": "HIGH|MEDIUM|LOW"
      }
    }
  ],
  "positive_observations": [
    {
      "file": "path/to/file.ext",
      "lines": "45-50",
      "pattern": "Specific good practice name",
      "message": "Why this is exemplary"
    }
  ],
  "metrics": {
    "files_reviewed": <integer>,
    "total_issues": <integer>,
    "critical_issues": <integer>,
    "comments_generated": <integer>
  }
}
</output_schema>

<severity_decision_tree>
Is it a security vulnerability or data loss risk? → critical
Does it break existing functionality? → major
Does it violate established patterns? → minor
Is it a suggestion or FYI? → info
</severity_decision_tree>

<comment_quality_rules>
1. Each comment must be actionable
2. Include specific line numbers, not vague locations
3. Explain impact, not just the problem
4. Provide fixes for all OBSERVABLE and most PATTERN issues
5. Mark Questions clearly with uncertainty indicators
</comment_quality_rules>

<skip_conditions>
DO NOT comment on:
- Working code following reasonable practices
- Style preferences (unless violating explicit standards)
- Successful improvements and refactoring
- Test additions (unless they have bugs)
- Comments/documentation (unless misleading)
</skip_conditions>

<escape_hatch>
If PR is too large (>50 files or >2000 lines):
- Focus on critical/major issues only
- Add summary note about partial review
- Suggest breaking into smaller PRs
</escape_hatch>

Output ONLY the JSON object. No markdown, no explanations outside JSON."""


@dataclass
class AnalysisResult:
    risk_level: str
    focus_areas: List[str]
    specific_symbols: List[str]
    search_queries: List[str]
    key_concerns: List[str]
    file_patterns: List[str]


def build_symbol_map(chunks: List[dict]) -> Dict[str, List[dict]]:
    symbol_map: Dict[str, List[dict]] = defaultdict(list)
    for chunk in chunks:
        if name := chunk.get('name'):
            symbol_map[name].append(chunk)
    return dict(symbol_map)


def prime_lsp(lsp: LspClient, paths: Iterable[str], repo_root: Path) -> None:
    """Prime LSP with files, handling both absolute and relative paths."""
    seen_paths = set()

    for path in paths:
        if Path(path).is_absolute():
            abs_path = Path(path)
        else:
            abs_path = (repo_root / path).resolve()

        if abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)

        print(abs_path)

        if not abs_path.exists():
            print(f"Skipping non-existent file for LSP: {abs_path}")
            continue

        try:
            lsp.open_file(str(abs_path))
        except Exception as e:
            print(f"Warning: Failed to prime LSP for {abs_path}: {e}")


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

def build_analysis_prompt(diff_text: str, identifiers: Dict, extracted_content: ExtractedDiffContent) -> str:
    return f"""## Change Metrics
                Analyze this pull request and return the results in JSON format:
                Files Modified: {len(identifiers.get('modified_files', set()))}
                Functions Added: {len(identifiers.get('functions_added', set()))}
                Functions Modified: {len(identifiers.get('functions_modified', set()))}
                Functions Removed: {len(identifiers.get('functions_removed', set()))}
                Classes Added: {len(identifiers.get('classes_added', set()))}
                Classes Modified: {len(identifiers.get('classes_modified', set()))}
                Code Block Delta: +{len(extracted_content.added_blocks)} -{len(extracted_content.removed_blocks)} ~{len(extracted_content.modified_after)}

                ## Diff Content
                ```diff
                {diff_text}
                ```

                Please analyze this code change and provide your assessment in the following JSON format:

                ```json
                {{
                    "complexity_level": "HIGH|MEDIUM|LOW",
                    "complexity_rationale": "Brief explanation of complexity assessment",

                    "focus_areas": [
                        {{
                            "area": "Description of code area to examine",
                            "reason": "Why this needs attention",
                            "type": "OBSERVABLE|PATTERN|QUESTION"
                        }}
                    ],

                    "specific_symbols": [
                        {{
                            "symbol": "function_or_class_name",
                            "concern": "What about this symbol needs review",
                            "files": ["file1.py", "file2.py"]
                        }}
                    ],

                    "reviewer_questions": [
                        "Specific question a reviewer with full context should answer"
                    ],

                    "search_queries": [
                        {{
                            "query": "search terms for similar code",
                            "purpose": "Why this search would help review"
                        }}
                    ],

                    "positive_observations": [
                        "Good patterns you noticed (if any)"
                    ],

                    "key_concerns": [
                        "Primary concerns identified in the changes"
                    ],

                    "file_patterns": [
                        "Patterns in file paths that might be relevant (e.g., */test/*, */api/*)"
                    ]
                }}
                ```"""


def build_review_prompt(diff_text: str, context: List[str], lsp_context: str,
                        analysis: AnalysisResult, extracted_content: ExtractedDiffContent) -> str:
    context_blob = "\n---\n".join(context) if context else "No relevant context found."


    return f"""## Analysis Results from Step 1
                Complexity Level: {analysis.risk_level}
                Rationale: {getattr(analysis, 'complexity_rationale', 'N/A')}

                Key Concerns:
                {chr(10).join(f"- {concern}" for concern in analysis.key_concerns) if analysis.key_concerns else "- None identified"}

                Focus Areas:
                {chr(10).join(f"- {area}" for area in analysis.focus_areas) if analysis.focus_areas else "- General review"}

                ## Repository Context
                ```
                {context_blob}
                ```

                ## Cross-Reference Data (LSP)
                ```
                {lsp_context}
                ```

                ## Complete Diff
                ```diff
                {diff_text}
                ```

                Please provide a detailed code review based on this information, focusing on the identified concerns and areas."""

def parse_analysis_result(response_text: str) -> AnalysisResult:
    try:
        print(response_text)
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


    return f"""## PR Analysis Results
                Complexity Level: **{analysis.risk_level}**
                Key Concerns:
                {concerns_text}

                Focus Areas:
                {focus_text}

                Change Summary:
                {change_summary_text}

                ## Repository Context
                {context_blob}

                ## Cross-Reference Analysis (LSP)
                {lsp_context}

                ## Complete Diff to Review
                ```diff
                {diff_text}
                ```

                Please review this code change, paying special attention to the identified concerns and focus areas."""

def main():
    try:
        total_start = time.time()

        load_dotenv()
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in .env file")
            return

        groq_client = Groq(api_key=api_key)

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
    try:
        response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": analysis_prompt}
                ],
                top_p=1,
                reasoning_effort="high",
                response_format={"type": "json_object"},
                temperature=1,
            )
        llm1_time = time.time() - llm1_start

        print(f"Analysis completed in {llm1_time:.2f}s")

        print(response.choices[0].message.content)
        analysis = parse_analysis_result(response.choices[0].message.content)

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
        prime_lsp(lsp, relevant_files,REPO_PATH)
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
        response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": review_prompt}
                ],
                reasoning_effort="high",
                temperature=LLM_TEMPERATURE,

        )
        llm2_time = time.time() - llm2_start

        print(f"Detailed review completed in {llm2_time:.2f}s")
        print("\n" + "="*60)
        print("FINAL REVIEW RESULTS")
        print("="*60)
        print(response.choices[0].message.content)
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
