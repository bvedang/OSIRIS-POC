import os
import faiss
import pickle
import ast
import re
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
from dotenv import load_dotenv
from lsp_spike import LspClient
import numpy as np
from datetime import datetime
import json
from unidiff import PatchSet

# Load environment variables
load_dotenv()

# ==================== Data Classes ====================

@dataclass
class CodeChunk:
    file_path: str
    name: str
    code: str
    type: str  # 'function', 'class', 'method', 'module'
    start_line: int
    end_line: int
    imports: List[str]
    calls: List[str]
    metadata: Dict[str, Any]

@dataclass
class ChangeInfo:
    new_functions: List[Dict[str, Any]]
    modified_functions: List[Dict[str, Any]]
    deleted_functions: List[Dict[str, Any]]
    new_classes: List[Dict[str, Any]]
    modified_logic: List[Dict[str, Any]]
    new_imports: List[str]
    config_changes: List[Dict[str, Any]]
    affected_files: Set[str]

@dataclass
class ImpactAnalysis:
    directly_affected: List[str]
    indirectly_affected: List[str]
    test_coverage: List[str]
    api_changes: List[Dict[str, Any]]
    breaking_changes: List[Dict[str, Any]]
    performance_risks: List[str]
    security_risks: List[str]

@dataclass
class ReviewResult:
    blocking: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    nits: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    auto_fixes: List[Dict[str, Any]]
    raw_review: str

# ==================== AST Analysis Utilities ====================

class ASTAnalyzer:
    """Advanced AST analysis for code understanding"""
    
    @staticmethod
    def extract_function_calls(code: str) -> List[str]:
        """Extract all function calls from code"""
        calls = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.append(node.func.attr)
        except:
            pass
        return calls
    
    @staticmethod
    def extract_imports(code: str) -> List[str]:
        """Extract all imports from code"""
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
        except:
            pass
        return imports
    
    @staticmethod
    def extract_class_methods(code: str) -> Dict[str, List[str]]:
        """Extract classes and their methods"""
        classes = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    classes[node.name] = methods
        except:
            pass
        return classes
    
    @staticmethod
    def calculate_complexity(code: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
        except:
            pass
        return complexity

# ==================== Diff Parser ====================

from unidiff import PatchSet

class DiffParser:
    """Parse and analyze code diffs using unidiff"""
    
    @staticmethod
    def parse_diff(diff: str) -> List[Dict[str, Any]]:
        """Parse diff into structured format using unidiff"""
        try:
            patch = PatchSet(diff)
            changes = []
            
            for patched_file in patch:
                for hunk in patched_file:
                    change = {
                        'file': patched_file.target_file or patched_file.source_file,
                        'old_start': hunk.source_start,
                        'old_lines': hunk.source_length,
                        'new_start': hunk.target_start,
                        'new_lines': hunk.target_length,
                        'added': [],
                        'removed': [],
                        'context': []
                    }
                    
                    for line in hunk:
                        if line.line_type == '+':
                            change['added'].append(line.value)
                        elif line.line_type == '-':
                            change['removed'].append(line.value)
                        else:  # ' ' for context
                            change['context'].append(line.value)
                    
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            print(f"Error parsing diff: {e}")
            # Fallback to simple parsing if unidiff fails
            return DiffParser._simple_parse_fallback(diff)
    
    @staticmethod
    def _simple_parse_fallback(diff: str) -> List[Dict[str, Any]]:
        """Simple fallback parser for non-standard diffs"""
        # This handles cases where the diff might not be in proper format
        lines = diff.split('\n')
        changes = [{
            'file': 'unknown',
            'old_start': 0,
            'old_lines': 0,
            'new_start': 0,
            'new_lines': len(lines),
            'added': [line for line in lines if not line.startswith('#')],
            'removed': [],
            'context': []
        }]
        return changes
    
    @staticmethod
    def extract_changed_functions(diff: str) -> List[Dict[str, Any]]:
        """Extract functions that were changed in the diff"""
        changed_functions = []
        
        try:
            patch = PatchSet(diff)
            
            for patched_file in patch:
                file_path = patched_file.target_file or patched_file.source_file
                
                # Process added lines
                for hunk in patched_file:
                    for line in hunk:
                        if line.line_type == '+' and 'def ' in line.value:
                            match = re.search(r'def\s+(\w+)\s*\(', line.value)
                            if match:
                                changed_functions.append({
                                    'name': match.group(1),
                                    'file': file_path,
                                    'type': 'added',
                                    'line': line.target_line_no or 0
                                })
                        elif line.line_type == '-' and 'def ' in line.value:
                            match = re.search(r'def\s+(\w+)\s*\(', line.value)
                            if match:
                                # Check if this function exists in added lines (modified) or not (deleted)
                                func_name = match.group(1)
                                is_modified = any(
                                    func_name in l.value 
                                    for h in patched_file 
                                    for l in h 
                                    if l.line_type == '+' and 'def ' in l.value
                                )
                                
                                changed_functions.append({
                                    'name': func_name,
                                    'file': file_path,
                                    'type': 'modified' if is_modified else 'removed',
                                    'line': line.source_line_no or 0
                                })
        
        except Exception as e:
            print(f"Error extracting functions from diff: {e}")
            # Fallback to regex-based extraction
            return DiffParser._extract_functions_fallback(diff)
        
        return changed_functions
    
    @staticmethod
    def _extract_functions_fallback(diff: str) -> List[Dict[str, Any]]:
        """Fallback function extraction for non-standard diffs"""
        changed_functions = []
        
        # Simple regex-based extraction
        for match in re.finditer(r'def\s+(\w+)\s*\(', diff):
            changed_functions.append({
                'name': match.group(1),
                'file': 'unknown',
                'type': 'added',
                'line': 0
            })
        
        return changed_functions

# ==================== Enhanced Search System ====================

class EnhancedSearchSystem:
    """Multi-modal search with re-ranking"""
    
    def __init__(self, vector_index, all_chunks, bm25_index, model_encoder):
        self.vector_index = vector_index
        self.all_chunks = all_chunks
        self.bm25_index = bm25_index
        self.model_encoder = model_encoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.ast_analyzer = ASTAnalyzer()
        
    def vector_search(self, query: str, k: int = 10) -> List[CodeChunk]:
        """Semantic vector search"""
        query_embedding = self.model_encoder.encode([query])
        distances, indices = self.vector_index.search(query_embedding, k)
        return [self.all_chunks[i] for i in indices[0]]
    
    def bm25_search(self, query: str, k: int = 10) -> List[CodeChunk]:
        """Keyword-based BM25 search"""
        tokenized_query = query.split()
        return self.bm25_index.get_top_n(tokenized_query, self.all_chunks, n=k)
    
    def ast_search(self, query: str, k: int = 10) -> List[CodeChunk]:
        """AST-based structural search"""
        # Extract function names and classes from query
        query_functions = set(re.findall(r'\b(\w+)\s*\(', query))
        query_classes = set(re.findall(r'class\s+(\w+)', query))
        
        results = []
        for chunk in self.all_chunks:
            score = 0
            chunk_functions = set(re.findall(r'def\s+(\w+)\s*\(', chunk['code']))
            chunk_classes = set(re.findall(r'class\s+(\w+)', chunk['code']))
            
            # Score based on matching function/class names
            score += len(query_functions.intersection(chunk_functions)) * 2
            score += len(query_classes.intersection(chunk_classes)) * 2
            
            if score > 0:
                results.append((score, chunk))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in results[:k]]
    
    def cross_encoder_rerank(self, query: str, candidates: List[CodeChunk], top_k: int = 10) -> List[CodeChunk]:
        """Re-rank candidates using cross-encoder"""
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, chunk['code'][:512]] for chunk in candidates]  # Truncate for efficiency
        
        # Get scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score
        scored_candidates = list(zip(scores, candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [chunk for _, chunk in scored_candidates[:top_k]]
    
    def enhanced_search(self, query: str, k: int = 10) -> List[CodeChunk]:
        """Multi-stage retrieval with re-ranking"""
        # Stage 1: Cast a wide net
        candidates = []
        candidates.extend(self.vector_search(query, k=k*3))
        candidates.extend(self.bm25_search(query, k=k*3))
        candidates.extend(self.ast_search(query, k=k*3))
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for chunk in candidates:
            key = (chunk['file_path'], chunk['name'])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(chunk)
        
        # Stage 2: Re-rank using cross-encoder
        reranked = self.cross_encoder_rerank(query, unique_candidates, top_k=k)
        
        # Stage 3: Expand with related code
        expanded = []
        for result in reranked[:5]:
            # Get test files for this code
            test_files = self.find_test_files(result['file_path'])
            expanded.extend(test_files)
            
            # Get parent class if this is a method
            if result.get('type') == 'method':
                parent = self.get_parent_class(result)
                if parent:
                    expanded.append(parent)
        
        return reranked + expanded
    
    def find_test_files(self, file_path: str) -> List[CodeChunk]:
        """Find test files related to a given file"""
        base_name = Path(file_path).stem
        test_patterns = [
            f"test_{base_name}",
            f"{base_name}_test",
            f"tests/{base_name}",
            f"test/{base_name}"
        ]
        
        test_chunks = []
        for chunk in self.all_chunks:
            chunk_path = chunk['file_path']
            for pattern in test_patterns:
                if pattern in chunk_path:
                    test_chunks.append(chunk)
                    break
        
        return test_chunks
    
    def get_parent_class(self, method_chunk: CodeChunk) -> Optional[CodeChunk]:
        """Get the parent class of a method"""
        # Look for class definition in the same file
        for chunk in self.all_chunks:
            if (chunk['file_path'] == method_chunk['file_path'] and 
                chunk.get('type') == 'class' and
                chunk['start_line'] < method_chunk['start_line'] < chunk['end_line']):
                return chunk
        return None

# ==================== Context Building ====================

class ContextBuilder:
    """Build comprehensive context for code review"""
    
    def __init__(self, search_system: EnhancedSearchSystem, lsp_client: LspClient):
        self.search_system = search_system
        self.lsp = lsp_client
        self.ast_analyzer = ASTAnalyzer()
    
    def get_multi_hop_context(self, initial_results: List[CodeChunk], depth: int = 2) -> List[CodeChunk]:
        """Follow imports and function calls to gather deeper context"""
        all_context = set()
        to_explore = list(initial_results)
        
        # Add initial results
        for item in initial_results:
            all_context.add((item['file_path'], item['name']))
        
        for _ in range(depth):
            new_items = []
            for item in to_explore:
                # Extract function calls and imports
                calls = self.ast_analyzer.extract_function_calls(item['code'])
                imports = self.ast_analyzer.extract_imports(item['code'])
                
                # Search for these in the codebase
                for query in calls + imports:
                    results = self.search_system.enhanced_search(query, k=3)
                    for result in results:
                        key = (result['file_path'], result['name'])
                        if key not in all_context:
                            all_context.add(key)
                            new_items.append(result)
            
            to_explore = new_items
        
        # Convert back to list
        final_results = []
        for chunk in initial_results:
            final_results.append(chunk)
        
        return final_results
    
    def build_dependency_context(self, changed_file: str) -> Dict[str, List[str]]:
        """Find all files that depend on or are dependencies of the changed file"""
        context = {
            'upstream': [],  # Files this file depends on
            'downstream': [],  # Files that depend on this file
            'tests': []  # Related test files
        }
        
        # Find imports in the changed file
        for chunk in self.search_system.all_chunks:
            if chunk['file_path'] == changed_file:
                imports = self.ast_analyzer.extract_imports(chunk['code'])
                context['upstream'].extend(imports)
        
        # Find files that import the changed file
        module_name = Path(changed_file).stem
        for chunk in self.search_system.all_chunks:
            imports = self.ast_analyzer.extract_imports(chunk['code'])
            if any(module_name in imp for imp in imports):
                context['downstream'].append(chunk['file_path'])
        
        # Find related tests
        context['tests'] = [
            chunk['file_path'] 
            for chunk in self.search_system.find_test_files(changed_file)
        ]
        
        return context
    
    def get_historical_context(self, file_path: str, function_name: str) -> Dict[str, Any]:
        """Find similar past changes and their reviews (simulated)"""
        # In a real implementation, this would query a database of past reviews
        return {
            'similar_changes': [],  # Would contain similar historical changes
            'past_bugs': [],  # Bugs found in this area
            'review_patterns': [],  # Common issues in this file
            'author_history': []  # Author's common mistakes
        }

# ==================== Enhanced LSP Integration ====================

class EnhancedLSPAnalyzer:
    """Extended LSP functionality"""
    
    def __init__(self, project_root: Path):
        # Ensure project_root is absolute before passing to LspClient
        self.project_root = project_root.resolve()
        self.lsp = LspClient(self.project_root)
    
    def start(self):
        """Start the LSP server"""
        self.lsp.start()
        self.lsp.initialize()
    
    def shutdown(self):
        """Shutdown the LSP server"""
        self.lsp.shutdown()
    
    def comprehensive_analysis(self, file_path: str, function_name: str) -> Dict[str, Any]:
        """Perform comprehensive LSP analysis"""
        # Ensure file_path is absolute
        if not Path(file_path).is_absolute():
            file_path = str((self.project_root / file_path).resolve())
        
        line, char = self.find_function_location(file_path, function_name)
        if line is None:
            return {}
        
        analysis = {
            'references': self.lsp.get_references(file_path, line, char),
            'diagnostics': self.get_diagnostics(file_path),
            'call_hierarchy': self.get_call_hierarchy(file_path, line, char),
            'type_info': self.get_type_info(file_path, line, char)
        }
        
        return analysis
    
    def find_function_location(self, file_path: str, function_name: str) -> Tuple[Optional[int], Optional[int]]:
        """Find the line and character of a function definition"""
        try:
            # Ensure file_path is absolute
            if not Path(file_path).is_absolute():
                file_path = str((self.project_root / file_path).resolve())
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if f"def {function_name}" in line:
                    return i, line.find(function_name)
        except FileNotFoundError:
            return None, None
        return None, None
    
    def get_diagnostics(self, file_path: str) -> List[Dict[str, Any]]:
        """Get diagnostics (errors, warnings) for a file"""
        # This would use textDocument/publishDiagnostics in a real implementation
        return []
    
    def get_call_hierarchy(self, file_path: str, line: int, char: int) -> Dict[str, Any]:
        """Get incoming and outgoing calls"""
        # This would use callHierarchy/incomingCalls and callHierarchy/outgoingCalls
        return {
            'incoming': [],
            'outgoing': []
        }
    
    def get_type_info(self, file_path: str, line: int, char: int) -> Dict[str, Any]:
        """Get type information at a position"""
        # This would use textDocument/hover or similar
        return {}

# ==================== Impact Analysis ====================

class ImpactAnalyzer:
    """Analyze the impact of code changes"""
    
    def __init__(self, search_system: EnhancedSearchSystem, lsp_analyzer: EnhancedLSPAnalyzer):
        self.search_system = search_system
        self.lsp_analyzer = lsp_analyzer
        self.ast_analyzer = ASTAnalyzer()
    
    def analyze_change_impact(self, pr_diff: str, context: Dict[str, Any]) -> ImpactAnalysis:
        """Determine the blast radius of changes"""
        impact = ImpactAnalysis(
            directly_affected=[],
            indirectly_affected=[],
            test_coverage=[],
            api_changes=[],
            breaking_changes=[],
            performance_risks=[],
            security_risks=[]
        )
        
        # Extract changed functions
        changed_functions = DiffParser.extract_changed_functions(pr_diff)
        
        # Find direct references
        for func in changed_functions:
            if func['type'] != 'removed':
                analysis = self.lsp_analyzer.comprehensive_analysis(
                    func['file'], 
                    func['name']
                )
                if analysis.get('references'):
                    for ref in analysis['references'].get('result', []):
                        ref_path = ref['uri'].replace('file://', '')
                        impact.directly_affected.append(ref_path)
        
        # Analyze for API changes
        impact.api_changes = self.detect_api_changes(pr_diff)
        
        # Check for breaking changes
        impact.breaking_changes = self.detect_breaking_changes(pr_diff, changed_functions)
        
        # Assess performance risks
        impact.performance_risks = self.assess_performance_risks(pr_diff)
        
        # Assess security risks
        impact.security_risks = self.assess_security_risks(pr_diff)
        
        return impact
    
    def detect_api_changes(self, pr_diff: str) -> List[Dict[str, Any]]:
        """Detect changes to public APIs"""
        api_changes = []
        
        # Look for changes to public functions/classes
        for match in re.finditer(r'[-+]\s*def\s+(\w+)\s*\((.*?)\):', pr_diff, re.MULTILINE):
            if not match.group(1).startswith('_'):  # Public method
                api_changes.append({
                    'type': 'function',
                    'name': match.group(1),
                    'signature': match.group(2),
                    'change_type': 'modified' if pr_diff.count(f"def {match.group(1)}") > 1 else 'added'
                })
        
        return api_changes
    
    def detect_breaking_changes(self, pr_diff: str, changed_functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect breaking changes"""
        breaking_changes = []
        
        # Check for removed functions
        for func in changed_functions:
            if func['type'] == 'removed':
                breaking_changes.append({
                    'type': 'function_removed',
                    'name': func['name'],
                    'file': func['file']
                })
        
        # Check for signature changes
        # This is simplified - real implementation would parse more carefully
        for match in re.finditer(r'-\s*def\s+(\w+)\s*\((.*?)\):', pr_diff):
            old_sig = match.group(2)
            func_name = match.group(1)
            
            # Look for the new signature
            new_match = re.search(rf'\+\s*def\s+{func_name}\s*\((.*?)\):', pr_diff)
            if new_match and new_match.group(1) != old_sig:
                breaking_changes.append({
                    'type': 'signature_change',
                    'function': func_name,
                    'old_signature': old_sig,
                    'new_signature': new_match.group(1)
                })
        
        return breaking_changes
    
    def assess_performance_risks(self, pr_diff: str) -> List[str]:
        """Identify potential performance issues"""
        risks = []
        
        # Check for N+1 query patterns
        if 'for' in pr_diff and any(db_op in pr_diff for db_op in ['query', 'fetch', 'get', 'select']):
            risks.append("Potential N+1 query pattern detected in loop")
        
        # Check for inefficient operations
        if 'in' in pr_diff and 'for' in pr_diff:
            # Nested loops
            if pr_diff.count('for') > 1:
                risks.append("Nested loops detected - potential O(n²) complexity")
        
        # Check for large data structure operations
        if any(op in pr_diff for op in ['sort', 'sorted']) and 'key=' not in pr_diff:
            risks.append("Sorting without key function - may be inefficient")
        
        return risks
    
    def assess_security_risks(self, pr_diff: str) -> List[str]:
        """Identify potential security issues"""
        risks = []
        
        # SQL injection risks
        if any(db in pr_diff.lower() for db in ['select', 'insert', 'update', 'delete']):
            if any(concat in pr_diff for concat in ['%', 'format', 'f"', "f'"]):
                risks.append("Potential SQL injection - use parameterized queries")
        
        # Dangerous functions
        dangerous_funcs = ['eval', 'exec', 'compile', '__import__']
        for func in dangerous_funcs:
            if func in pr_diff:
                risks.append(f"Use of potentially dangerous function: {func}")
        
        # Hardcoded secrets
        secret_patterns = [
            r'api_key\s*=\s*["\'][\w]+["\']',
            r'password\s*=\s*["\'][\w]+["\']',
            r'secret\s*=\s*["\'][\w]+["\']'
        ]
        for pattern in secret_patterns:
            if re.search(pattern, pr_diff, re.IGNORECASE):
                risks.append("Potential hardcoded secret detected")
        
        return risks

# ==================== Semantic Diff Analysis ====================

class SemanticDiffAnalyzer:
    """Analyze diffs semantically to understand changes"""
    
    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
    
    def analyze_diff_semantically(self, pr_diff: str) -> ChangeInfo:
        """Extract meaningful changes from the diff using unidiff"""
        changes = ChangeInfo(
            new_functions=[],
            modified_functions=[],
            deleted_functions=[],
            new_classes=[],
            modified_logic=[],
            new_imports=[],
            config_changes=[],
            affected_files=set()
        )
        
        try:
            patch = PatchSet(pr_diff)
            
            for patched_file in patch:
                file_path = patched_file.target_file or patched_file.source_file
                if file_path:
                    # Remove leading 'b/' or 'a/' from git diff paths
                    file_path = re.sub(r'^[ab]/', '', file_path)
                    changes.affected_files.add(file_path)
                
                # Collect all added and removed lines
                added_lines = []
                removed_lines = []
                
                for hunk in patched_file:
                    for line in hunk:
                        if line.line_type == '+':
                            added_lines.append(line.value)
                        elif line.line_type == '-':
                            removed_lines.append(line.value)
                
                # Analyze added code
                added_code = '\n'.join(added_lines)
                if added_code:
                    # New functions
                    for match in re.finditer(r'def\s+(\w+)\s*\((.*?)\):', added_code):
                        func_name = match.group(1)
                        signature = match.group(2)
                        
                        # Check if this is a modification (exists in removed)
                        if any(f'def {func_name}' in line for line in removed_lines):
                            changes.modified_functions.append({
                                'name': func_name,
                                'signature': signature,
                                'file': file_path
                            })
                        else:
                            changes.new_functions.append({
                                'name': func_name,
                                'signature': signature,
                                'file': file_path,
                                'line': 0  # Could track actual line if needed
                            })
                    
                    # New classes
                    for match in re.finditer(r'class\s+(\w+)(?:\((.*?)\))?:', added_code):
                        changes.new_classes.append({
                            'name': match.group(1),
                            'bases': match.group(2) or '',
                            'file': file_path,
                            'line': 0
                        })
                    
                    # New imports
                    try:
                        imports = self.ast_analyzer.extract_imports(added_code)
                        changes.new_imports.extend(imports)
                    except:
                        # Fallback to regex if AST parsing fails
                        import_matches = re.findall(r'(?:from\s+[\w.]+\s+)?import\s+[\w, ]+', added_code)
                        changes.new_imports.extend(import_matches)
                
                # Analyze removed code for deleted functions
                removed_code = '\n'.join(removed_lines)
                if removed_code:
                    for match in re.finditer(r'def\s+(\w+)\s*\((.*?)\):', removed_code):
                        func_name = match.group(1)
                        # Only mark as deleted if not in added code
                        if not any(f'def {func_name}' in line for line in added_lines):
                            changes.deleted_functions.append({
                                'name': func_name,
                                'signature': match.group(2),
                                'file': file_path
                            })
        
        except Exception as e:
            print(f"Error in semantic diff analysis: {e}")
            # Fallback to simple analysis
            changes.affected_files.add('unknown')
            # Try basic regex extraction
            for match in re.finditer(r'def\s+(\w+)\s*\((.*?)\):', pr_diff):
                changes.new_functions.append({
                    'name': match.group(1),
                    'signature': match.group(2),
                    'file': 'unknown',
                    'line': 0
                })
        
        return changes
    
    def detect_change_type(self, pr_diff: str) -> str:
        """Detect the type of change (new_feature, refactor, bugfix, etc.)"""
        changes = self.analyze_diff_semantically(pr_diff)
        
        # Heuristics for change type detection
        if changes.new_functions or changes.new_classes:
            if len(changes.deleted_functions) == 0:
                return "new_feature"
            else:
                return "refactor"
        elif changes.deleted_functions and not changes.new_functions:
            return "cleanup"
        elif "fix" in pr_diff.lower() or "bug" in pr_diff.lower():
            return "bugfix"
        elif changes.modified_functions:
            return "enhancement"
        else:
            return "other"

# ==================== Review Cache ====================

class ReviewCache:
    """Smart caching for expensive operations"""
    
    def __init__(self, cache_dir: Path = Path(".review_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.pattern_cache = {}
        self.context_cache = {}
        self.memory_cache = {}  # In-memory cache for current session
    
    def get_cache_key(self, content: str) -> str:
        """Generate a cache key for content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_or_compute_context(self, file_content: str, compute_fn) -> Any:
        """Cache expensive context computations"""
        cache_key = self.get_cache_key(file_content)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                self.memory_cache[cache_key] = result
                return result
        
        # Compute and cache
        result = compute_fn()
        self.memory_cache[cache_key] = result
        
        # Save to disk
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    
    def cache_pattern(self, pattern_type: str, pattern: str, result: Any):
        """Cache common patterns and their analysis"""
        if pattern_type not in self.pattern_cache:
            self.pattern_cache[pattern_type] = {}
        self.pattern_cache[pattern_type][pattern] = result
    
    def get_pattern(self, pattern_type: str, pattern: str) -> Optional[Any]:
        """Retrieve cached pattern analysis"""
        return self.pattern_cache.get(pattern_type, {}).get(pattern)

# ==================== Prompt Engineering ====================

class PromptBuilder:
    def build_prompt(self, pr_diff: str, context: str, additional_info: str = '') -> str:
        """Legacy method for backward compatibility"""
        return self.build_enhanced_prompt(
            pr_diff=pr_diff,
            full_context={'semantic_context': context},
            analysis_data={'lsp_analysis': additional_info},
            change_type='general'
        )
    
    def build_enhanced_prompt(self, pr_diff: str, full_context: dict, analysis_data: dict, change_type: str = 'general') -> str:
        """Enhanced prompt building with structured context and analysis data"""
        
        # Extract semantic context
        semantic_context = full_context.get('semantic_context', '')
        
        # Extract analysis info - handle both dict and object formats
        lsp_analysis = ''
        if isinstance(analysis_data, dict):
            if 'lsp_analysis' in analysis_data:
                lsp_analysis = analysis_data['lsp_analysis']
            elif 'changes' in analysis_data:
                # Handle the changes.__dict__ format
                changes_data = analysis_data['changes']
                lsp_analysis = self._format_changes_data(changes_data)
        
        return f"""
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
{semantic_context}
</existing_code>

2. Code-reference analysis (where functions/classes are used):
<lsp_analysis>
{lsp_analysis}
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

1. **Reference real lines**; don't invent code.  
2. **Only comment on observed issues**; no speculation.  
3. **Keep it actionable**—issue, impact, fix.  
4. **Be concise**; skip boilerplate praise.  
5. **Follow output format exactly**; otherwise reviewers waste time re-parsing.

---

*End of system prompt.*
"""

    def _format_changes_data(self, changes_data: dict) -> str:
        """Format changes data dictionary into readable analysis string"""
        if not changes_data:
            return "No changes data available"
        
        # Convert the changes dict to a formatted string
        formatted_parts = []
        for key, value in changes_data.items():
            if value:  # Only include non-empty values
                formatted_parts.append(f"{key}: {value}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No analysis data"


# ==================== Multi-Model Review System ====================

class MultiModelReviewer:
    """Use multiple models for different aspects of review"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.main_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.fast_model = genai.GenerativeModel('gemini-1.5-flash')
        # In reality, you might use different models for these
        self.security_model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    async def style_review(self, pr_diff: str) -> Dict[str, Any]:
        """Quick style and formatting review"""
        prompt = f"""
Review this code for style and formatting issues only:

{pr_diff}

Focus on:
- Naming conventions
- Code formatting
- Import organization
- Comment quality
- Function/class organization

Output format:
- Issue: [description]
  Location: [file:line]
  Fix: [suggested correction]
"""
        
        response = await self.fast_model.generate_content_async(prompt)
        return self.parse_style_review(response.text)
    
    async def security_review(self, pr_diff: str) -> Dict[str, Any]:
        """Specialized security review"""
        prompt = f"""
Perform a security audit of this code:

{pr_diff}

Check for:
- Injection vulnerabilities (SQL, command, etc.)
- Authentication/authorization issues
- Cryptographic weaknesses
- Input validation problems
- Information disclosure
- SSRF/XXE vulnerabilities
- Insecure deserialization

For each issue found:
- Severity: [CRITICAL/HIGH/MEDIUM/LOW]
- Issue: [description]
- Location: [where in code]
- Impact: [what could happen]
- Fix: [how to fix it]
"""
        
        response = await self.security_model.generate_content_async(prompt)
        return self.parse_security_review(response.text)
    
    async def logic_review(self, pr_diff: str, context: str, prompt: str) -> str:
        """Main logic and architecture review"""
        print(f"Running logic review with context: {context}")
        response = await self.main_model.generate_content_async(prompt)
        print(f"Logic review response: {response.text}")
        return response.text
    
    def parse_style_review(self, review_text: str) -> Dict[str, Any]:
        """Parse style review into structured format"""
        # Simple parsing - in practice would be more robust
        issues = []
        for line in review_text.split('\n'):
            if line.startswith('- Issue:'):
                issues.append({'type': 'style', 'description': line[8:].strip()})
        return {'style_issues': issues}
    
    def parse_security_review(self, review_text: str) -> Dict[str, Any]:
        """Parse security review into structured format"""
        issues = []
        current_issue = {}
        
        for line in review_text.split('\n'):
            if line.startswith('- Severity:'):
                if current_issue:
                    issues.append(current_issue)
                current_issue = {'severity': line[11:].strip()}
            elif line.startswith('- Issue:'):
                current_issue['description'] = line[8:].strip()
            elif line.startswith('- Impact:'):
                current_issue['impact'] = line[9:].strip()
            elif line.startswith('- Fix:'):
                current_issue['fix'] = line[6:].strip()
        
        if current_issue:
            issues.append(current_issue)
        
        return {'security_issues': issues}
    
    async def ensemble_review(self, pr_diff: str, context: str, full_prompt: str) -> Dict[str, Any]:
        """Combine multiple model reviews"""
        # Run reviews in parallel
        style_task = asyncio.create_task(self.style_review(pr_diff))
        security_task = asyncio.create_task(self.security_review(pr_diff))
        logic_task = asyncio.create_task(self.logic_review(pr_diff, context, full_prompt))
        
        # Wait for all reviews
        style_result = await style_task
        security_result = await security_task
        logic_result = await logic_task
        
        # Combine results
        return {
            'style': style_result,
            'security': security_result,
            'logic': logic_result,
            'combined': self.merge_reviews(style_result, security_result, logic_result)
        }
    
    def merge_reviews(self, style: Dict, security: Dict, logic: str) -> str:
        """Merge multiple reviews into a cohesive response"""
        # Parse the logic review
        parsed_logic = self.parse_logic_review(logic)
        
        # Combine all issues
        all_blocking = []
        all_consider = []
        all_nits = []
        
        # Add security issues as blocking
        for issue in security.get('security_issues', []):
            if issue['severity'] in ['CRITICAL', 'HIGH']:
                all_blocking.append(f"[SECURITY] {issue['description']}")
            else:
                all_consider.append(f"[SECURITY] {issue['description']}")
        
        # Add style issues as nits
        for issue in style.get('style_issues', []):
            all_nits.append(issue['description'])
        
        # Add logic review issues
        all_blocking.extend(parsed_logic.get('blocking', []))
        all_consider.extend(parsed_logic.get('consider', []))
        all_nits.extend(parsed_logic.get('nits', []))
        
        # Format the combined review
        combined = []
        
        if all_blocking:
            combined.append("**BLOCKING** 🚫")
            for issue in all_blocking:
                combined.append(f"- {issue}")
            combined.append("")
        
        if all_consider:
            combined.append("**CONSIDER** 💭")
            for issue in all_consider:
                combined.append(f"- {issue}")
            combined.append("")
        
        if all_nits:
            combined.append("**NITS** 💅")
            for issue in all_nits:
                combined.append(f"- {issue}")
            combined.append("")
        
        if not any([all_blocking, all_consider, all_nits]):
            combined.append("LGTM 🎉")
        
        return '\n'.join(combined)
    
    def parse_logic_review(self, review_text: str) -> Dict[str, List[str]]:
        """Parse the main logic review"""
        result = {
            'blocking': [],
            'consider': [],
            'nits': []
        }
        
        current_section = None
        for line in review_text.split('\n'):
            if 'BLOCKING' in line:
                current_section = 'blocking'
            elif 'CONSIDER' in line:
                current_section = 'consider'
            elif 'NITS' in line:
                current_section = 'nits'
            elif current_section and line.strip().startswith('-'):
                result[current_section].append(line.strip()[1:].strip())
        
        return result

# ==================== Main Review System ====================

class EnhancedCodeReviewSystem:
    """Main orchestrator for the enhanced code review system"""
    
    def __init__(self, api_key: str, project_root: Path = Path(".")):
        self.api_key = api_key
        self.project_root = project_root.resolve()  # Ensure absolute path
        self.cache = ReviewCache()
        self.setup_components()
    
    def setup_components(self):
        """Initialize all components"""
        print("Setting up enhanced code review system...")
        
        # Load indexes
        print("Loading indexes...")
        self.vector_index = faiss.read_index("repo.faiss")
        with open("repo_chunks.pkl", "rb") as f:
            self.all_chunks = pickle.load(f)
        with open("repo_bm25.pkl", "rb") as f:
            self.bm25_index = pickle.load(f)
        
        # Initialize models
        self.model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize components
        self.search_system = EnhancedSearchSystem(
            self.vector_index, 
            self.all_chunks, 
            self.bm25_index, 
            self.model_encoder
        )
        
        self.lsp_analyzer = EnhancedLSPAnalyzer(Path("/Users/vedangbarhate/Desktop/github_projects/rag"))
        self.context_builder = ContextBuilder(self.search_system, self.lsp_analyzer.lsp)
        self.impact_analyzer = ImpactAnalyzer(self.search_system, self.lsp_analyzer)
        self.diff_analyzer = SemanticDiffAnalyzer()
        self.prompt_builder = PromptBuilder()
        self.multi_model_reviewer = MultiModelReviewer(self.api_key)
        # self.output_parser = ReviewOutputParser()
        
        print("Setup complete!")
    
    async def review_pr(self, pr_diff: str, use_ensemble: bool = False) -> ReviewResult:
        """Main entry point for PR review"""
        print("\n" + "="*80)
        print("ENHANCED CODE REVIEW SYSTEM")
        print("="*80)
        
        # 1. Analyze the diff semantically
        print("\n📊 Analyzing diff semantically...")
        changes = self.diff_analyzer.analyze_diff_semantically(pr_diff)
        change_type = self.diff_analyzer.detect_change_type(pr_diff)
        print(f"Change type detected: {change_type}")
        print(f"Files affected: {len(changes.affected_files)}")
        
        # 2. Search for relevant context
        print("\n🔍 Searching for relevant context...")
        search_results = self.search_system.enhanced_search(pr_diff, k=10)
        print(f"Found {len(search_results)} relevant code chunks")
        
        # 3. Build multi-hop context
        print("\n🕸️ Building multi-hop context...")
        expanded_context = self.context_builder.get_multi_hop_context(search_results[:5], depth=2)
        
        # 4. Build dependency context
        print("\n🔗 Analyzing dependencies...")
        dependency_context = {}
        for file in changes.affected_files:
            dependency_context[file] = self.context_builder.build_dependency_context(file)
        
        # 5. Perform LSP analysis
        print("\n🔬 Performing LSP analysis...")
        self.lsp_analyzer.start()
        lsp_results = {}
        try:
            for func in changes.new_functions + changes.modified_functions:
                if 'name' in func and 'file' in func:
                    lsp_results[func['name']] = self.lsp_analyzer.comprehensive_analysis(
                        func['file'], 
                        func['name']
                    )
        finally:
            self.lsp_analyzer.shutdown()
        
        # 6. Analyze impact
        print("\n💥 Analyzing change impact...")
        impact = self.impact_analyzer.analyze_change_impact(pr_diff, {'lsp': lsp_results})
        print(f"Directly affected files: {len(impact.directly_affected)}")
        print(f"Security risks: {len(impact.security_risks)}")
        print(f"Performance risks: {len(impact.performance_risks)}")
        
        # 7. Format context for prompt
        context_str = self.format_context(expanded_context)
        lsp_str = self.format_lsp_results(lsp_results)
        
        # 8. Build comprehensive prompt
        print("\n📝 Building review prompt...")
        full_context = {
            'semantic_context': context_str,
            'dependencies': dependency_context,
            'lsp_analysis': lsp_str
        }
        
        analysis_data = {
            'changes': changes.__dict__,
            'impact': impact.__dict__
        }
        
        prompt = self.prompt_builder.build_enhanced_prompt(
            pr_diff, 
            full_context, 
            analysis_data,
            change_type
        )

        print("Prompt built successfully!")
        print(prompt)
        
        # 9. Perform review
        print("\n🤖 Performing code review...")
        if use_ensemble:
            print("Using ensemble review (multiple models)...")
            review_results = await self.multi_model_reviewer.ensemble_review(
                pr_diff, 
                context_str, 
                prompt
            )
            raw_review = review_results['combined']
        else:
            print("Using single model review...")
            raw_review = await self.multi_model_reviewer.logic_review(
                pr_diff, 
                context_str, 
                prompt
            )
        
        # 10. Parse and structure output
        # print("\n📋 Parsing review results...")
        # review_result = self.output_parser.parse_and_enhance_review(raw_review, pr_diff)
        
        # 11. Display results
        # self.display_review_results(review_result)
        
        return raw_review
    
    def format_context(self, context_chunks: List[CodeChunk]) -> str:
        """Format context chunks for prompt"""
        if not context_chunks:
            return "No relevant context found"
        
        formatted = []
        for chunk in context_chunks[:10]:  # Limit to prevent prompt overflow
            formatted.append(f"""
File: {chunk['file_path']}
Function/Class: {chunk['name']}
Type: {chunk.get('type', 'unknown')}
```python
{chunk['code']}
```
""")
        
        return "\n---\n".join(formatted)
    
    def format_lsp_results(self, lsp_results: Dict[str, Any]) -> str:
        """Format LSP results for prompt"""
        if not lsp_results:
            return "No LSP analysis available"
        
        formatted = []
        for func_name, analysis in lsp_results.items():
            refs = analysis.get('references', {}).get('result', [])
            ref_count = len(refs)
            
            formatted.append(f"""
Function: {func_name}
- References found: {ref_count}
- Has diagnostics: {'diagnostics' in analysis and len(analysis['diagnostics']) > 0}
""")
            
            if refs and ref_count <= 5:  # Show actual references if not too many
                formatted.append("Referenced in:")
                for ref in refs:
                    file_path = ref['uri'].replace('file://', '')
                    line = ref['range']['start']['line'] + 1
                    formatted.append(f"  - {file_path}:{line}")
        
        return "\n".join(formatted)
    
    def display_review_results(self, result: ReviewResult):
        """Display review results in a formatted way"""
        print("\n" + "="*80)
        print("📊 REVIEW RESULTS")
        print("="*80)
        
        # Blocking issues
        if result.blocking:
            print("\n🚫 BLOCKING ISSUES:")
            for i, issue in enumerate(result.blocking, 1):
                print(f"\n{i}. [{issue['type']}] {issue['description']}")
                if issue.get('why'):
                    print(f"   Why: {issue['why']}")
                if issue.get('fix'):
                    print(f"   Fix: {issue['fix']}")
        
        # Suggestions
        if result.suggestions:
            print("\n💭 SUGGESTIONS:")
            for i, suggestion in enumerate(result.suggestions, 1):
                print(f"\n{i}. [{suggestion['type']}] {suggestion['description']}")
                if suggestion.get('fix'):
                    print(f"   Suggestion: {suggestion['fix']}")
        
        # Nits
        if result.nits:
            print("\n💅 NITS:")
            for nit in result.nits:
                print(f"- {nit['description']}")
        
        # Metrics
        print("\n📈 METRICS:")
        for key, value in result.metrics.items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
        
        # Auto-fixes
        if result.auto_fixes:
            print("\n🔧 AUTO-FIXES AVAILABLE:")
            for fix in result.auto_fixes:
                print(f"\n{fix['description']}:")
                print(f"```python\n{fix['code']}\n```")
        
        # Summary
        total_issues = len(result.blocking) + len(result.suggestions) + len(result.nits)
        if total_issues == 0:
            print("\n✅ LGTM - No issues found!")
        else:
            print(f"\n📊 Total issues: {total_issues} ({len(result.blocking)} blocking)")

# ==================== Main Execution ====================

async def main():
    """Main execution function"""
    # Load API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        return
    
    # Initialize system
    review_system = EnhancedCodeReviewSystem(api_key)
    
    # Example PR diff (same as before for testing)
    # Using proper diff format
    with open('diff.txt', 'r') as f:
        pr_diff = f.read()
    
    # Run the review
    try:
        # Run with single model (faster)
        print("\n🚀 Running enhanced code review...")
        result = await review_system.review_pr(pr_diff, use_ensemble=True)
        
        # Optionally run with ensemble (more thorough but slower)
        # print("\n🚀 Running ensemble review with multiple models...")
        # result = await review_system.review_pr(pr_diff, use_ensemble=True)
        
        # Save results
        # print("\n💾 Saving review results...")
        # with open("review_results.json", "w") as f:
        #     json.dump({
        #         'timestamp': datetime.now().isoformat(),
        #         'blocking_count': len(result.blocking),
        #         'suggestion_count': len(result.suggestions),
        #         'nit_count': len(result.nits),
        #         'metrics': result.metrics,
        #         'auto_fixes': result.auto_fixes
        #     }, f, indent=2)
        
        # print("\n✅ Review complete! Results saved to review_results.json")
        
    except Exception as e:
        print(f"\n❌ Error during review: {e}")
        import traceback
        traceback.print_exc()

# ==================== Streaming Review (Bonus Feature) ====================

class StreamingReviewer:
    """Progressive review with streaming results"""
    
    def __init__(self, review_system: EnhancedCodeReviewSystem):
        self.review_system = review_system
        self.quick_checks = QuickChecks()
    
    async def stream_review(self, pr_diff: str):
        """Stream review results as they're generated"""
        print("\n🌊 Starting streaming review...\n")
        
        # Quick style checks
        yield "🔍 Running quick style checks...\n"
        style_issues = await self.quick_checks.check_style(pr_diff)
        if style_issues:
            yield "Style issues found:\n"
            for issue in style_issues:
                yield f"  - {issue}\n"
        else:
            yield "✅ No style issues found\n"
        yield "\n"
        
        # Security scan
        yield "🔒 Scanning for security issues...\n"
        security_issues = await self.quick_checks.scan_security(pr_diff)
        if security_issues:
            yield "⚠️ Security concerns:\n"
            for issue in security_issues:
                yield f"  - {issue}\n"
        else:
            yield "✅ No obvious security issues\n"
        yield "\n"
        
        # Deep analysis
        yield "🧠 Performing deep code analysis...\n"
        yield "This may take a moment...\n\n"
        
        # Build context and prompt
        changes = self.review_system.diff_analyzer.analyze_diff_semantically(pr_diff)
        search_results = self.review_system.search_system.enhanced_search(pr_diff, k=5)
        context_str = self.review_system.format_context(search_results)
        
        prompt = self.review_system.prompt_builder.build_enhanced_prompt(
            pr_diff,
            {'semantic_context': context_str},
            {'changes': changes.__dict__},
            'general'
        )
        print("Prompt built successfully!")
        print(prompt)
        
        # Stream the response
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = await model.generate_content_async(prompt, stream=True)
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text

class QuickChecks:
    """Fast preliminary checks"""
    
    async def check_style(self, pr_diff: str) -> List[str]:
        """Quick style checks without AI"""
        issues = []
        
        lines = pr_diff.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('+'):
                content = line[1:]
                
                # Line length
                if len(content) > 100:
                    issues.append(f"Line {i}: exceeds 100 characters")
                
                # Trailing whitespace
                if content.rstrip() != content:
                    issues.append(f"Line {i}: trailing whitespace")
                
                # TODO comments
                if 'TODO' in content or 'FIXME' in content:
                    issues.append(f"Line {i}: contains TODO/FIXME")
                
                # Print statements
                if 'print(' in content and 'DEBUG' not in content:
                    issues.append(f"Line {i}: contains print statement")
        
        return issues
    
    async def scan_security(self, pr_diff: str) -> List[str]:
        """Quick security checks without AI"""
        issues = []
        
        # Dangerous patterns
        dangerous_patterns = [
            (r'eval\s*\(', "Use of eval() - potential code injection"),
            (r'exec\s*\(', "Use of exec() - potential code injection"),
            (r'pickle\.loads', "Pickle deserialization - potential security risk"),
            (r'os\.system', "Use of os.system - consider subprocess instead"),
            (r'shell\s*=\s*True', "Shell injection risk with subprocess"),
            (r'password\s*=\s*["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\']', "Hardcoded API key detected"),
            (r'\.format\s*\(.*SELECT|INSERT|UPDATE|DELETE', "Potential SQL injection"),
            (r'f["\'].*SELECT|INSERT|UPDATE|DELETE', "Potential SQL injection with f-string"),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, pr_diff, re.IGNORECASE):
                issues.append(message)
        
        return issues

# ==================== CLI Interface ====================

def create_cli_interface():
    """Create a command-line interface for the review system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AI Code Review System')
    parser.add_argument('--diff-file', type=str, help='Path to diff file')
    parser.add_argument('--pr-url', type=str, help='GitHub PR URL (not implemented)')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble review')
    parser.add_argument('--stream', action='store_true', help='Stream results')
    parser.add_argument('--output', type=str, default='review_results.json', help='Output file')
    
    return parser

# ==================== Entry Point ====================

if __name__ == "__main__":
    import sys
    
    # Check if running with arguments
    if len(sys.argv) > 1:
        parser = create_cli_interface()
        args = parser.parse_args()
        
        # Load diff
        if args.diff_file:
            with open(args.diff_file, 'r') as f:
                pr_diff = f.read()
        else:
            print("Please provide a diff file with --diff-file")
            sys.exit(1)
        
        # Run review
        async def run_cli():
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("❌ Error: GOOGLE_API_KEY not found in .env file")
                return
            
            review_system = EnhancedCodeReviewSystem(api_key)
            
            if args.stream:
                streamer = StreamingReviewer(review_system)
                async for chunk in streamer.stream_review(pr_diff):
                    print(chunk, end='', flush=True)
            else:
                result = await review_system.review_pr(pr_diff, use_ensemble=args.ensemble)
                
                # Save to specified output
                with open(args.output, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'blocking_count': len(result.blocking),
                        'suggestion_count': len(result.suggestions),
                        'nit_count': len(result.nits),
                        'metrics': result.metrics,
                        'auto_fixes': result.auto_fixes,
                        'raw_review': result.raw_review
                    }, f, indent=2)
                
                print(f"\n✅ Review complete! Results saved to {args.output}")
        
        asyncio.run(run_cli())
    else:
        # Run default example
        asyncio.run(main())

