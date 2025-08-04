from __future__ import annotations

import io
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

from unidiff import PatchSet


FUNC_RE = re.compile(r"^\s*(async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[:\(]")
FROM_IMPORT_RE = re.compile(r"^\s*from\s+((?:\.[\.]*)?[A-Za-z_][\w.]*)\s+import")
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)")


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

def extract_import_modules(line: str) -> Set[str]:
    """Extract module names from import statements."""
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

def extract_code_blocks_from_diff(diff_text: str) -> ExtractedDiffContent:
    """Extract code blocks from a diff, categorizing them by operation type."""
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
    """Identify blocks that were modified (as opposed to purely added/removed)."""
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
    """Check if two blocks are related (e.g., same function/class being modified)."""
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


def extract_identifiers_from_diff(diff_text: str) -> Dict[str, Set[str]]:
    """Extract all identifiers (functions, classes, imports, files) from a diff."""
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


def extract_import_modules(line: str) -> Set[str]:
    """Extract module names from import statements."""
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


def create_code_block_embedding_text(block: CodeBlock) -> str:
    """Create text representation of a code block suitable for embedding."""
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