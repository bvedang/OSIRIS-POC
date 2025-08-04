from __future__ import annotations

import io
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum

from unidiff import PatchSet


class IdentifierType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE = "type"
    ENUM = "enum"
    COMPONENT = "component"
    HOOK = "hook"
    EXPORT = "export"
    IMPORT = "import"


@dataclass
class ExtractedIdentifier:
    name: str
    identifier_type: IdentifierType
    is_default_export: bool = False
    is_async: bool = False
    is_generator: bool = False
    generic_params: Optional[str] = None


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


class LanguageParser(ABC):
    """Abstract base class for language-specific parsers."""
    
    @abstractmethod
    def extract_function_name(self, line: str) -> Optional[str]:
        """Extract function name from a line if it contains a function declaration."""
        pass
    
    @abstractmethod
    def extract_class_name(self, line: str) -> Optional[str]:
        """Extract class name from a line if it contains a class declaration."""
        pass
    
    @abstractmethod
    def extract_import_modules(self, line: str) -> Set[str]:
        """Extract module names from import statements."""
        pass
    
    @abstractmethod
    def get_language_name(self) -> str:
        """Return the language name for this parser."""
        pass
    
    def extract_identifiers(self, content: str) -> Dict[str, Set[str]]:
        """Extract all identifiers from code content."""
        functions = set()
        classes = set()
        imports = set()
        
        for line in content.split('\n'):
            if func := self.extract_function_name(line):
                functions.add(func)
            elif cls := self.extract_class_name(line):
                classes.add(cls)
            else:
                imports.update(self.extract_import_modules(line))
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports
        }


class PythonParser(LanguageParser):
    """Parser for Python files."""
    
    FUNC_RE = re.compile(r"^\s*(async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
    CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[:\(]")
    FROM_IMPORT_RE = re.compile(r"^\s*from\s+((?:\.[\.]*)?[A-Za-z_][\w.]*)\s+import")
    IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)")

    def extract_function_name(self, line: str) -> Optional[str]:
        if m := self.FUNC_RE.match(line):
            return m.group(2)
        return None
    
    def extract_class_name(self, line: str) -> Optional[str]:
        if m := self.CLASS_RE.match(line):
            return m.group(1)
        return None
    
    def extract_import_modules(self, line: str) -> Set[str]:
        modules: Set[str] = set()
        
        if m := self.FROM_IMPORT_RE.match(line):
            module = m.group(1)
            if not module.startswith('.'):
                modules.add(module.split('.')[0])
        elif m := self.IMPORT_RE.match(line):
            import_segment = line[m.start(1):]
            for item in re.split(r'\s*,\s*', import_segment):
                item = item.strip().split()[0]
                if item:
                    modules.add(item.split('.')[0])
        
        return modules
    
    def get_language_name(self) -> str:
        return 'python'


class JavaScriptParser(LanguageParser):
    """Parser for JavaScript/TypeScript files (including JSX/TSX)."""
    
    def __init__(self):
        super().__init__()
        self.FUNCTION_PATTERNS = [
            (re.compile(r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s*(\*)?\s*([A-Za-z_$][\w$]*)\s*(<[^>]+>)?\s*\("), 
             lambda m: (m.group(5), bool(m.group(3)), bool(m.group(4)), m.group(6), bool(m.group(2)))),
            (re.compile(r"^\s*(export\s+)?(const|let|var)\s+([A-Za-z_$][\w$]*)\s*:\s*\([^)]*\)\s*=>\s*[^=]"), 
             lambda m: (m.group(3), False, False, None, False)),
            (re.compile(r"^\s*(export\s+)?(const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(async\s+)?(<[^>]+>)?\s*\([^)]*\)\s*=>"), 
             lambda m: (m.group(3), bool(m.group(4)), False, m.group(5), False)),
            (re.compile(r"^\s*(export\s+)?(const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(async\s+)?([A-Za-z_$][\w$]*)\s*=>"), 
             lambda m: (m.group(3), bool(m.group(4)), False, None, False)),
            (re.compile(r"^\s*(export\s+)?(const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(async\s+)?function\s*(\*)?\s*([A-Za-z_$][\w$]*)?\s*\("), 
             lambda m: (m.group(3), bool(m.group(4)), bool(m.group(5)), None, False)),
            (re.compile(r"^\s*(static\s+)?(async\s+)?(get|set)?\s*([A-Za-z_$][\w$]*)\s*(<[^>]+>)?\s*\([^)]*\)\s*[\{:]"), 
             lambda m: (m.group(4), bool(m.group(2)), False, m.group(5), False)),
            (re.compile(r"^\s*constructor\s*\([^)]*\)\s*\{"), 
             lambda m: ("constructor", False, False, None, False)),
            (re.compile(r"^\s*(export\s+)?(const|let|var)\s+(use[A-Z][A-Za-z]*)\s*="), 
             lambda m: (m.group(3), False, False, None, False)),
            (re.compile(r"^\s*(export\s+)?function\s+(use[A-Z][A-Za-z]*)\s*(<[^>]+>)?\s*\("), 
             lambda m: (m.group(2), False, False, m.group(3), False)),
            (re.compile(r"^\s*(export\s+)?(default\s+)?function\s+([A-Z][A-Za-z]*)\s*(<[^>]+>)?\s*\("), 
             lambda m: (m.group(3), False, False, m.group(4), bool(m.group(2)))),
            (re.compile(r"^\s*(export\s+)?(default\s+)?(const|let|var)\s+([A-Z][A-Za-z]*)\s*:\s*(?:React\.)?(?:FC|FunctionComponent)"), 
             lambda m: (m.group(4), False, False, None, bool(m.group(2)))),
            (re.compile(r"^\s*(export\s+)?(default\s+)?(const|let|var)\s+([A-Z][A-Za-z]*)\s*=\s*(?:\([^)]*\)|[^=]+)\s*=>"), 
             lambda m: (m.group(4), False, False, None, bool(m.group(2)))),
            (re.compile(r"^\s*(export\s+)?(const|let|var)\s+(with[A-Z][A-Za-z]*)\s*="), 
             lambda m: (m.group(3), False, False, None, False)),
            (re.compile(r"^\s*([A-Za-z_$][\w$]*)\s*:\s*(async\s+)?function\s*(\*)?\s*\("), 
             lambda m: (m.group(1), bool(m.group(2)), bool(m.group(3)), None, False)),
            (re.compile(r"^\s*\(\s*(async\s+)?function\s*(\*)?\s*([A-Za-z_$][\w$]*)?\s*\([^)]*\)\s*\{"), 
             lambda m: (m.group(3) or "anonymous", bool(m.group(1)), bool(m.group(2)), None, False)),
        ]
        
        self.CLASS_PATTERNS = [
            (re.compile(r"^\s*(export\s+)?(default\s+)?(abstract\s+)?class\s+([A-Za-z_$][\w$]*)\s*(<[^>]+>)?\s*(?:extends\s+[^{]+)?\s*\{"), 
             lambda m: (m.group(4), IdentifierType.CLASS, m.group(5), bool(m.group(2)))),            
            (re.compile(r"^\s*(export\s+)?(const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*class\s*([A-Za-z_$][\w$]*)?\s*(?:extends\s+[^{]+)?\s*\{"), 
             lambda m: (m.group(3), IdentifierType.CLASS, None, False)),
            (re.compile(r"^\s*(export\s+)?(interface)\s+([A-Za-z_$][\w$]*)\s*(<[^>]+>)?\s*(?:extends\s+[^{]+)?\s*\{"), 
             lambda m: (m.group(3), IdentifierType.INTERFACE, m.group(4), False)),
            (re.compile(r"^\s*(export\s+)?type\s+([A-Za-z_$][\w$]*)\s*(<[^>]+>)?\s*="), 
             lambda m: (m.group(2), IdentifierType.TYPE, m.group(3), False)),
            (re.compile(r"^\s*(export\s+)?(const\s+)?enum\s+([A-Za-z_$][\w$]*)\s*\{"), 
             lambda m: (m.group(3), IdentifierType.ENUM, None, False)),
            (re.compile(r"^\s*(export\s+)?(namespace|module)\s+([A-Za-z_$][\w$]*)\s*\{"), 
             lambda m: (m.group(3), IdentifierType.CLASS, None, False)),
        ]
        
        self.EXPORT_PATTERNS = [
            re.compile(r"^\s*export\s*\{\s*([^}]+)\s*\}"),
            re.compile(r"^\s*export\s*\*\s*(?:as\s+([A-Za-z_$][\w$]*)\s*)?from"),
            re.compile(r"^\s*export\s+default\s+([A-Za-z_$][\w$]*)(?:\s|;|$)"),
            re.compile(r"^\s*export\s*\{\s*([^}]+)\s*\}\s*from"),
            re.compile(r"^\s*export\s*=\s*([A-Za-z_$][\w$]*)"),
            re.compile(r"^\s*module\.exports\s*="),
            re.compile(r"^\s*exports\.([A-Za-z_$][\w$]*)\s*="),
        ]
        
        self.IMPORT_PATTERNS = [
            (re.compile(r"^\s*import\s*\{\s*([^}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]"), 
             lambda m: (self._parse_import_list(m.group(1)), m.group(2))),
            (re.compile(r"^\s*import\s+([A-Za-z_$][\w$]*)\s+from\s*['\"]([^'\"]+)['\"]"), 
             lambda m: ([m.group(1)], m.group(2))),
            (re.compile(r"^\s*import\s+([A-Za-z_$][\w$]*)\s*,\s*\{\s*([^}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]"), 
             lambda m: ([m.group(1)] + self._parse_import_list(m.group(2)), m.group(3))),
            (re.compile(r"^\s*import\s*\*\s*as\s+([A-Za-z_$][\w$]*)\s+from\s*['\"]([^'\"]+)['\"]"), 
             lambda m: ([m.group(1)], m.group(2))),
            (re.compile(r"^\s*import\s*['\"]([^'\"]+)['\"]"), 
             lambda m: ([], m.group(1))),
            (re.compile(r"^\s*import\s+type\s*\{\s*([^}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]"), 
             lambda m: (self._parse_import_list(m.group(1)), m.group(2))),
            (re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"), 
             lambda m: ([m.group(1)], m.group(2))),
            (re.compile(r"^\s*(?:const|let|var)\s*\{\s*([^}]+)\s*\}\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"), 
             lambda m: (self._parse_import_list(m.group(1)), m.group(2))),
            (re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:await\s+)?import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"), 
             lambda m: ([m.group(1)], m.group(2))),
        ]
    
    def _parse_import_list(self, import_str: str) -> List[str]:
        """Parse comma-separated import list, handling 'as' aliases."""
        items = []
        for item in import_str.split(','):
            item = item.strip()
            if ' as ' in item:
                items.append(item.split(' as ')[1].strip())
            else:
                items.append(item)
        return items
    
    def extract_all_identifiers(self, line: str) -> List[ExtractedIdentifier]:
        """Extract all identifiers from a line."""
        identifiers = []
        
        for pattern, extractor in self.FUNCTION_PATTERNS:
            if match := pattern.match(line):
                name, is_async, is_generator, generics, is_default = extractor(match)
                if name:
                    id_type = IdentifierType.FUNCTION
                    if name.startswith('use') and len(name) > 3 and name[3].isupper():
                        id_type = IdentifierType.HOOK
                    elif name[0].isupper():
                        id_type = IdentifierType.COMPONENT
                    
                    identifiers.append(ExtractedIdentifier(
                        name=name,
                        identifier_type=id_type,
                        is_default_export=is_default,
                        is_async=is_async,
                        is_generator=is_generator,
                        generic_params=generics
                    ))
        
        for pattern, extractor in self.CLASS_PATTERNS:
            if match := pattern.match(line):
                name, id_type, generics, is_default = extractor(match)
                if name:
                    identifiers.append(ExtractedIdentifier(
                        name=name,
                        identifier_type=id_type,
                        is_default_export=is_default,
                        generic_params=generics
                    ))
        
        for pattern in self.EXPORT_PATTERNS:
            if match := pattern.match(line):
                identifiers.append(ExtractedIdentifier(
                    name="export",
                    identifier_type=IdentifierType.EXPORT,
                    is_default_export='default' in line
                ))
        
        return identifiers
    
    def extract_function_name(self, line: str) -> Optional[str]:
        """Extract function name from a line."""
        identifiers = self.extract_all_identifiers(line)
        for id in identifiers:
            if id.identifier_type in [IdentifierType.FUNCTION, IdentifierType.HOOK, IdentifierType.COMPONENT]:
                return id.name
        return None
    
    def extract_class_name(self, line: str) -> Optional[str]:
        """Extract class/interface/type name from a line."""
        identifiers = self.extract_all_identifiers(line)
        for id in identifiers:
            if id.identifier_type in [IdentifierType.CLASS, IdentifierType.INTERFACE, 
                                     IdentifierType.TYPE, IdentifierType.ENUM]:
                return id.name
        return None
    
    def extract_import_modules(self, line: str) -> Set[str]:
        """Extract module names from import statements."""
        modules: Set[str] = set()
        for pattern, extractor in self.IMPORT_PATTERNS:
            if match := pattern.match(line):
                _, module = extractor(match)
                if module:
                    modules.add(self.get_module_name(module))
        return modules
    
    def get_module_name(self, module_path: str) -> str:
        """Extract the base module name from a path."""
        if module_path.startswith('.'):
            return '(relative)'
        
        if module_path.startswith('@'):
            parts = module_path.split('/')
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        
        return module_path.split('/')[0]
    
    def get_language_name(self) -> str:
        return 'javascript'
    
    def get_detailed_identifiers(self, line: str) -> List[ExtractedIdentifier]:
        """Get detailed identifier information for advanced analysis."""
        return self.extract_all_identifiers(line)


class ParserFactory:
    """Factory to create appropriate parser based on file extension."""
    
    _parsers = {
        '.py': PythonParser(),
        '.js': JavaScriptParser(),
        '.jsx': JavaScriptParser(),
        '.ts': JavaScriptParser(),
        '.tsx': JavaScriptParser(),
        '.mjs': JavaScriptParser(),
        '.cjs': JavaScriptParser(),
    }
    
    @classmethod
    def get_parser(cls, file_path: str) -> LanguageParser:
        """Get appropriate parser for file path."""
        ext = os.path.splitext(file_path)[1].lower()
        return cls._parsers.get(ext, PythonParser())
    
    @classmethod
    def register_parser(cls, extension: str, parser: LanguageParser):
        """Register a new parser for a file extension."""
        cls._parsers[extension] = parser


def _blocks_are_related(block1: CodeBlock, block2: CodeBlock, parser: LanguageParser) -> bool:
    """Check if two blocks are related using language-specific parser."""
    ids1 = parser.extract_identifiers(block1.content)
    ids2 = parser.extract_identifiers(block2.content)
    
    return bool(
        (ids1['functions'] & ids2['functions']) or 
        (ids1['classes'] & ids2['classes'])
    )


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
        parser = ParserFactory.get_parser(file_path)
        
        if added and removed and len(added) == len(removed):
            for add_block, rem_block in zip(added, removed):
                if _blocks_are_related(add_block, rem_block, parser):
                    result.modified_before.append(rem_block)
                    result.modified_after.append(add_block)
                    result.added_blocks.remove(add_block)
                    result.removed_blocks.remove(rem_block)

def _clean_diff_prefix(raw: str) -> str:
    return raw[1:] if raw and raw[0] in "+- " else raw


def extract_code_blocks_from_diff(diff_text: str) -> ExtractedDiffContent:
    """Extract code blocks from a diff, categorizing them by operation type."""
    result = ExtractedDiffContent()
    
    for pfile in PatchSet(io.StringIO(diff_text)):
        file_path = re.sub(r'^[ab]/', '', pfile.target_file or pfile.source_file or '')
        parser = ParserFactory.get_parser(file_path)
        language = parser.get_language_name()
        
        for hunk in pfile:
            current_block = []
            current_op = None
            start_line = None
            
            for line_obj in hunk:
                line = line_obj.value.rstrip('\n')
                raw = line_obj.value.rstrip('\n')
                clean = _clean_diff_prefix(raw)
                
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
                        end_line=start_line + len(current_block) - 1 if start_line else 0,
                        language=language
                    )
                    
                    if current_op == 'added':
                        result.added_blocks.append(block)
                    elif current_op == 'removed':
                        result.removed_blocks.append(block)
                    
                    current_block = []
                
                if op != 'context':
                    if not current_block:
                        start_line = (line_obj.target_line_no if op == 'added'else line_obj.source_line_no)
                    current_block.append(clean)
                    current_op = op
            
            if current_block and current_op != 'context':
                block = CodeBlock(
                    content='\n'.join(current_block),
                    file_path=file_path,
                    block_type=current_op,
                    start_line=start_line or 0,
                    end_line=start_line + len(current_block) - 1 if start_line else 0,
                    language=language
                )
                
                if current_op == 'added':
                    result.added_blocks.append(block)
                elif current_op == 'removed':
                    result.removed_blocks.append(block)
    
    _identify_modified_blocks(result)
    
    return result


def extract_identifiers_from_diff(diff_text: str) -> Dict[str, Set[str]]:
    """Extract all identifiers from a diff using language-specific parsers."""
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
        parser = ParserFactory.get_parser(path)
        
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
                txt = _clean_diff_prefix(line.value.rstrip('\n'))\
                
                if line.is_added:
                    op = 'added'
                elif line.is_removed:
                    op = 'removed'
                else:
                    continue
                
                if func := parser.extract_function_name(txt):
                    result[f'functions_{op}'].add(func)
                    file_changes[path][f'funcs_{op}'].add(func)
                elif cls := parser.extract_class_name(txt):
                    result[f'classes_{op}'].add(cls)
                    file_changes[path][f'classes_{op}'].add(cls)
                else:
                    for module in parser.extract_import_modules(txt):
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


def create_code_block_embedding_text(block: CodeBlock) -> str:
    """Create text representation of a code block suitable for embedding."""
    parser = ParserFactory.get_parser(block.file_path)
    parts = []
    
    parts.append(f"file: {block.file_path}")
    parts.append(f"language: {block.language}")
    parts.append(f"change_type: {block.block_type}")
    
    if block.language != 'javascript':
        identifiers = parser.extract_identifiers(block.content)
        
        if identifiers['functions']:
            parts.append(f"functions: {', '.join(list(identifiers['functions'])[:5])}")
        if identifiers['classes']:
            parts.append(f"classes: {', '.join(list(identifiers['classes'])[:5])}")
        if identifiers['imports']:
            parts.append(f"imports: {', '.join(list(identifiers['imports'])[:3])}")
    else:

        js_parser = parser
        functions = []
        classes = []
        imports = []
        components = []
        hooks = []
        exports = []
        
        for line in block.content.split('\n'):
            if hasattr(js_parser, 'get_detailed_identifiers'):
                detailed_ids = js_parser.get_detailed_identifiers(line)
                for id in detailed_ids:
                    if id.identifier_type == IdentifierType.FUNCTION:
                        desc = id.name
                        if id.is_async:
                            desc = f"async {desc}"
                        if id.is_generator:
                            desc = f"*{desc}"
                        functions.append(desc)
                    elif id.identifier_type == IdentifierType.COMPONENT:
                        components.append(id.name)
                    elif id.identifier_type == IdentifierType.HOOK:
                        hooks.append(id.name)
                    elif id.identifier_type in [IdentifierType.CLASS, IdentifierType.INTERFACE, 
                                              IdentifierType.TYPE, IdentifierType.ENUM]:
                        classes.append(f"{id.identifier_type.value}:{id.name}")
                    elif id.identifier_type == IdentifierType.EXPORT:
                        exports.append("default" if id.is_default_export else "named")
            
            import_mods = js_parser.extract_import_modules(line)
            imports.extend(import_mods)
        
        if functions:
            parts.append(f"functions: {', '.join(functions[:3])}")
        if components:
            parts.append(f"components: {', '.join(components[:3])}")
        if hooks:
            parts.append(f"hooks: {', '.join(hooks[:3])}")
        if classes:
            parts.append(f"types: {', '.join(classes[:3])}")
        if exports:
            parts.append(f"exports: {', '.join(set(exports))}")
        if imports:
            parts.append(f"imports: {', '.join(list(set(imports))[:3])}")
    
    code_preview = ' '.join(block.content.split('\n')[:5])
    if len(code_preview) > 200:
        code_preview = code_preview[:200] + "..."
    parts.append(f"code: {code_preview}")
    
    return " | ".join(parts)


