# POC 2

import os
import pickle
import faiss
from pathlib import Path 
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from lsp_spike import LspClient
from collections import defaultdict
import json
import re

# Import the custom parser
from diff_parser import parse_unidiff

def quick_extract_symbols_from_hunk(content):
    """Quick and dirty symbol extraction from hunk content."""
    symbols = {
        'functions': [],
        'classes': [],
        'imports': [],
        'variables': [],
        'api_calls': []
    }
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        
        # Functions
        if 'def ' in line:
            match = re.search(r'def\s+(\w+)', line)
            if match:
                symbols['functions'].append(match.group(1))
        
        # Classes
        if 'class ' in line:
            match = re.search(r'class\s+(\w+)', line)
            if match:
                symbols['classes'].append(match.group(1))
        
        # Imports
        if 'import ' in line:
            symbols['imports'].append(line)
        
        # Variable assignments (dirty regex)
        if '=' in line and not '==' in line:
            match = re.match(r'(\w+)\s*=', line)
            if match:
                symbols['variables'].append(match.group(1))
        
        # API/method calls (super dirty - just look for .something())
        api_calls = re.findall(r'\.(\w+)\(', line)
        symbols['api_calls'].extend(api_calls)
    
    return symbols

def create_hunk_embeddings(parsed_diff, model_encoder):
    """Create embeddings for each hunk."""
    hunk_data = []
    
    for file_info in parsed_diff['files']:
        file_path = file_info['path']
        
        for hunk in file_info['hunks']:
            # Create a rich text representation for embedding
            hunk_text = f"File: {file_path}\n"
            hunk_text += f"Lines {hunk['start_line']}-{hunk['end_line']}\n"
            hunk_text += f"Change type: {hunk['change_type']}\n"
            hunk_text += f"Code:\n{hunk['content']}"
            
            # Quick symbol extraction
            symbols = quick_extract_symbols_from_hunk(hunk['content'])
            
            # Add symbols to text for better embedding
            if symbols['functions']:
                hunk_text += f"\nFunctions: {', '.join(symbols['functions'])}"
            if symbols['api_calls']:
                hunk_text += f"\nAPI calls: {', '.join(set(symbols['api_calls']))}"
            
            hunk_data.append({
                'file': file_path,
                'lines': f"{hunk['start_line']}-{hunk['end_line']}",
                'change_type': hunk['change_type'],
                'content': hunk['content'],
                'symbols': symbols,
                'embedding_text': hunk_text,
                'size': hunk['added_lines'] + hunk['removed_lines']
            })
    
    # Create embeddings for all hunks
    if hunk_data:
        texts = [h['embedding_text'] for h in hunk_data]
        embeddings = model_encoder.encode(texts)
        
        for i, hunk in enumerate(hunk_data):
            hunk['embedding'] = embeddings[i]
    
    return hunk_data

def search_similar_code_for_hunks(hunk_data, vector_index, all_chunks, k=2):
    """Find similar code for each hunk."""
    results = {}
    
    for hunk in hunk_data:
        # Skip tiny changes
        if hunk['size'] < 3:
            continue
            
        # Search for similar code
        embedding = hunk['embedding'].reshape(1, -1)
        distances, indices = vector_index.search(embedding, k)
        
        similar_chunks = []
        for idx in indices[0]:
            chunk = all_chunks[idx]
            similar_chunks.append({
                'file': chunk['file_path'],
                'name': chunk['name'],
                'type': chunk['type'],
                'code': chunk['code'][:200] + '...' if len(chunk['code']) > 200 else chunk['code']
            })
        
        results[f"{hunk['file']}:{hunk['lines']}"] = {
            'hunk': hunk,
            'similar_code': similar_chunks
        }
    
    return results

def analyze_change_patterns(hunk_data):
    """Quick analysis of what's changing."""
    patterns = {
        'api_changes': defaultdict(int),
        'modified_functions': [],
        'new_imports': [],
        'large_changes': [],
        'test_changes': []
    }
    
    for hunk in hunk_data:
        # Track API usage changes
        for api in hunk['symbols']['api_calls']:
            patterns['api_changes'][api] += 1
        
        # Track function modifications
        if hunk['change_type'] == 'modified' and hunk['symbols']['functions']:
            patterns['modified_functions'].extend(hunk['symbols']['functions'])
        
        # Track new imports
        if hunk['change_type'] == 'added' and hunk['symbols']['imports']:
            patterns['new_imports'].extend(hunk['symbols']['imports'])
        
        # Flag large changes
        if hunk['size'] > 50:
            patterns['large_changes'].append(f"{hunk['file']}:{hunk['lines']} ({hunk['size']} lines)")
        
        # Test file changes
        if 'test' in hunk['file'].lower():
            patterns['test_changes'].append(hunk['file'])
    
    return patterns

def analyze_import_impact(new_imports, all_chunks):
    """See who else uses these libraries."""
    impact = {}
    for imp in new_imports:
        # Extract library name (dirty)
        lib = imp.split()[1].split('.')[0] if 'import' in imp else imp
        
        # Find other files using it
        users = []
        for chunk in all_chunks:
            if lib in chunk['code'] and lib != '':
                users.append(chunk['file_path'])
        
        impact[lib] = list(set(users))[:5]  # Top 5
    return impact

def check_test_coverage(hunk_data):
    """Quick check if changed functions have tests."""
    changed_functions = set()
    test_functions = set()
    
    for hunk in hunk_data:
        funcs = hunk['symbols']['functions']
        if 'test' not in hunk['file'].lower():
            changed_functions.update(funcs)
        else:
            # Look for test_functionname pattern
            for line in hunk['content'].split('\n'):
                if 'def test_' in line:
                    test_functions.add(line.strip())
                if 'assert' in line and funcs:
                    # Assume this tests the function in this hunk
                    test_functions.update(funcs)
    
    return {
        'changed': list(changed_functions),
        'has_tests': len(test_functions) > 0,
        'test_count': len(test_functions)
    }

def score_hunk_complexity(hunk):
    """Dirty complexity score."""
    score = 0
    content = hunk['content']
    
    # More ifs = more complex
    score += content.count('if ') * 2
    score += content.count('for ') * 3
    score += content.count('while ') * 3
    score += content.count('try:') * 2
    score += content.count('except') * 2
    
    # Nested stuff (look for indented control flow)
    score += content.count('    if ') * 2  # Indented if
    score += content.count('        if ') * 3  # Double indented
    
    # API calls
    score += len(hunk['symbols']['api_calls'])
    
    # Size factor
    score += hunk['size'] // 10
    
    return score

def build_dirty_dep_graph(hunk_data, all_chunks):
    """Who calls what - quick and dirty."""
    graph = defaultdict(set)
    
    # Get all modified functions
    modified_funcs = set()
    for hunk in hunk_data:
        modified_funcs.update(hunk['symbols']['functions'])
    
    # Find who might call these functions
    for func in modified_funcs:
        for chunk in all_chunks[:200]:  # Limit for POC
            if func in chunk['code'] and chunk['name'] != func:
                # Check if it's actually a call (has parenthesis)
                if f"{func}(" in chunk['code']:
                    graph[func].add(f"{chunk['file_path']}:{chunk['name']}")
    
    return dict(graph)

# --- Main Script ---
print("🚀 Enhanced POC with custom diff parser")
load_dotenv()

# Setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Load indexes
vector_index = faiss.read_index("repo.faiss")
with open("repo_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

# Load diff
with open('diff.txt', 'r') as f:
    pr_diff = f.read()

# Parse diff with custom parser
print("\n📊 Parsing diff with custom parser...")
parsed_diff = parse_unidiff(pr_diff)
print(f"Found {len(parsed_diff['files'])} files with changes")

# Create hunk embeddings
print("\n🧮 Creating hunk-level embeddings...")
hunk_data = create_hunk_embeddings(parsed_diff, model_encoder)
print(f"Created embeddings for {len(hunk_data)} hunks")

# Analyze change patterns
print("\n🔍 Analyzing change patterns...")
patterns = analyze_change_patterns(hunk_data)
print(f"- API calls changed: {dict(patterns['api_changes'])}")
print(f"- Modified functions: {patterns['modified_functions']}")
print(f"- New imports: {len(patterns['new_imports'])}")
print(f"- Large changes: {patterns['large_changes']}")

# Search for similar code per hunk
print("\n🔎 Searching for similar code per hunk...")
hunk_similarities = search_similar_code_for_hunks(hunk_data, vector_index, all_chunks)

# Test coverage analysis
print("\n🧪 Analyzing test coverage...")
test_coverage = check_test_coverage(hunk_data)
print(f"- Changed functions: {test_coverage['changed']}")
print(f"- Has tests: {test_coverage['has_tests']}")

# Import impact analysis
print("\n📦 Analyzing import impact...")
import_impact = analyze_import_impact(patterns['new_imports'], all_chunks)
for lib, users in list(import_impact.items())[:3]:
    print(f"- {lib} is used in: {len(users)} files")

# Complexity scoring
print("\n🔥 Scoring complexity...")
complexity_scores = []
for hunk in hunk_data:
    score = score_hunk_complexity(hunk)
    if score > 10:  # Only track complex hunks
        complexity_scores.append((f"{hunk['file']}:{hunk['lines']}", score))
complexity_scores.sort(key=lambda x: x[1], reverse=True)
print(f"Most complex hunks: {complexity_scores[:3]}")

# Dependency graph
print("\n🕸️ Building dependency graph...")
dep_graph = build_dirty_dep_graph(hunk_data, all_chunks)
print(f"Functions with callers: {len(dep_graph)}")

# Build enhanced context
enhanced_context_parts = []
enhanced_context_parts.append("## 📊 Change Analysis:")
enhanced_context_parts.append(f"Total hunks: {len(hunk_data)}")
enhanced_context_parts.append(f"Files affected: {len(parsed_diff['files'])}")

# Add pattern insights
if patterns['api_changes']:
    enhanced_context_parts.append("\n### Most used APIs in changes:")
    for api, count in sorted(patterns['api_changes'].items(), key=lambda x: x[1], reverse=True)[:5]:
        enhanced_context_parts.append(f"- {api}: {count} times")

# Add complexity analysis
if complexity_scores:
    enhanced_context_parts.append("\n### ⚠️ High complexity changes:")
    for location, score in complexity_scores[:3]:
        enhanced_context_parts.append(f"- {location}: complexity score {score}")

# Add test coverage insight
enhanced_context_parts.append(f"\n### 🧪 Test Coverage:")
enhanced_context_parts.append(f"- Functions changed: {len(test_coverage['changed'])}")
enhanced_context_parts.append(f"- Test presence: {'✅ Yes' if test_coverage['has_tests'] else '❌ No'}")

# Add dependency insight
if dep_graph:
    enhanced_context_parts.append(f"\n### 🕸️ Functions with dependencies:")
    for func, callers in list(dep_graph.items())[:3]:
        enhanced_context_parts.append(f"- {func} is called by {len(callers)} locations")

# Add import impact
if import_impact:
    enhanced_context_parts.append(f"\n### 📦 Import impact:")
    for lib, users in list(import_impact.items())[:3]:
        if users:
            enhanced_context_parts.append(f"- {lib}: used in {len(users)} existing files")

# Add hunk-specific similar code
enhanced_context_parts.append("\n### Similar code found for key changes:")
for location, data in list(hunk_similarities.items())[:3]:  # Top 3 for POC
    if data['hunk']['change_type'] in ['modified', 'added']:
        enhanced_context_parts.append(f"\n**{location}** ({data['hunk']['change_type']}):")
        enhanced_context_parts.append(f"Changed code preview: {data['hunk']['content'][:100]}...")
        enhanced_context_parts.append("Similar existing code:")
        for similar in data['similar_code'][:2]:
            enhanced_context_parts.append(f"- {similar['file']} ({similar['type']} {similar['name']})")

enhanced_context = "\n".join(enhanced_context_parts)

# Run LSP on important hunks only (functions that were modified)
print("\n🔧 Running targeted LSP analysis...")
repo_path = Path("/Users/vedangbarhate/Desktop/github_projects/rag").resolve()
important_symbols = set()

for hunk in hunk_data:
    if hunk['change_type'] == 'modified' and hunk['symbols']['functions']:
        important_symbols.update(hunk['symbols']['functions'])

print(f"Focusing LSP on {len(important_symbols)} modified functions")

# Generate review with enhanced context
print("\n🤖 Generating review with enhanced context...")
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

enhanced_prompt = f"""
<review_context>
You will be provided with:
- Code diffs showing changes (additions/deletions)
- LSP context including type information, references, and semantic data
- File context and project structure when relevant
- The purpose or description of the changes
</review_context>

<communication_style>
- Use clear, simple language that junior developers can understand
- Be constructive and educational, not just critical
- Provide specific examples and suggested fixes
- Prioritize issues by severity (Critical → High → Medium → Low)
- Use markdown formatting for code snippets and organization
</communication_style>

<review_categories>
You must systematically review code across these 8 categories:

## 1. LOGICAL ERRORS
Check for bugs that would cause runtime errors or incorrect behavior:
- Null/undefined references
- Type mismatches
- Incorrect conditionals or loops
- Missing return statements
- Race conditions or async issues

## 2. EDGE CASES
Identify unhandled scenarios:
- Empty inputs or arrays
- Boundary values (0, -1, MAX_INT)
- Network failures or timeouts
- Invalid user inputs
- Concurrent access issues

## 3. NAMING & STYLE
Ensure code readability:
- Variable/function names that clearly express intent
- Consistent naming patterns (camelCase, snake_case)
- Proper indentation and formatting
- File and folder organization
- Comment quality and placement

## 4. PERFORMANCE
Spot optimization opportunities:
- Unnecessary loops or computations
- Memory leaks or excessive allocations
- Inefficient algorithms (O(n²) when O(n) possible)
- Database query optimization
- Caching opportunities

## 5. SECURITY
Identify vulnerabilities:
- SQL injection risks
- XSS (Cross-site scripting) vulnerabilities
- Exposed sensitive data or credentials
- Missing input validation/sanitization
- Insecure dependencies
- Authentication/authorization flaws

## 6. CODE CLARITY
Flag confusing code:
- Complex logic needing documentation
- Magic numbers or strings
- Unclear business logic
- Missing or outdated comments
- Functions doing too many things

## 7. DEBUG CODE
Find development artifacts:
- console.log/print statements
- Commented-out code blocks
- TODO/FIXME without tickets
- Test data or mock values
- Development-only configurations

## 8. GENERAL IMPROVEMENTS
Suggest enhancements for:
- Code reusability (DRY principle)
- Design pattern applications
- Error handling improvements
- Test coverage gaps
- Dependency updates
- Architecture considerations
</review_categories>

---

## Context Supplied to You

### 1. Enhanced Change Analysis:
{enhanced_context}

### 2. Risk Indicators:
- **Test Coverage**: {'✅ Tests present' if test_coverage['has_tests'] else '❌ No tests found for ' + str(len(test_coverage['changed'])) + ' changed functions'}
- **Complexity Hotspots**: {[f"{loc} (score: {score})" for loc, score in complexity_scores[:3]] if complexity_scores else "No complex changes"}
- **Breaking Change Risk**: {len(dep_graph)} functions have external callers

### 3. Dependency Impact:
```json
{json.dumps({k: list(v) for k, v in dep_graph.items()}, indent=2) if dep_graph else "No function dependencies found"}
```

### 4. Import Analysis:
{json.dumps({k: f"{len(v)} files" for k, v in list(import_impact.items())[:3]}, indent=2) if import_impact else "No new imports"}

### 5. Similar Code Found:
{chr(10).join([f"- {loc}: similar to {data['similar_code'][0]['file']}" for loc, data in list(hunk_similarities.items())[:3]]) if hunk_similarities else "No similar patterns found"}

---

## Task

Review the following code change:

```diff
{pr_diff}...  # Truncated for POC
```

---
## Output Format (Markdown)

```markdown
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
  // Concrete code example
  ```

### ⚠️ IMPORTANT
*Should fix: maintainability, performance, quality issues.*

#### [Issue Title]
- **Location**: `file_path:line_start-line_end`
- **Issue**: ...
- **Why fix**: Value gained
- **Suggestion**: Approach or snippet

### 💡 CONSIDER
*Nice-to-have improvements.*
- **Location**: Brief suggestion

---

Include code snippets when helpful:
```language
// Current
problematic code here

// Suggested
improved code here

```


**Summary**
One or two sentences on overall quality and key concerns.
```

### Important Rules
1. **Reference real lines from the diff**
2. **Use complexity scores to prioritize review**  
3. **Check dependency graph for breaking changes**
4. **Flag missing tests using coverage analysis**
5. **Keep it actionable**—issue, impact, fix

Remember: Your goal is to help developers write better, safer, more maintainable code. Be thorough but also pragmatic about what truly matters for code quality.


---
"""

try:
    response = gemini_model.generate_content(enhanced_prompt)
    print("\n✨ ENHANCED REVIEW:")
    print("-" * 50)
    print(response.text)
except Exception as e:
    print(f"Error: {e}")