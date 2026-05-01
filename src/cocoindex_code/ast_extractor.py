"""AST-based code extraction using tree-sitter for semantic analysis.

Provides headless AST extraction for the Kit engine, outputting structured
semantic information as JSONL via stdout.
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

# Silence all logging for headless mode
logging.basicConfig(level=logging.CRITICAL)

# Try to import tree-sitter-languages; if not available, we'll handle it gracefully
try:
    import tree_sitter_languages as _tsl_module
    _TREE_SITTER_AVAILABLE = True
    _tsl: Any = _tsl_module
except ImportError:
    _TREE_SITTER_AVAILABLE = False
    _tsl = None


def _get_parser(language: str) -> Any | None:
    """Get tree-sitter parser for the given language.

    Returns None if tree-sitter is not available or language is not supported.
    """
    if not _TREE_SITTER_AVAILABLE or _tsl is None:
        return None

    # Use tree_sitter_languages.get_parser() which is the standard way
    try:
        return _tsl.get_parser(language)
    except Exception:
        return None


def _extract_dependencies(node: Any, _content: str, language: str) -> list[str]:
    """Extract dependencies (imports, function calls, type references) from a node.

    This is a basic implementation; can be enhanced for more sophisticated analysis.
    """
    dependencies: set[str] = set()

    # For Python: extract imports
    if language == "python":
        # Find all import statements and imported names
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name" or child.type == "aliased_import":
                    for name in child.children:
                        if name.type == "identifier":
                            dependencies.add(name.text.decode())
        elif node.type == "import_from_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    dependencies.add(child.text.decode())
                elif child.type == "import_list":
                    for name in child.children:
                        if name.type == "identifier" or name.type == "dotted_name":
                            dependencies.add(name.text.decode())

    # For function/method calls across languages
    if node.type in ("call", "function_call", "method_call"):
        for child in node.children:
            if child.type in ("identifier", "attribute_expression"):
                call_text = child.text.decode()
                # Extract the function/method name
                if "." in call_text:
                    func_name = call_text.split(".")[-1]
                else:
                    func_name = call_text
                builtin_functions = (
                    "print",
                    "len",
                    "str",
                    "int",
                    "float",
                    "list",
                    "dict",
                    "set",
                    "tuple",
                )
                if func_name and func_name not in builtin_functions:
                    dependencies.add(func_name)

    return sorted(dependencies)


def _extract_node_info(
    node: Any,
    content: bytes,
    file_path: str,
    language: str,
    parent_name: str = "",
) -> dict[str, Any]:
    """Extract information from a single AST node."""
    # Get the full text content of this node
    start_byte = node.start_byte
    end_byte = node.end_byte
    ast_content = content[start_byte:end_byte].decode("utf-8", errors="replace")

    # Determine the node type - unified mapping without duplicates
    node_type_map = {
        # Python
        "function_definition": "function",
        "class_definition": "class",
        "module": "file",
        # JavaScript/TypeScript
        "function_declaration": "function",
        "method_definition": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "interface",
        "program": "file",
        # Go
        "type_declaration": "class",
        "interface_type": "interface",
        "source_file": "file",
        # Rust
        "function_item": "function",
        "impl_item": "class",
        "struct_item": "class",
        "trait_item": "interface",
        # Java
        "method_declaration": "function",
        # C/C++
        "class_specifier": "class",
        "struct_specifier": "class",
        "translation_unit": "file",
    }

    node_type = node_type_map.get(node.type, node.type.lower())

    # Extract symbol name
    symbol_name = ""
    for child in node.children:
        if child.type == "identifier":
            symbol_name = child.text.decode("utf-8", errors="replace")
            break
        elif child.type == "name":
            for name_child in child.children:
                if name_child.type == "identifier":
                    symbol_name = name_child.text.decode("utf-8", errors="replace")
                    break

    # If this is a method inside a class, prefix with parent name
    if parent_name and node_type == "function":
        if symbol_name:
            symbol_name = f"{parent_name}.{symbol_name}"

    # Fallback: if no name found, use node type
    if not symbol_name:
        if node_type == "file":
            symbol_name = file_path
        else:
            symbol_name = f"<anonymous_{node_type}>"

    # Extract dependencies
    dependencies = _extract_dependencies(node, ast_content, language)

    return {
        "file_path": file_path,
        "symbol_name": symbol_name,
        "node_type": node_type,
        "ast_content": ast_content,
        "dependencies": dependencies,
    }


def _traverse_ast(
    node: Any,
    content: bytes,
    file_path: str,
    language: str,
    parent_name: str = "",
) -> Generator[dict[str, Any], None, None]:
    """Traverse the AST and yield extracted node information.

    Implements flattening: nested nodes (like methods in classes) are yielded
    as separate, independent chunks with hierarchical symbol names.
    """
    # Extract current node info if it's a significant type
    significant_types = {
        "function_definition",
        "class_definition",
        "function_declaration",
        "method_definition",
        "class_declaration",
        "interface_declaration",
        "type_declaration",
        "function_item",
        "impl_item",
        "struct_item",
        "trait_item",
        "method_declaration",
        "interface_type",
        "type_alias_declaration",
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "module",
        "program",
        "source_file",
        "translation_unit",
    }

    if node.type in significant_types:
        node_info = _extract_node_info(node, content, file_path, language, parent_name)
        yield node_info

        # For class/module nodes, traverse children with updated parent name
        if node.type in ("class_definition", "class_declaration", "impl_item", "module"):
            # Get the class/module name for children
            current_name = node_info["symbol_name"]
            if current_name and not current_name.startswith("<anonymous_"):
                child_parent = current_name
            else:
                child_parent = parent_name

            for child in node.children:
                yield from _traverse_ast(child, content, file_path, language, child_parent)
        else:
            # For other nodes, just traverse without updating parent
            for child in node.children:
                yield from _traverse_ast(child, content, file_path, language, parent_name)
    else:
        # Recurse into children
        for child in node.children:
            yield from _traverse_ast(child, content, file_path, language, parent_name)


def extract_from_file(file_path: Path, project_root: Path) -> list[dict[str, Any]]:
    """Extract AST nodes from a single file.

    Returns a list of node dictionaries with the schema:
    {
        "file_path": str,
        "symbol_name": str,
        "node_type": str,
        "ast_content": str,
        "dependencies": list[str],
    }
    """
    # Detect language from file extension
    ext_to_lang = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".kt": "kotlin",
    }

    suffix = file_path.suffix.lower()
    language = ext_to_lang.get(suffix)

    if not language:
        return []

    # Get parser
    parser = _get_parser(language)
    if not parser:
        return []

    # Read file content
    try:
        content = file_path.read_bytes()
    except (OSError, UnicodeDecodeError):
        return []

    # Parse the file
    try:
        tree = parser.parse(content)
    except Exception:
        return []

    # Get relative path from project root
    try:
        rel_path = file_path.relative_to(project_root).as_posix()
    except ValueError:
        rel_path = file_path.as_posix()

    # Extract nodes
    nodes = []
    for node_info in _traverse_ast(tree.root_node, content, rel_path, language):
        nodes.append(node_info)

    return nodes


def walk_and_extract(
    project_root: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Walk directory tree and extract AST nodes from all supported files.

    Yields node dictionaries one at a time for streaming output.

    Args:
        project_root: Root directory of the project
        include_patterns: Glob patterns for files to include (default: all code files)
        exclude_patterns: Glob patterns for files to exclude (default: common exclusions)
    """
    from pathspec import PathSpec

    # Default exclude patterns
    if exclude_patterns is None:
        exclude_patterns = [
            "**/.git/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/node_modules/**",
            "**/venv/**",
            "**/.venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.cocoindex_code/**",
            "**/target/**",
            "**/build/**",
            "**/dist/**",
            "**/*.egg-info/**",
        ]

    # Default include patterns (all code files we support)
    if include_patterns is None:
        include_patterns = [
            "**/*.py",
            "**/*.js",
            "**/*.jsx",
            "**/*.ts",
            "**/*.tsx",
            "**/*.go",
            "**/*.rs",
            "**/*.c",
            "**/*.cpp",
            "**/*.cc",
            "**/*.cxx",
            "**/*.h",
            "**/*.hpp",
            "**/*.java",
            "**/*.rb",
            "**/*.php",
            "**/*.kt",
        ]

    # Compile path specs
    include_spec = PathSpec.from_lines("gitwildmatch", include_patterns)
    exclude_spec = PathSpec.from_lines("gitwildmatch", exclude_patterns)

    # Walk the directory
    for file_path in project_root.rglob("*"):
        if not file_path.is_file():
            continue

        rel_path = file_path.relative_to(project_root)
        rel_path_str = rel_path.as_posix()

        # Check include/exclude
        if not include_spec.match_file(rel_path_str):
            continue
        if exclude_spec.match_file(rel_path_str):
            continue

        # Extract from file
        try:
            nodes = extract_from_file(file_path, project_root)
            yield from nodes
        except Exception:
            # Silently skip files that fail to parse
            continue
