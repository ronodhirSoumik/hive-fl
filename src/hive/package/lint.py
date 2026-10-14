"""
AST-based static linter for Hive model packages.
Scans Python source files (model.py, preprocess.py) for suspicious imports,
dangerous built-ins, and file write attempts outside temporary directories.
Note: Static linting provides audit findings; runtime containment is enforced
separately by the sandbox environment.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Modules that indicate networking, process spawning, or low-level memory access
SUSPICIOUS_MODULES = {
    "socket",
    "subprocess",
    "ctypes",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "asyncio.subprocess",
}

# Dangerous dynamic execution builtins
DANGEROUS_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
}


@dataclass(frozen=True)
class LintFinding:
    """Record of a potential security concern detected in package source code."""
    severity: str  # "WARNING" or "ERROR"
    file: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.file}:{self.line}: {self.message}"


class _PackageASTVisitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.findings: list[LintFinding] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if alias.name in SUSPICIOUS_MODULES or base_mod in SUSPICIOUS_MODULES:
                self.findings.append(
                    LintFinding(
                        severity="WARNING",
                        file=self.filename,
                        line=node.lineno,
                        message=f"Disallowed or network-capable module imported: '{alias.name}'",
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0]
            if node.module in SUSPICIOUS_MODULES or base_mod in SUSPICIOUS_MODULES:
                self.findings.append(
                    LintFinding(
                        severity="WARNING",
                        file=self.filename,
                        line=node.lineno,
                        message=f"Disallowed or network-capable module imported: '{node.module}'",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct calls like eval(...), exec(...)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in DANGEROUS_CALLS:
                self.findings.append(
                    LintFinding(
                        severity="WARNING",
                        file=self.filename,
                        line=node.lineno,
                        message=f"Call to dynamic execution builtin '{func_name}'",
                    )
                )
            elif func_name == "open" and len(node.args) >= 2:
                # Check for write mode: open(path, 'w')
                mode_node = node.args[1]
                if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
                    mode_str = mode_node.value
                    if any(c in mode_str for c in ("w", "a", "x", "+")):
                        # Check target path
                        path_arg = node.args[0]
                        is_safe_tmp = False
                        if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
                            if path_arg.value.startswith("/tmp") or path_arg.value.startswith("tmp"):
                                is_safe_tmp = True
                        if not is_safe_tmp:
                            self.findings.append(
                                LintFinding(
                                    severity="WARNING",
                                    file=self.filename,
                                    line=node.lineno,
                                    message=f"File open with write mode '{mode_str}' outside /tmp",
                                )
                            )

        # Check attribute calls like os.system(...)
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in ("system", "popen", "spawn"):
                self.findings.append(
                    LintFinding(
                        severity="WARNING",
                        file=self.filename,
                        line=node.lineno,
                        message=f"Call to OS process execution method '{attr_name}'",
                    )
                )

        self.generic_visit(node)


def lint_source(code: str, filename: str = "<source>") -> list[LintFinding]:
    """
    Parses Python code into an AST and inspects for disallowed patterns.
    Returns a list of LintFinding objects.
    """
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as e:
        return [
            LintFinding(
                severity="ERROR",
                file=filename,
                line=e.lineno or 0,
                message=f"Python syntax error in package source: {e}",
            )
        ]
    visitor = _PackageASTVisitor(filename)
    visitor.visit(tree)
    return visitor.findings


def lint_directory(dir_path: Path | str) -> list[LintFinding]:
    """Scans all Python (.py) files within a directory and aggregates findings."""
    p = Path(dir_path)
    all_findings: list[LintFinding] = []
    if not p.is_dir():
        return all_findings
    for py_file in p.glob("**/*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            all_findings.extend(lint_source(content, filename=py_file.name))
        except Exception as e:
            all_findings.append(
                LintFinding(
                    severity="ERROR",
                    file=py_file.name,
                    line=0,
                    message=f"Unable to read file: {e}",
                )
            )
    return all_findings