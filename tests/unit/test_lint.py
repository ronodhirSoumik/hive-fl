"""
Unit tests for package static code linter.
"""

from hive.package.lint import lint_source


def test_lint_clean_code() -> None:
    clean_code = """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)
"""
    findings = lint_source(clean_code, filename="model.py")
    assert len(findings) == 0


def test_lint_suspicious_imports() -> None:
    code = """
import socket
import subprocess
from requests import get

def exploit():
    s = socket.socket()
"""
    findings = lint_source(code, filename="exploit.py")
    assert len(findings) >= 3
    messages = [f.message for f in findings]
    assert any("socket" in m for m in messages)
    assert any("subprocess" in m for m in messages)
    assert any("requests" in m for m in messages)


def test_lint_dangerous_builtins() -> None:
    code = """
def run_dynamic(cmd):
    eval(cmd)
    exec("import os")
"""
    findings = lint_source(code, filename="dynamic.py")
    assert len(findings) >= 2
    messages = [f.message for f in findings]
    assert any("eval" in m for m in messages)
    assert any("exec" in m for m in messages)


def test_lint_file_write_outside_tmp() -> None:
    code_bad = """
with open("user_data.txt", "w") as f:
    f.write("stolen")
"""
    findings_bad = lint_source(code_bad, filename="steal.py")
    assert len(findings_bad) >= 1
    assert any("outside /tmp" in f.message for f in findings_bad)

    code_safe = """
with open("/tmp/cache.bin", "w") as f:
    f.write("temporary")
"""
    findings_safe = lint_source(code_safe, filename="safe.py")
    assert len(findings_safe) == 0


def test_lint_syntax_error() -> None:
    invalid_code = "def broken_func(: pass"
    findings = lint_source(invalid_code, filename="syntax.py")
    assert len(findings) == 1
    assert findings[0].severity == "ERROR"
    assert "syntax error" in findings[0].message.lower()