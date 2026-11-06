"""Fingerprint local dependencies of trainV10, not unrelated project scripts.

Parse imports without executing the trainer. Local packages are conservatively
covered in full (Python/YAML) for dynamic model configuration imports. External
installed libraries are environment dependencies, not part of this manifest.
"""
import ast
import hashlib
from pathlib import Path


def training_source_snapshot(root):
    root = Path(root).resolve()
    pending = [root/'trainV10.py']
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(path)
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
                base = root
            elif isinstance(node, ast.ImportFrom):
                base = path.parent if node.level else root
                for _ in range(max(0, node.level-1)):
                    base = base.parent
                names = [node.module or '']
                names += ['.'.join(filter(None, [node.module, a.name])) for a in node.names]
            else:
                continue
            for name in names:
                top = base / name.split('.')[0] if name else base
                module = top.with_suffix('.py')
                if module.is_file() and module.is_relative_to(root):
                    pending.append(module)
                elif top.is_dir() and top.is_relative_to(root) and (top/'__init__.py').is_file():
                    pending.extend(p for p in top.rglob('*') if p.suffix in ('.py', '.yaml', '.yml') and '__pycache__' not in p.parts)
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(found)}


def verify_training_sources(root, expected):
    current = training_source_snapshot(root)
    changes = sorted(k for k in expected.keys() | current.keys() if expected.get(k) != current.get(k))
    if changes:
        raise RuntimeError('Training dependency files changed: ' + ', '.join(changes))
