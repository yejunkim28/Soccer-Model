from pathlib import Path

SKIP_DIRS = {".venv"}

def print_tree(root: Path, prefix=""):
    entries = [
        p for p in root.iterdir()
        if p.name not in SKIP_DIRS
    ]
    entries.sort(key=lambda p: (p.is_file(), p.name.lower()))

    for i, path in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + path.name)

        if path.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

print_tree(Path("."))