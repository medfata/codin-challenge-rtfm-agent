"""Step 1: Load and parse documentation .asc files."""

import os
from pathlib import Path


def extract_heading(content: str) -> str | None:
    """Extract the first heading line from AsciiDoc content.

    Looks for == Heading (chapter level) or === Sub-heading (section level).
    """
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("== "):
            return stripped[3:].strip()
        if stripped.startswith("=== "):
            return stripped[4:].strip()
    return None


def load_asc_files(directory: str, max_files: int | None = None, recursive: bool = False) -> list[dict]:
    """Load .asc files from directory.

    By default only top-level files are loaded (the sample docs). Set
    recursive=True to walk subdirectories as well.

    Returns list of dicts with keys:
        - source_file: path relative to `directory`, posix-style
          (e.g., "ch01-getting-started.asc" or "book/01-introduction/01-getting-started.asc")
        - heading: first heading found, or derived from filename
        - content: full file text
    """
    docs = []
    base = Path(directory)
    pattern = base.rglob("*.asc") if recursive else base.glob("*.asc")

    for asc_file in sorted(pattern):
        # skip .git directory
        if ".git" in asc_file.parts:
            continue

        try:
            content = asc_file.read_text(encoding="utf-8", errors="replace")
        except UnicodeDecodeError:
            try:
                content = asc_file.read_text(encoding="latin-1", errors="replace")
            except Exception:
                continue

        heading = extract_heading(content)
        # fallback: use stem (filename without extension) as heading
        if heading is None:
            heading = asc_file.stem.replace("_", " ")

        # Relative posix path keeps keys unique when two dirs contain
        # same-named files (e.g., book/01-introduction/.../command-line.asc)
        source_file = asc_file.relative_to(base).as_posix()

        docs.append({
            "source_file": source_file,
            "heading": heading,
            "content": content,
        })

        if max_files and len(docs) >= max_files:
            break

    return docs


if __name__ == "__main__":
    docs = load_asc_files("docs/progit2")
    print(f"Loaded {len(docs)} documentation files")
    for doc in docs[:5]:
        print(f'  - {doc["source_file"]}: heading={doc["heading"]!r}')
    if len(docs) > 5:
        print(f"  ... and {len(docs) - 5} more")