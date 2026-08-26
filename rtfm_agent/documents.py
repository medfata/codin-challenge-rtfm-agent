"""Step 1: Load and parse documentation .asc files."""

import os
from pathlib import Path

# File extensions treated as documentation sources. Step 13 web-crawl pages
# are persisted as markdown (.md) and merged into the same corpus.
DOC_EXTENSIONS = (".asc", ".md")


def extract_heading(content: str) -> str | None:
    """Extract the first heading line from AsciiDoc content.

    Looks for == Heading (chapter level) or === Sub-heading (section level).
    Markdown h1 (`# Heading`) is recognised too so crawled pages stored as
    .md surface their titles.
    """
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("== "):
            return stripped[3:].strip()
        if stripped.startswith("=== "):
            return stripped[4:].strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            return stripped[2:].strip()
    return None


def load_asc_files(directory: str, max_files: int | None = None,
                   recursive: bool = False,
                   extensions: tuple[str, ...] = DOC_EXTENSIONS) -> list[dict]:
    """Load documentation files from directory.

    By default only top-level files are loaded (the sample docs). Set
    recursive=True to walk subdirectories as well. `extensions` selects
    which suffixes count as docs (AsciiDoc + markdown since Step 13).

    Returns list of dicts with keys:
        - source_file: path relative to `directory`, posix-style
          (e.g., "ch01-getting-started.asc" or "book/01-introduction/01-getting-started.asc")
        - heading: first heading found, or derived from filename
        - content: full file text
    """
    docs = []
    base = Path(directory)

    def iter_paths():
        seen = set()
        for ext in extensions:
            pattern = base.rglob(f"*{ext}") if recursive else base.glob(f"*{ext}")
            for path in sorted(pattern):
                if path in seen or ".git" in path.parts:
                    continue
                seen.add(path)
                yield path

    for doc_file in iter_paths():
        try:
            content = doc_file.read_text(encoding="utf-8", errors="replace")
        except UnicodeDecodeError:
            try:
                content = doc_file.read_text(encoding="latin-1", errors="replace")
            except Exception:
                continue
        except OSError:
            continue

        heading = extract_heading(content)
        # fallback: use stem (filename without extension) as heading
        if heading is None:
            heading = doc_file.stem.replace("_", " ")

        # Relative posix path keeps keys unique when two dirs contain
        # same-named files (e.g., book/01-introduction/.../command-line.asc)
        source_file = doc_file.relative_to(base).as_posix()

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