"""Input / output helpers: getting the book, cleaning it, and writing files."""

import csv
import re
from pathlib import Path
from urllib.request import Request, urlopen

import settings as cfg


def create_folders() -> None:
    cfg.OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def download_raw_text() -> bytes:
    request = Request(
        cfg.BOOK_URL,
        headers={"User-Agent": "Python text-mining script"},
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def obtain_raw_bytes() -> bytes:
    """Use the cached copy if we already have it, otherwise download once.

    This way the first run downloads from Gutenberg and saves the file, and
    every run after that (even with no internet) just reads the cache.
    """
    if cfg.RAW_TEXT_PATH.exists():
        print(f"Using cached book at {cfg.RAW_TEXT_PATH}")
        return cfg.RAW_TEXT_PATH.read_bytes()

    print(f"Downloading book from {cfg.BOOK_URL}")
    raw_bytes = download_raw_text()
    cfg.RAW_TEXT_PATH.write_bytes(raw_bytes)
    return raw_bytes


def decode_text(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw_bytes.decode("utf-8", errors="replace")


def normalized_marker_line(line: str) -> str:
    return " ".join(line.strip().upper().split())


def is_gutenberg_marker(line: str, marker_type: str) -> bool:
    normalized = normalized_marker_line(line)
    return (
        normalized.startswith("***")
        and normalized.endswith("***")
        and marker_type in normalized
        and "PROJECT GUTENBERG EBOOK" in normalized
    )


def remove_gutenberg_header_footer(raw_text: str) -> str:
    """Drop the Gutenberg license header/footer and the table of contents,
    keeping the story from the first chapter onward."""
    lines = raw_text.splitlines(keepends=True)
    start_index = None
    end_index = None

    for index, line in enumerate(lines):
        if is_gutenberg_marker(line, "START OF"):
            start_index = index
            break
    if start_index is None:
        raise ValueError("Could not find the Project Gutenberg start marker.")

    for index in range(start_index + 1, len(lines)):
        if is_gutenberg_marker(lines[index], "END OF"):
            end_index = index
            break
    if end_index is None:
        raise ValueError("Could not find the Project Gutenberg end marker.")

    book_lines = lines[start_index + 1 : end_index]
    for index, line in enumerate(book_lines):
        if line.strip() == "CHAPTER I.":
            return "".join(book_lines[index:]).strip()

    raise ValueError('Could not find the first chapter heading: "CHAPTER I."')


def write_text_file(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)


def write_ranked_count_csv(
    path: Path,
    ranked_counts: list[tuple[int, str, int]],
    value_column: str = "token",
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", value_column, "count"])
        writer.writerows(ranked_counts)


def write_repeated_word_matches_csv(
    path: Path,
    matches: list[tuple],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["rank", "match", "repeated_word", "category", "start_index", "context"]
        )
        writer.writerows(matches)


def create_log_frequency_rank_plot(
    ranked_counts: list[tuple[int, str, int]],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = [rank for rank, _, _ in ranked_counts]
    counts = [count for _, _, count in ranked_counts]

    plt.figure(figsize=(9, 6), dpi=150)
    plt.plot(
        ranks, counts, marker=".", linestyle="none", markersize=4,
        color="#2b6cb0", label="tokens",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Log rank")
    plt.ylabel("Log frequency")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
