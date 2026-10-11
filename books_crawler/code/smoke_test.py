"""smoke_test.py — parametric end-to-end test before the full crawl.

Crawls N_CATEGORIES × N_PAGES × N_BOOKS_PER_PAGE, builds outputs, verifies them.
Output dir: output_smoke/  (separate from real output/).

Usage
-----
    python smoke_test.py                      # ~1 min:  1 cat × 1 page × 3 books
    python smoke_test.py 5 3 7                # ~9 min:  5 cats × 3 pages × 7 books
    python smoke_test.py 10 2 5               # ~9 min: 10 cats × 2 pages × 5 books

Stops early on Ctrl-C; outputs are still built from whatever was crawled.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import books_crawler as bc  # noqa: E402

# ---- parse CLI args (all optional) ----
def _arg(idx, default):
    try:
        return int(sys.argv[idx])
    except (IndexError, ValueError):
        return default

N_CATEGORIES       = _arg(1, 1)
N_PAGES            = _arg(2, 1)
N_BOOKS_PER_PAGE   = _arg(3, 3)

# ---- redirect bc's output to output_smoke/ ----
SMOKE_DIR = bc.PROJECT_DIR / "output_smoke"
SMOKE_DIR.mkdir(exist_ok=True)
bc.OUTPUT_DIR = SMOKE_DIR
bc.LOG_PATH = SMOKE_DIR / "crawl_log.jsonl"
bc.MAX_PAGES_PER_CATEGORY = N_PAGES


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def estimate_seconds():
    """Rough time budget: 30s bootstrap + per-cat (cat page + extra pages + books × delay)."""
    bootstrap = 30
    per_book = bc.DELAY + 1                 # delay + ~1s parse
    per_page = bc.DELAY + 1
    per_cat = per_page + (N_PAGES - 1) * per_page + (N_PAGES * N_BOOKS_PER_PAGE) * per_book
    return bootstrap + N_CATEGORIES * per_cat


def smoke_crawl():
    est = estimate_seconds()
    banner(f"TEST RUN — {N_CATEGORIES} cats × {N_PAGES} pages × {N_BOOKS_PER_PAGE} books/page")
    print(f"  Total books target: ~{N_CATEGORIES * N_PAGES * N_BOOKS_PER_PAGE}")
    print(f"  Estimated time:     ~{est // 60}m {est % 60}s")
    print(f"  Output:             {SMOKE_DIR}")
    print(f"  Press Ctrl-C anytime — outputs still build from what's collected.")

    if bc.LOG_PATH.exists():
        bc.LOG_PATH.unlink()

    print("\n[1] Fetching homepage (Chrome opens; first time may take 30s)...")
    home = bc.fetch(bc.START_URL)
    if home is None:
        raise RuntimeError("Could not fetch homepage")
    cats = bc.discover_categories(home)
    if not cats:
        raise RuntimeError("No categories found")
    print(f"    ✓ Discovered {len(cats)} categories — using first {N_CATEGORIES}")

    started = time.time()
    book_count = 0

    for ci, cat in enumerate(cats[:N_CATEGORIES], 1):
        print(f"\n[{ci}/{N_CATEGORIES}] Category: {cat['name']}")
        time.sleep(bc.DELAY)
        cat_soup = bc.fetch(cat["url"])
        if cat_soup is None:
            print("    ✗ category page failed")
            continue

        # Pagination
        page_urls = bc.get_pagination(cat_soup, cat["url"])[:N_PAGES]
        print(f"    pages: {len(page_urls)}")

        for pi, purl in enumerate(page_urls):
            soup = cat_soup if pi == 0 else (time.sleep(bc.DELAY) or bc.fetch(purl))
            if soup is None:
                continue
            links = bc.get_book_links(soup)[:N_BOOKS_PER_PAGE]
            print(f"      page {pi + 1}: visiting {len(links)} books")
            for url in links:
                time.sleep(bc.DELAY)
                book_soup = bc.fetch(url)
                rec = bc.parse_book(book_soup, cat["name"], url)
                if rec is None:
                    print(f"        ✗ failed: {url}")
                    continue
                bc.append_log(rec)
                book_count += 1
                elapsed = time.time() - started
                print(f"        + [{book_count}] {rec.get('Title', '?')[:60]}   ({elapsed:.0f}s)")


def verify():
    import pandas as pd
    banner("VERIFICATION")

    csv_path = SMOKE_DIR / "books_raw.csv"
    if not csv_path.exists():
        print(f"  ✗ {csv_path} not produced — pipeline failed")
        return False

    df = pd.read_csv(csv_path)
    print(f"  rows × cols: {len(df)} × {len(df.columns)}")

    missing = [c for c in bc.REQUIRED_FIELDS if c not in df.columns]
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        return False
    print(f"  ✓ All {len(bc.REQUIRED_FIELDS)} required columns present")

    if len(df) == 0:
        print("  ✗ Zero rows in output")
        return False

    max_title = int(df["Title"].astype(str).str.len().max())
    print(f"  {'✓' if max_title > 50 else '⚠'} Max title length: {max_title} chars")

    # How many distinct categories made it in
    cats_seen = df["Category"].nunique()
    print(f"  ✓ Distinct categories crawled: {cats_seen}")

    print("\n  Field coverage (non-null count out of total rows):")
    for col in ["Title", "Authors", "Synopsis", "Language", "Format",
                "Dimensions", "Weight", "ISBN/ISBN13", "Categories",
                "Price in NIS", "Year", "StarRating"]:
        n = df[col].notna().sum()
        pct = (n / len(df)) * 100
        mark = "✓" if n > 0 else "○"
        print(f"    {mark} {col:<18} {n:>4}/{len(df)}   ({pct:5.1f}%)")

    print(f"\n  Files in {SMOKE_DIR.name}/:")
    for f in sorted(SMOKE_DIR.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    return True


def main():
    ok = False
    try:
        smoke_crawl()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted — building outputs from what was collected so far...")
    except Exception as e:
        print(f"\n✗ Crawl failed: {e}")
    finally:
        try:
            bc.build_outputs()
        except Exception as e:
            print(f"\n✗ build_outputs failed: {e}")
        ok = verify()
        if bc._driver is not None:
            try:
                bc._driver.quit()
            except Exception:
                pass

    banner("RESULT: PASS — ready for the full crawl"
           if ok else "RESULT: FAIL — fix above before running the full crawl")
    if ok:
        print("\nFor the full crawl:")
        print("  rm -rf output/")
        print("  python books_crawler.py")


if __name__ == "__main__":
    main()
