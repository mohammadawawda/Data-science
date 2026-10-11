"""
books_crawler.py — bookdelivery.com crawler.

Design goals
------------
* Crawl is resumable.  Every successfully-parsed book is appended to
  output/crawl_log.jsonl right away (one JSON per line).  If the script
  is interrupted, restarting it simply skips URLs already in the log.

* Output build is decoupled from the crawl.  build_outputs() reads the
  log and produces every required CSV/JSON, so the final files are
  always reproducible from the log alone.

* main() runs build_outputs() in a `finally` clause, so even a crash
  during the crawl still leaves you with usable output files.

Usage
-----
    pip install pandas beautifulsoup4 lxml requests undetected-chromedriver
    cd code && python books_crawler.py            # full crawl + build
    cd code && python books_crawler.py rebuild    # rebuild outputs from log

Notes
-----
* The site is behind Cloudflare, so we open Chrome once via
  undetected-chromedriver, lift the cookies into a requests.Session, and
  use plain HTTP from then on (much faster than driving Chrome per page).
* 3-second polite delay between requests.  ~4000 books ≈ 3-4 hours.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://www.bookdelivery.com"
START_URL = BASE_URL + "/il-en/"
EXCHANGE_RATE = 3.01
MAX_PAGES_PER_CATEGORY = 5
DELAY = 3                  # seconds between every request (politeness)
RETRIES = 3

REQUIRED_FIELDS = [
    "id", "url", "Title", "Category", "Categories", "Authors",
    "Price in NIS", "Price in USD", "Year", "Synopsis", "Synopsis length",
    "StarRating", "NumberOfReviews", "Language", "Format",
    "Dimensions", "Dimensions unit", "Weight", "Weight unit", "ISBN/ISBN13",
]

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUTPUT_DIR / "crawl_log.jsonl"

ERROR_TITLES = {
    "500 Internal Server Error",
    "502 Bad Gateway",
    "503 Service Temporarily Unavailable",
    "504 Gateway Time-out",
}

# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
_driver = None
_session = None


def _ensure_distutils():
    """Python 3.12+ removed distutils; undetected-chromedriver still imports it.
    Patch in setuptools' vendored copy before the offending import happens."""
    try:
        import distutils.version  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    try:
        import setuptools._distutils as _d
        import setuptools._distutils.version as _dv
    except ImportError as exc:
        raise SystemExit(
            "Missing distutils (Python 3.12+ removed it).\n"
            "Run:  pip install --upgrade setuptools\n"
            f"({exc})"
        )
    sys.modules["distutils"] = _d
    sys.modules["distutils.version"] = _dv


def _detect_chrome_major_version():
    """Best-effort detection of installed Chrome's major version. Returns int or None."""
    import subprocess
    candidates = []
    if sys.platform == "darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    elif sys.platform.startswith("linux"):
        candidates = ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]
    elif sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
    for path in candidates:
        try:
            out = subprocess.check_output(
                [path, "--version"], stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            m = re.search(r"(\d+)\.\d+\.\d+", out)
            if m:
                return int(m.group(1))
        except (subprocess.SubprocessError, FileNotFoundError, OSError, PermissionError):
            continue
    return None


def _get_driver():
    """Lazy-init undetected Chrome (only used to bootstrap cookies + screenshot)."""
    global _driver
    if _driver is None:
        _ensure_distutils()
        import undetected_chromedriver as uc
        # Avoid a noisy del-time error from undetected_chromedriver
        uc.Chrome.__del__ = lambda self: None
        kwargs = {"headless": False}
        major = _detect_chrome_major_version()
        if major:
            print(f"  [bootstrap] detected Chrome {major}, pinning chromedriver to match", flush=True)
            kwargs["version_main"] = major
        else:
            print("  [bootstrap] could not detect Chrome version, will use latest chromedriver", flush=True)
        _driver = uc.Chrome(**kwargs)
    return _driver


def _bootstrap_session():
    """Open homepage in Chrome once, copy cookies into a requests.Session."""
    global _session
    if _session is not None:
        return _session
    import requests
    drv = _get_driver()
    drv.get(START_URL)
    time.sleep(8)  # give Cloudflare a moment to clear
    s = requests.Session()
    s.headers.update({
        "User-Agent": drv.execute_script("return navigator.userAgent"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    for c in drv.get_cookies():
        s.cookies.set(c["name"], c["value"], domain=c.get("domain"))
    _session = s
    return s


def _is_error(soup):
    if soup is None:
        return True
    title = soup.title.get_text(strip=True) if soup.title else ""
    return title in ERROR_TITLES


def _is_cloudflare_challenge(soup):
    """Detect Cloudflare interstitial / 'JavaScript is disabled' pages."""
    if soup is None:
        return False
    text = soup.get_text(" ", strip=True)[:3000].lower()
    markers = (
        "just a moment",
        "checking your browser",
        "javascript is disabled",
        "enable javascript and cookies",
        "cf-chl-",
        "cloudflare ray id",
    )
    return any(m in text for m in markers)


def _refresh_session_via_browser(target_url=None):
    """Re-bootstrap the requests session by driving Chrome through the URL."""
    global _session
    import requests
    try:
        drv = _get_driver()
        drv.get(target_url or START_URL)
        # Wait up to 30s for CF to clear
        for _ in range(15):
            time.sleep(2)
            html = drv.page_source[:3000].lower()
            if "just a moment" not in html and "javascript is disabled" not in html and "checking your browser" not in html:
                break
        s = requests.Session()
        s.headers.update({
            "User-Agent": drv.execute_script("return navigator.userAgent"),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        for c in drv.get_cookies():
            s.cookies.set(c["name"], c["value"], domain=c.get("domain"))
        _session = s
        print("  [session] refreshed via browser", flush=True)
        return True
    except Exception as e:
        print(f"  [session] refresh failed: {e}", flush=True)
        return False


def fetch(url):
    """GET url with retries; refresh session via browser on Cloudflare challenges."""
    last = None
    cf_count = 0
    for attempt in range(1, RETRIES + 1):
        try:
            s = _bootstrap_session()
            r = s.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            if _is_error(soup):
                raise ValueError("server error page")
            if _is_cloudflare_challenge(soup):
                cf_count += 1
                print(f"  [CF] challenge on {url} (attempt {attempt}); refreshing session", flush=True)
                _refresh_session_via_browser(url)
                raise ValueError("cloudflare challenge")
            return soup
        except Exception as e:
            last = e
            time.sleep(2 * attempt)
    print(f"  FAIL {url}: {last}")
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def discover_categories(soup):
    """Read the inyectaSubCategorias JSON from the homepage scripts."""
    cats = []
    for tag in soup.find_all("script", type="text/javascript"):
        if not tag.string or "inyectaSubCategorias" not in tag.string:
            continue
        m = re.search(r"JSON\.parse\('(\[.*?\])'\)", tag.string)
        if not m:
            continue
        try:
            data = json.loads(m.group(1).replace("\\'", "'"))
        except Exception:
            continue
        seen = set()
        for c in data:
            href = c.get("url") or ""
            if not href:
                continue
            full = href if href.startswith("http") else BASE_URL + href
            if full in seen:
                continue
            seen.add(full)
            cats.append({"name": c.get("title", "?"), "url": full})
    return cats


def get_pagination(soup, base_url):
    urls = [base_url]
    pagn = soup.find("div", id="pagn")
    if pagn:
        seen = {base_url}
        for a in pagn.find_all("a", href=True):
            full = a["href"] if a["href"].startswith("http") else BASE_URL + a["href"]
            m = re.search(r"page=(\d+)", full)
            if m and int(m.group(1)) <= MAX_PAGES_PER_CATEGORY and full not in seen:
                seen.add(full)
                urls.append(full)
    return urls[:MAX_PAGES_PER_CATEGORY]


def get_book_links(soup):
    items = soup.find_all("div", class_="box-producto") or soup.find_all("div", class_="producto")
    out = []
    for it in items:
        a = it.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        if not any(s in href for s in ("/p/", "/book-", "/libro-")):
            continue
        full = href if href.startswith("http") else BASE_URL + href
        out.append(full)
    return list(dict.fromkeys(out))  # dedupe, preserve order


def parse_book(soup, category, url):
    if soup is None:
        return None
    out = {"url": url, "Category": category}

    # JSON-LD Product (most stable source)
    ld = {}
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
        except Exception:
            continue
        for c in (data if isinstance(data, list) else [data]):
            if isinstance(c, dict) and c.get("@type") == "Product":
                ld = c
                break
        if ld:
            break

    # Sidebar info table
    info = {}
    ficha = soup.find("div", class_="ficha") or soup.find("section", id="producto")
    if ficha:
        for row in ficha.find_all("div", class_="row"):
            label = row.find("div", class_="col-xs-5")
            value = row.find("div", class_="col-xs-7")
            if label and value:
                k = label.get_text(strip=True).rstrip(":")
                v = value.get_text(strip=True)
                if k and v:
                    info[k] = v

    # Title (no truncation!)
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ld.get("name", "")
    if title:
        out["Title"] = title

    # Authors
    abox = soup.find("div", id="metadata-autor")
    if abox:
        names = [
            a.get_text(strip=True)
            for a in abox.find_all("a")
            if a.get_text(strip=True) and "centro-de-autores" not in a.get("href", "")
        ]
        out["Authors"] = ", ".join(names) if names else abox.get_text(strip=True)
    elif ld.get("author"):
        author = ld["author"]
        if isinstance(author, list):
            out["Authors"] = ", ".join(a.get("name", "") for a in author)
        elif isinstance(author, dict):
            out["Authors"] = author.get("name", "")
        else:
            out["Authors"] = str(author)

    # Categories — comma-separated, from breadcrumb
    bc = soup.find("ol", class_="breadcrumb")
    if bc:
        crumbs = [a.get_text(strip=True) for a in bc.find_all("a") if a.get_text(strip=True)]
        if crumbs:
            out["Categories"] = ", ".join(crumbs)
    out.setdefault("Categories", f"{category}, Books")

    # Price (NIS + USD)
    price_tag = (
        soup.find("strong", class_="precio")
        or soup.find("div", class_="precioAhora")
        or soup.find("div", class_="precio")
    )
    if price_tag:
        m = re.search(r"\d{1,3}(,\d{3})*(\.\d+)?", price_tag.get_text(strip=True))
        if m:
            nis = float(m.group().replace(",", ""))
            out["Price in NIS"] = math.ceil(nis * 100) / 100.0
            out["Price in USD"] = math.ceil((nis / EXCHANGE_RATE) * 100) / 100.0

    # Year
    ym = re.search(r"\b(19|20)\d{2}\b", info.get("Year", ""))
    if not ym:
        ybox = soup.find("div", id="metadata-ano")
        if ybox:
            ym = re.search(r"\b(19|20)\d{2}\b", ybox.get_text())
    if ym:
        out["Year"] = int(ym.group())

    # Synopsis
    synopsis = None
    div = soup.find("div", class_="descripcionBreve")
    if div:
        sp = div.find("span", id="texto-descripcion")
        if sp:
            synopsis = sp.get_text(" ", strip=True)
        else:
            for x in div.find_all(["style", "script", "button", "h1", "h2", "h3", "h4"]):
                x.extract()
            for x in div.find_all(class_=re.compile(r"(translate|colaborador|author)", re.I)):
                x.extract()
            synopsis = " ".join(div.stripped_strings)
    if not synopsis:
        synopsis = ld.get("description")
    if synopsis:
        out["Synopsis"] = synopsis
        out["Synopsis length"] = len(synopsis)

    # NumberOfReviews + StarRating ("None" string when no reviews)
    revs = ld.get("review", [])
    if isinstance(revs, dict):
        revs = [revs]
    elif not isinstance(revs, list):
        revs = []
    out["NumberOfReviews"] = len(revs)
    if revs:
        scores = []
        for r in revs:
            v = r.get("reviewRating", {}).get("ratingValue")
            try:
                scores.append(float(v))
            except (TypeError, ValueError):
                continue
        if scores:
            out["StarRating"] = math.ceil((sum(scores) / len(scores)) * 100) / 100.0
        else:
            out["StarRating"] = "None"
    else:
        agg = ld.get("aggregateRating", {}) or {}
        try:
            cnt = int(agg.get("reviewCount", 0))
        except (TypeError, ValueError):
            cnt = 0
        rv = agg.get("ratingValue")
        if cnt > 0 and rv is not None:
            try:
                out["StarRating"] = math.ceil(float(rv) * 100) / 100.0
                out["NumberOfReviews"] = cnt
            except (TypeError, ValueError):
                out["StarRating"] = "None"
        else:
            out["StarRating"] = "None"

    # Language
    if "Language" in info:
        out["Language"] = info["Language"]
    else:
        b = soup.find("div", id="metadata-idioma")
        if b:
            out["Language"] = b.get_text(strip=True)

    # Format
    if "Format" in info:
        out["Format"] = info["Format"]
    else:
        for fid in ("metadata-encuadernación", "metadata-encuadernacion"):
            b = soup.find("div", id=fid)
            if b:
                out["Format"] = b.get_text(strip=True)
                break

    # Dimensions
    dim_text = info.get("Dimensions")
    if not dim_text:
        b = soup.find("div", id="metadata-dimensiones")
        if b:
            dim_text = b.get_text(strip=True)
    if dim_text:
        nums = re.findall(r"[\d.]+", dim_text)
        if nums:
            out["Dimensions"] = ", ".join(nums)
            um = re.search(r"(cm|mm|inch|in)\b", dim_text, re.IGNORECASE)
            out["Dimensions unit"] = um.group() if um else "cm"

    # Weight
    weight_text = info.get("Weight")
    if not weight_text:
        b = soup.find("div", id="metadata-peso")
        if b:
            weight_text = b.get_text(strip=True)
    if weight_text:
        m = re.search(r"[\d.]+", weight_text)
        if m:
            out["Weight"] = float(m.group())
        um = re.search(r"(kg|g|lb|oz)\.?", weight_text, re.IGNORECASE)
        if um:
            out["Weight unit"] = um.group().rstrip(".")

    # ISBN / ISBN13
    isbn = info.get("ISBN13") or info.get("ISBN") or ld.get("isbn")
    if isbn:
        out["ISBN/ISBN13"] = isbn

    return out


# ---------------------------------------------------------------------------
# Persistence (JSON-lines log + resume)
# ---------------------------------------------------------------------------
def append_log(record):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, default=str)
        f.write("\n")


def load_log():
    if not LOG_PATH.exists():
        return []
    out = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def already_visited():
    return {r["url"] for r in load_log() if r.get("url")}


# ---------------------------------------------------------------------------
# Crawl driver
# ---------------------------------------------------------------------------
def crawl():
    print("=" * 60)
    print("books_crawler.py — clean restart edition (resumable + finally)")
    print("=" * 60)
    visited = already_visited()
    if visited:
        print(f"Resuming — {len(visited)} books already in log")

    print("Fetching homepage...")
    home = fetch(START_URL)
    if home is None:
        raise RuntimeError("Could not fetch homepage")
    cats = discover_categories(home)
    if not cats:
        raise RuntimeError("No categories found")
    print(f"Discovered {len(cats)} categories")

    screenshot_taken = (OUTPUT_DIR / "books_example.jpg").exists()

    for ci, cat in enumerate(cats, 1):
        print(f"\n=== [{ci}/{len(cats)}] {cat['name']} ===")
        time.sleep(DELAY)
        cat_soup = fetch(cat["url"])
        if cat_soup is None:
            continue

        pages = get_pagination(cat_soup, cat["url"])
        print(f"  pages: {len(pages)}")

        all_links = []
        for pi, purl in enumerate(pages):
            if pi == 0:
                soup = cat_soup
            else:
                time.sleep(DELAY)
                soup = fetch(purl)
            if soup is None:
                continue
            links = get_book_links(soup)
            print(f"    page {pi + 1}: {len(links)} books")
            all_links.extend(links)

        for url in dict.fromkeys(all_links):
            if url in visited:
                continue
            time.sleep(DELAY)
            book_soup = fetch(url)
            rec = parse_book(book_soup, cat["name"], url)
            if rec is None:
                continue
            append_log(rec)
            visited.add(url)
            print(f"    + {rec.get('Title', '?')[:70]}")

            if not screenshot_taken:
                try:
                    drv = _get_driver()
                    drv.get(url)
                    time.sleep(3)
                    drv.save_screenshot(str(OUTPUT_DIR / "books_example.jpg"))
                    screenshot_taken = True
                    print("    ↳ saved books_example.jpg")
                except Exception as e:
                    print(f"    screenshot failed: {e}")


# ---------------------------------------------------------------------------
# Output construction (Steps 2–5)
# ---------------------------------------------------------------------------
def _is_blank(v):
    if v is None or v == "" or v == "N/A":
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def save_json(df, path):
    """{ "records": { "record": [ ... ] } }, with empty fields stripped."""
    recs = []
    for r in df.to_dict(orient="records"):
        clean = {}
        for k, v in r.items():
            if k == "StarRating" and v == "None":
                clean[k] = v
                continue
            if _is_blank(v):
                continue
            clean[k] = v
        recs.append(clean)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"records": {"record": recs}}, f, indent=4, ensure_ascii=False, default=str)


def build_outputs():
    records = load_log()
    if not records:
        print("Log is empty — nothing to build.")
        return

    # Drop server-error rows + Cloudflare interstitials that look like books
    JUNK_TITLE_PHRASES = (
        "javascript is disabled",
        "just a moment",
        "checking your browser",
        "cloudflare",
        "ray id",
        "attention required",
    )

    def _is_junk(rec):
        t = str(rec.get("Title", "")).strip()
        if not t:
            return True
        if t in ERROR_TITLES:
            return True
        if t.startswith(("500 ", "502 ", "503 ", "504 ")):
            return True
        tl = t.lower()
        if any(p in tl for p in JUNK_TITLE_PHRASES):
            return True
        # A real book should at least have a price OR an ISBN
        if not rec.get("Price in NIS") and not rec.get("ISBN/ISBN13"):
            return True
        return False

    n_before = len(records)
    clean = [r for r in records if not _is_junk(r)]
    n_dropped = n_before - len(clean)
    if n_dropped:
        print(f"Filtered {n_dropped} junk/error records ({n_before} → {len(clean)})")

    df = pd.DataFrame(clean)

    # Make sure every required column exists (NaN if missing)
    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            df[col] = pd.NA

    # Numeric casts
    for col in ["Price in NIS", "Price in USD", "Synopsis length",
                "Weight", "NumberOfReviews", "Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reset id, fix column order
    df = df.reset_index(drop=True)
    df["id"] = (df.index + 1).astype(str)
    df = df[REQUIRED_FIELDS]

    # ---- Step 2: raw ----
    df.to_csv(OUTPUT_DIR / "books_raw.csv", index=False)
    save_json(df, OUTPUT_DIR / "books_raw.json")
    save_json(df.head(1), OUTPUT_DIR / "books_example.json")

    # ---- Step 3: preview / sort ----
    print("\n--- BEFORE sorting (first 10) ---")
    print(df.head(10).to_string())
    df.head(10).to_csv(OUTPUT_DIR / "books_before_sort.csv", index=False)

    df_sorted = df.sort_values("Title", ascending=True, na_position="last").reset_index(drop=True)
    print("\n--- AFTER sorting by Title (first 10) ---")
    print(df_sorted.head(10).to_string())
    df_sorted.head(10).to_csv(OUTPUT_DIR / "books_after_sort.csv", index=False)

    # ---- Step 4: derived features ----
    median_nis = df["Price in NIS"].median()
    df["IsExpensive"] = (df["Price in NIS"] > median_nis).astype("Int64")

    def n_authors(s):
        if pd.isna(s):
            return 0
        s = str(s).strip()
        if s in ("", "N/A", "None"):
            return 0
        return len([a for a in s.split(",") if a.strip()])

    df["NumberOfAuthors"] = df["Authors"].apply(n_authors)

    df.to_csv(OUTPUT_DIR / "books_processed.csv", index=False)
    save_json(df, OUTPUT_DIR / "books_processed.json")
    df.head(10).to_csv(OUTPUT_DIR / "books_processed_preview.csv", index=False)
    print("\n--- AFTER processing (first 10) ---")
    print(df.head(10).to_string())

    # ---- Step 5: summary stats (only non-missing values) ----
    summary_cols = ["Price in USD", "Year", "StarRating",
                    "NumberOfReviews", "NumberOfAuthors"]
    rows = {}
    for col in summary_cols:
        s = pd.to_numeric(df[col], errors="coerce") if col == "StarRating" \
            else pd.to_numeric(df[col], errors="coerce")
        rows[col] = {
            "mean":   s.mean(),
            "std":    s.std(),
            "min":    s.min(),
            "max":    s.max(),
            "median": s.median(),
        }
    summary_df = pd.DataFrame(rows).T
    summary_df["total_rows"] = len(df)
    summary_df.to_csv(OUTPUT_DIR / "books_summary.csv")
    print("\n--- Summary ---")
    print(summary_df.to_string())

    print(f"\nDone. {len(df)} books → {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    try:
        crawl()
    except KeyboardInterrupt:
        print("\nInterrupted — building outputs from what's in the log so far...")
    finally:
        try:
            build_outputs()
        finally:
            global _driver
            if _driver is not None:
                try:
                    _driver.quit()
                except Exception:
                    pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        # Re-build CSV/JSON outputs from the existing crawl_log.jsonl
        # (no network calls, no Chrome — useful after a crawl is complete)
        build_outputs()
    else:
        main()
