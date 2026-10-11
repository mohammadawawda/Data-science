"""build_report.py — generate consolidated_report.pdf from the output/ files.

Run this AFTER books_crawler.py finishes:
    cd ~/needle_ex1/code
    python build_report.py

Produces:
    output/consolidated_report.pdf

Also fixes the books_example.jpg bytes (Selenium saves PNG with .jpg name).

Edit the GROUP_MEMBERS list below before running to change the author.
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================================
# EDIT THESE BEFORE RUNNING
# ============================================================================
GROUP_MEMBERS = [
    {"name": "Mohammad Awawda"},
]

EXCHANGE_RATE = 3.01
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
REPORT_PATH = OUTPUT_DIR / "consolidated_report.pdf"

# Columns to show in tables — full schema is too wide; pick the meaningful ones
TABLE_COLS = [
    "id", "Title", "Category", "Authors",
    "Price in NIS", "Price in USD", "Year",
    "StarRating", "NumberOfReviews", "Synopsis length",
]
TABLE_COLS_PROCESSED = TABLE_COLS + ["IsExpensive", "NumberOfAuthors"]

# Visible width budget per column for the wide tables
TRUNCATE = {
    "Title": 35,
    "Category": 22,
    "Authors": 22,
}


def fix_jpg_extension():
    """Selenium saves PNG bytes with a .jpg name. Convert to real JPEG."""
    jpg = OUTPUT_DIR / "books_example.jpg"
    if not jpg.exists():
        return
    try:
        from PIL import Image as PILImage
        with open(jpg, "rb") as f:
            head = f.read(8)
        if head.startswith(b"\x89PNG"):
            print("  Re-encoding books_example.jpg as real JPEG…")
            img = PILImage.open(jpg).convert("RGB")
            img.save(jpg, "JPEG", quality=90)
    except ImportError:
        print("  (Pillow not installed — leaving books_example.jpg as-is. "
              "It opens fine; just technically PNG bytes.)")


def shorten(value, limit):
    s = "" if pd.isna(value) else str(value)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def df_to_table_data(df: pd.DataFrame, cols):
    headers = cols
    rows = [headers]
    for _, r in df[cols].iterrows():
        row = []
        for c in cols:
            v = r[c]
            if c in TRUNCATE:
                row.append(shorten(v, TRUNCATE[c]))
            elif c in ("Price in NIS", "Price in USD"):
                row.append("" if pd.isna(v) else f"{float(v):.2f}")
            elif c == "Year":
                row.append("" if pd.isna(v) else f"{int(float(v))}")
            elif pd.isna(v):
                row.append("")
            else:
                row.append(str(v))
        rows.append(row)
    return rows


def make_table(data, repeat_header=True):
    t = Table(data, repeatRows=1 if repeat_header else 0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 7.5),
        ("FONTSIZE",   (0, 1), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f2f7")]),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 1), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 2),
    ]))
    return t


def build():
    if not (OUTPUT_DIR / "books_raw.csv").exists():
        raise SystemExit(f"Missing {OUTPUT_DIR/'books_raw.csv'} — run books_crawler.py first.")

    fix_jpg_extension()

    df_raw   = pd.read_csv(OUTPUT_DIR / "books_raw.csv")
    df_proc  = pd.read_csv(OUTPUT_DIR / "books_processed.csv")
    summary  = pd.read_csv(OUTPUT_DIR / "books_summary.csv", index_col=0)

    df_before = df_raw.head(10)
    df_after  = df_raw.sort_values("Title", ascending=True, na_position="last").head(10)
    df_after_proc = df_proc.head(10)

    n_books    = len(df_raw)
    n_cats     = df_raw["Category"].nunique()
    median_nis = df_raw["Price in NIS"].median()

    # --- Document setup ---
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=landscape(A4),
        leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="Books Crawler - Consolidated Report",
        author=", ".join(m["name"] for m in GROUP_MEMBERS),
    )

    base = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=base["Heading1"], fontSize=20,
                        textColor=colors.HexColor("#1f3864"), spaceAfter=12)
    H2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#1f3864"), spaceAfter=6,
                        spaceBefore=10)
    P  = ParagraphStyle("P",  parent=base["BodyText"], fontSize=10, leading=14)
    SMALL = ParagraphStyle("SMALL", parent=base["BodyText"], fontSize=8.5, leading=11)
    CAPTION = ParagraphStyle("CAPTION", parent=base["BodyText"], fontSize=9,
                             leading=11, textColor=colors.HexColor("#666666"),
                             alignment=1)

    story = []

    # ---------------- Cover ----------------
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph("Books Crawler", H1))
    story.append(Paragraph("Crawling, cleaning and summarizing bookdelivery.com", P))
    story.append(Paragraph("Consolidated Report", P))
    story.append(Spacer(1, 14 * mm))

    story.append(Paragraph("Author", H2))
    member_rows = [["#", "Name"]]
    for i, m in enumerate(GROUP_MEMBERS, 1):
        member_rows.append([str(i), m["name"]])
    member_tbl = Table(member_rows, colWidths=[15 * mm, 80 * mm])
    member_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f2f7")]),
    ]))
    story.append(member_tbl)
    story.append(Spacer(1, 10 * mm))

    story.append(PageBreak())

    # ---------------- Crawl summary ----------------
    story.append(Paragraph("1. Crawl Summary", H1))
    summary_rows = [
        ["Source site",          "https://www.bookdelivery.com/"],
        ["Top-level categories crawled", f"{n_cats} (target: all top-level categories)"],
        ["Pages per category (max)", "5"],
        ["Total books collected", f"{n_books:,}"],
        ["Median Price in NIS (used for IsExpensive split)", f"{median_nis:.2f}"],
        ["NIS → USD exchange rate", f"{EXCHANGE_RATE} (fixed)"],
        ["Polite delay between requests", "3 seconds"],
        ["Retry policy",          "3 attempts with exponential backoff"],
        ["Tools",                 "Python, requests, BeautifulSoup, "
                                  "undetected-chromedriver (for Cloudflare bootstrap), pandas"],
    ]
    t = Table(summary_rows, colWidths=[80 * mm, 175 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.white, colors.HexColor("#f2f2f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    story.append(Paragraph("Pipeline (Steps 1–5)", H2))
    pipeline = (
        "<b>Step&nbsp;1 (Crawl).</b> Discover all category links from the homepage "
        "programmatically by parsing the <i>inyectaSubCategorias</i> JSON in the "
        "homepage script. For each category, follow up to 5 pagination pages, extract "
        "individual book URLs, and visit each one. Per book extract Title, Categories, "
        "Authors, Price (NIS &amp; USD with math.ceil rounding to 2 decimals), Year, "
        "Synopsis, NumberOfReviews, StarRating (\"None\" for zero reviews), Language, "
        "Format, Dimensions, Weight and ISBN/ISBN13.<br/>"
        "<b>Step&nbsp;2 (DataFrame).</b> Load every record into <i>df_books</i>, cast "
        "numeric fields, write <i>books_raw.csv</i>, <i>books_raw.json</i>, plus a "
        "single-record <i>books_example.json</i> alongside <i>books_example.jpg</i>.<br/>"
        "<b>Step&nbsp;3 (Preview / Sort).</b> Save the first 10 rows before sorting, "
        "then again after sorting by Title ascending.<br/>"
        "<b>Step&nbsp;4 (Derived features).</b> Add <i>IsExpensive</i> (price &gt; "
        "median NIS) and <i>NumberOfAuthors</i> (comma split). Save the processed "
        "DataFrame and an updated JSON.<br/>"
        "<b>Step&nbsp;5 (Summary).</b> Compute mean / std / min / max / median for "
        "Price&nbsp;in&nbsp;USD, Year, StarRating, NumberOfReviews and "
        "NumberOfAuthors over non-missing values, plus total row count."
    )
    story.append(Paragraph(pipeline, P))
    story.append(PageBreak())

    # ---------------- Tables ----------------
    note = ("Showing the most informative columns. The full 20-column schema "
            "(including Synopsis text, Language, Format, Dimensions, Weight, "
            "ISBN/ISBN13 and url) is preserved in the CSV/JSON files in output/.")

    story.append(Paragraph("2. First 10 rows — before sorting", H1))
    story.append(Paragraph(note, SMALL))
    story.append(Spacer(1, 4 * mm))
    story.append(make_table(df_to_table_data(df_before, TABLE_COLS)))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Source: <i>output/books_before_sort.csv</i>", SMALL))
    story.append(PageBreak())

    story.append(Paragraph("3. First 10 rows — after sorting by Title", H1))
    story.append(Paragraph(note, SMALL))
    story.append(Spacer(1, 4 * mm))
    story.append(make_table(df_to_table_data(df_after, TABLE_COLS)))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Source: <i>output/books_after_sort.csv</i>", SMALL))
    story.append(PageBreak())

    story.append(Paragraph("4. First 10 rows — after processing", H1))
    story.append(Paragraph(
        "Now includes the derived <b>IsExpensive</b> and <b>NumberOfAuthors</b> columns.",
        SMALL))
    story.append(Spacer(1, 4 * mm))
    story.append(make_table(df_to_table_data(df_after_proc, TABLE_COLS_PROCESSED)))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Source: <i>output/books_processed_preview.csv</i>", SMALL))
    story.append(PageBreak())

    # ---------------- Summary statistics ----------------
    story.append(Paragraph("5. Summary statistics", H1))
    story.append(Paragraph(
        "Computed only over books where the column has a value. "
        "<i>total_rows</i> is the row count of <i>books_processed.csv</i>.",
        SMALL))
    story.append(Spacer(1, 4 * mm))

    sum_data = [["column", "mean", "std", "min", "max", "median", "total_rows"]]
    for col, row in summary.iterrows():
        sum_data.append([
            col,
            f"{row['mean']:.2f}",
            f"{row['std']:.2f}",
            f"{row['min']:.2f}",
            f"{row['max']:.2f}",
            f"{row['median']:.2f}",
            f"{int(row['total_rows'])}",
        ])
    story.append(make_table(sum_data))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Source: <i>output/books_summary.csv</i>", SMALL))
    story.append(PageBreak())

    # ---------------- Files ----------------
    story.append(Paragraph("6. Output files", H1))
    files = [
        ("books_raw.csv",                "Raw DataFrame, all 20 columns (Step 2)"),
        ("books_raw.json",               "Same content as JSON, empty fields stripped (Step 2)"),
        ("books_example.json",           "Single example record (Step 2)"),
        ("books_example.jpg",            "Screenshot of one of the crawled pages (Step 2)"),
        ("books_before_sort.csv",        "First 10 rows before sorting (Step 3)"),
        ("books_after_sort.csv",         "First 10 rows after sort by Title ascending (Step 3)"),
        ("books_processed.csv",          "Full data + IsExpensive + NumberOfAuthors (Step 4)"),
        ("books_processed.json",         "Same content as JSON (Step 4)"),
        ("books_processed_preview.csv",  "First 10 rows after processing (Step 4)"),
        ("books_summary.csv",            "Summary statistics over the 5 columns (Step 5)"),
        ("consolidated_report.pdf",      "This report"),
    ]
    file_rows = [["File (under output/)", "Contents"]] + list(files)
    t = Table(file_rows, colWidths=[80 * mm, 175 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f2f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ---------------- Screenshot ----------------
    story.append(Paragraph("7. Crawled-page screenshot (Step 2)", H1))
    jpg = OUTPUT_DIR / "books_example.jpg"
    if jpg.exists():
        # landscape A4 minus margins ≈ 273 × 186 mm
        story.append(Image(str(jpg), width=255 * mm, height=140 * mm,
                           kind="proportional"))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "Screenshot of the first book page crawled. "
            "Source: <i>output/books_example.jpg</i>", CAPTION))
    else:
        story.append(Paragraph("books_example.jpg not found in output/.", P))

    doc.build(story)
    print(f"  Wrote {REPORT_PATH}")


if __name__ == "__main__":
    build()
