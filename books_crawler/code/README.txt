bookdelivery.com crawler
========================

Author
------
Mohammad Awawda


Files in this folder (code/)
----------------------------
  books_crawler.py    Main crawler (Steps 1–5)
  build_report.py     Generates output/consolidated_report.pdf
  smoke_test.py       Optional — small parametric test run for sanity checks
  README.txt          This file


How to reproduce the outputs
----------------------------
1) Install dependencies (Python 3.10+, Chrome installed locally):

       pip install pandas beautifulsoup4 lxml requests \
                   undetected-chromedriver setuptools certifi \
                   reportlab Pillow

   On macOS/Python from python.org, also run once to fix SSL certs:
       /Applications/Python\ 3.13/Install\ Certificates.command

2) Run the crawler (~3-4 hours for the full ~4,000 books):

       python books_crawler.py

   * 3-second delay between requests (politeness).
   * Resumable: every successful book is appended to
     output/crawl_log.jsonl immediately. Re-running skips URLs already
     in the log.
   * Cloudflare-aware: detects challenge pages and refreshes the session
     by driving Chrome through the URL.
   * Output build runs in a `finally` block so any interruption still
     produces CSV/JSON files from whatever was collected.

   To rebuild outputs from an existing log without re-crawling:
       python books_crawler.py rebuild

3) Generate the consolidated report:

       python build_report.py

   (Edit GROUP_MEMBERS at the top of build_report.py to change the author.)


Output files (under output/)
----------------------------
  crawl_log.jsonl              Append-only raw log (not in the repo)
  books_raw.csv                Raw DataFrame, all 20 columns          (Step 2)
  books_raw.json               Same as JSON, empty fields stripped    (Step 2)
  books_example.json           Single example record                  (Step 2)
  books_example.jpg            Screenshot of one crawled page         (Step 2)
  books_before_sort.csv        First 10 rows before sorting           (Step 3)
  books_after_sort.csv         First 10 rows after sort by Title      (Step 3)
  books_processed.csv          + IsExpensive, NumberOfAuthors         (Step 4)
  books_processed.json         Same as JSON                            (Step 4)
  books_processed_preview.csv  First 10 rows after processing         (Step 4)
  books_summary.csv            Summary statistics                     (Step 5)
  consolidated_report.pdf      Report PDF


Notes on design choices
-----------------------
* Cloudflare bootstrap. The site is behind Cloudflare, so we open the
  homepage once via undetected-chromedriver, copy the cookies into a
  requests.Session, and use plain HTTP from then on (much faster than
  driving Chrome per request).
* Field extraction. Per-book parsing prefers the JSON-LD Product block
  (most stable) and falls back to the sidebar info table and
  `metadata-*` divs in the HTML.
* StarRating. Computed by averaging individual review scores from the
  JSON-LD `review` array (rounded up to 2 decimals via math.ceil). When
  the page has zero reviews the literal string "None" is stored.
* Prices. Both NIS and USD use math.ceil(x*100)/100, so they are
  rounded up to 2 decimals. USD uses a fixed rate of 3.01.
* JSON. Records that did not have a particular field do not include
  that key in the JSON.
* Politeness. Single-threaded crawl with a 3-second delay between every
  category, pagination and book-page request, plus exponential backoff
  on transient errors (HTTP 5xx, network drops, Cloudflare challenges).
