# Books crawler

A crawler for bookdelivery.com. It goes over every category and page, collects about 4,000 books with 20 fields each, and then sorts, processes and summarizes them with pandas. It waits 3 seconds between requests, can continue after a stop, and gets past the site's Cloudflare check.

* `code/` the crawler, the report builder and a small smoke test. How to run is in `code/README.txt`
* `output/` sample outputs and the report

The full crawl files (books_raw and books_processed) are not included because they are big. Running the crawler makes them again.
