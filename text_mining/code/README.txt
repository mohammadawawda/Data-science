Text mining

how to run:
  1. Install the packages:
       pip install nltk wordcloud matplotlib
  2. Run the pipeline from inside the code/ folder:
       python text_mining_alice.py

  The script downloads the book once from Project Gutenberg and caches it in
  output/text/alice_raw.txt. Every later run reuses that cached file, so it also
  works with no internet. NLTK data (punkt, taggers, stopwords) is downloaded
  automatically on first use.

FILES:

  settings.py: paths, regex patterns, small config (no book-specific
               blacklists or hard-coded answers)
  io_utils.py: download/cache, Gutenberg cleaning, CSV and plot writers
  nlp_analysis.py: tokenizing, counts, stopwords, stemming, POS tagging,
		   proper nouns, and the repeated-word regex
  text_mining_alice.py: main script that ties it all together

what's produced in output/:
  figures/: the three log-log plots and the proper-noun word cloud (PNG)
  tables/: token counts, top-20 lists, proper-noun counts, repeated-word matches (CSV)
  text/: cleaning summary, POS sample, POS mistake, comparison notes, regex notes
  report.pdf: the report

notes:
 * Stemming is applied to the original tokens, not to the stopword-removed
 tokens, so the three views compare against the same baseline.
 * The tagging mistake is found automatically by scanning for an internal
 contradiction (a name tagged as a common noun in one place but as a proper
 noun elsewhere). No example is hard-coded.
 * Proper nouns come straight from the NNP/NNPS tags with only neutral
 cleaning, so the word cloud shows the tagger's real output, noise included.
 * The regex drops single-letter fragments, and every match is labeled as real
 repetition or a coincidental sentence-boundary repeat.
