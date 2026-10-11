from pathlib import Path


BOOK_TITLE = "Alice's Adventures in Wonderland"
AUTHOR = "Lewis Carroll"
BOOK_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TEXT_DIR = PROJECT_ROOT / "output" / "text"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "output" / "tables"

# The raw book is cached here after the first download
RAW_TEXT_PATH = OUTPUT_TEXT_DIR / "alice_raw.txt"
CLEAN_TEXT_PATH = OUTPUT_TEXT_DIR / "alice_clean.txt"
CLEANING_SUMMARY_PATH = OUTPUT_TEXT_DIR / "cleaning_summary.txt"
POS_TAGGED_SAMPLE_PATH = OUTPUT_TEXT_DIR / "pos_tagged_sample.txt"
POS_TAGGING_MISTAKE_PATH = OUTPUT_TEXT_DIR / "pos_tagging_mistake.txt"

ORIGINAL_COUNTS_PATH = OUTPUT_TABLES_DIR / "original_token_counts.csv"
TOP20_ORIGINAL_PATH = OUTPUT_TABLES_DIR / "top20_original_tokens.csv"
STOPWORD_REMOVED_COUNTS_PATH = OUTPUT_TABLES_DIR / "stopword_removed_token_counts.csv"
TOP20_STOPWORD_REMOVED_PATH = OUTPUT_TABLES_DIR / "top20_stopword_removed_tokens.csv"
STEMMED_COUNTS_PATH = OUTPUT_TABLES_DIR / "stemmed_token_counts.csv"
TOP20_STEMMED_PATH = OUTPUT_TABLES_DIR / "top20_stemmed_tokens.csv"
PROPER_NOUN_COUNTS_PATH = OUTPUT_TABLES_DIR / "proper_noun_counts.csv"
TOP30_PROPER_NOUNS_PATH = OUTPUT_TABLES_DIR / "top30_proper_nouns.csv"
REPEATED_WORD_MATCHES_PATH = OUTPUT_TABLES_DIR / "repeated_word_matches.csv"

ORIGINAL_LOG_PLOT_PATH = OUTPUT_FIGURES_DIR / "original_log_frequency_rank.png"
STOPWORD_REMOVED_LOG_PLOT_PATH = OUTPUT_FIGURES_DIR / "stopword_removed_log_frequency_rank.png"
STEMMED_LOG_PLOT_PATH = OUTPUT_FIGURES_DIR / "stemmed_log_frequency_rank.png"
PROPER_NOUN_WORDCLOUD_PATH = OUTPUT_FIGURES_DIR / "proper_noun_wordcloud.png"

# Part (h): two consecutive repeated words with optional punctuation between
# them. The word is 2+ letters, or one of the real one-letter words a / i / o,
# which keeps genuine cases like "I I" while dropping single letter fragments
# of stylised spelling such as the "e" in "e-e-evening".
REPEATED_WORD_REGEX_PATTERN = (
    r"\b([A-Za-z]{2,}|[AaIiOo])\b(?:[\s,;:!?.\-]|\u2014|\u2013)+\b\1\b"
)

# Only used if theres no internet AND nltk stopwords are missing.
FALLBACK_ENGLISH_STOPWORDS = set(
    """
    a about above after again against ain all am an and any are aren aren't as at
    be because been before being below between both but by can can't cannot could
    couldn couldn't d did didn didn't do does doesn doesn't doing don don't down
    during each few for from further had hadn hadn't has hasn hasn't have haven
    haven't having he he'd he'll he's her here hers herself him himself his how i
    i'd i'll i'm i've if in into is isn isn't it it's its itself just ll m ma me
    mightn mightn't more most mustn mustn't my myself needn needn't no nor not
    now o of off on once only or other our ours ourselves out over own re s same
    shan shan't she she'd she'll she's should shouldn shouldn't should've so some
    such t than that that'll the their theirs them themselves then there these
    they they'd they'll they're they've this those through to too under until up
    ve very was wasn wasn't we we'd we'll we're we've were weren weren't what
    when where which while who whom why will with won won't would wouldn wouldn't
    y you you'd you'll you're you've your yours yourself yourselves
    """.split()
)

POS_RESOURCE_DOWNLOADS = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab/english", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]