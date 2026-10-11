
import re
from collections import Counter, defaultdict

import settings as cfg
from io_utils import collapse_whitespace


def normalize_quotes(text: str) -> str:
    replacements = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def tokenize_text(text: str) -> list[str]:
    """Tokenize with NLTK, keep only word tokens (letters).

    word_tokenize also splits "alice's" into "alice" and "'s", so the possessive
    "'s" is dropped by the isalpha filter and the name is counted on its own.
    """
    import nltk

    normalized_text = normalize_quotes(text).lower()
    tokens = nltk.word_tokenize(normalized_text)
    return [token for token in tokens if token.isalpha()]


def build_ranked_counts(items: list[str]) -> list[tuple[int, str, int]]:
    """Count with nltk.FreqDist, then rank by count (ties breaks alphabetically)."""
    from nltk import FreqDist

    frequencies = FreqDist(items)
    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return [
        (rank, item, count)
        for rank, (item, count) in enumerate(ranked, start=1)
    ]



def load_english_stopwords() -> tuple[set[str], str]:
    try:
        import nltk
        from nltk.corpus import stopwords
    except Exception as error:
        print(f"Falling back to built-in stopwords: {error}")
        return set(cfg.FALLBACK_ENGLISH_STOPWORDS), "fallback English stopword set"

    try:
        return set(stopwords.words("english")), "nltk English stopwords"
    except LookupError:
        nltk.download("stopwords", quiet=False)
    try:
        return set(stopwords.words("english")), "nltk English stopwords"
    except Exception as error:
        print(f"Falling back to built-in stopwords: {error}")
        return set(cfg.FALLBACK_ENGLISH_STOPWORDS), "fallback English stopword set"


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    return [token for token in tokens if token not in stopwords]


def stem_tokens(tokens: list[str]) -> list[str]:
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]



def ensure_nltk_resource(nltk_module, resource_path: str, download_name: str) -> None:
    try:
        nltk_module.data.find(resource_path)
    except LookupError:
        nltk_module.download(download_name, quiet=False)


def ensure_pos_tagging_resources(nltk_module) -> None:
    for resource_path, download_name in cfg.POS_RESOURCE_DOWNLOADS:
        try:
            ensure_nltk_resource(nltk_module, resource_path, download_name)
        except Exception as error:
            print(f"Could not download {download_name}: {error}")


def pos_tag_sentences(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    import nltk

    ensure_pos_tagging_resources(nltk)
    sentences = nltk.sent_tokenize(text)
    tagged_sentences = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged_sentences.append((sentence, nltk.pos_tag(tokens)))
    return tagged_sentences


def format_tagged_tokens(tagged_tokens: list[tuple[str, str]]) -> str:
    return " ".join(f"{token}/{tag}" for token, tag in tagged_tokens)


def build_pos_tagged_sample_text(
    tagged_sentences: list[tuple[str, list[tuple[str, str]]]],
    sample_size: int = 50,
) -> str:
    lines = []
    for index, (_, tagged_tokens) in enumerate(tagged_sentences[:sample_size], start=1):
        lines.extend([f"Sentence {index}:", format_tagged_tokens(tagged_tokens), ""])
    return "\n".join(lines).rstrip() + "\n"


def build_surface_tag_counts(
    tagged_sentences: list[tuple[str, list[tuple[str, str]]]],
) -> dict[str, Counter]:
    """For every surface word, count how many times the tagger gave it each tag."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for _, tagged_tokens in tagged_sentences:
        for token, tag in tagged_tokens:
            counts[token][tag] += 1
    return counts


def find_pos_tagging_mistake(
    tagged_sentences: list[tuple[str, list[tuple[str, str]]]],
) -> tuple[bool, str]:
    """Automatically find a real tagging mistake by looking for an internal
    contradiction: a capitalised word that the tagger calls a common noun
    (NN/NNS) in one place, but calls a proper noun (NNP/NNPS) most of the time
    elsewhere. That word is almost certainly a name, so the common noun tag is
    the wrong one. We report the clearest (shortest) such sentence.
    """
    surface_counts = build_surface_tag_counts(tagged_sentences)

    # We want a real, readable sentence (not a one-word exclamation like
    # "Pat!"), so we only look at sentences of at least MIN_LEN tokens and then
    # take the shortest of those, which stays easy to read in the report.
    MIN_LEN = 8
    best = None  # (sentence_length, sentence, tagged_tokens, token, tag, nnp, nn)
    for sentence, tagged_tokens in tagged_sentences:
        if len(tagged_tokens) < MIN_LEN:
            continue
        for token, tag in tagged_tokens:
            if tag not in {"NN", "NNS"}:
                continue
            if not token.isalpha() or not token[:1].isupper():
                continue
            tag_counts = surface_counts[token]
            proper = tag_counts["NNP"] + tag_counts["NNPS"]
            common = tag_counts["NN"] + tag_counts["NNS"]
            # the word is usually a proper noun in this same book
            if proper >= 2 and proper > common:
                length = len(tagged_tokens)
                candidate = (length, sentence, tagged_tokens, token, tag, proper, common)
                if best is None or length < best[0]:
                    best = candidate

    if best is None:
        return False, "No clear tagging inconsistency was found automatically."

    _, sentence, tagged_tokens, token, tag, proper, _ = best
    text = "\n".join(
        [
            "Selected sentence:",
            collapse_whitespace(sentence),
            "",
            "POS-tagged version:",
            format_tagged_tokens(tagged_tokens),
            "",
            f"Questionable token: {token}",
            f"Tag assigned by NLTK: {tag} (common noun)",
            "More appropriate tag: NNP (proper noun)",
            "",
            "How this was found automatically:",
            f"The tagger labels '{token}' as a proper noun (NNP/NNPS) "
            f"{proper} times elsewhere in the same book, but here it tagged it "
            f"as a common noun ({tag}). Since '{token}' is used as a character "
            "name, the proper-noun tag is the correct one, so the tag in this "
            "sentence is a mistake.",
        ]
    )
    return True, text



def clean_proper_noun_token(token: str) -> str | None:
    """Neutral cleanup only: strip non-letters, drop very short leftovers.
    We do NOT remove specific words to make the cloud look correct."""
    cleaned = re.sub(r"[^A-Za-z]", "", token)
    if len(cleaned) < 2:
        return None
    return cleaned[:1].upper() + cleaned[1:].lower()


def extract_proper_nouns(
    tagged_sentences: list[tuple[str, list[tuple[str, str]]]],
) -> list[str]:
    proper_nouns = []
    for _, tagged_tokens in tagged_sentences:
        for token, tag in tagged_tokens:
            if tag in {"NNP", "NNPS"}:
                cleaned = clean_proper_noun_token(token)
                if cleaned is not None:
                    proper_nouns.append(cleaned)
    return proper_nouns


def create_proper_noun_wordcloud(
    ranked_proper_nouns: list[tuple[int, str, int]],
) -> bool:
    if not ranked_proper_nouns:
        print("No proper nouns found, so no word cloud was created.")
        return False
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("The wordcloud package is not installed (pip install wordcloud).")
        return False

    frequencies = {name: count for _, name, count in ranked_proper_nouns}
    wordcloud = WordCloud(
        width=1200, height=800, background_color="white",
        colormap="viridis", collocations=False, random_state=42,
    ).generate_from_frequencies(frequencies)
    wordcloud.to_file(cfg.PROPER_NOUN_WORDCLOUD_PATH)
    return True



def classify_repeated_match(separator: str) -> str:
    """Heuristic label. If the two copies are split by a period or colon, the
    second copy is starting a new sentence/clause, so it is a coincidental
    boundary repeat rather than real emphatic/stuttered repetition."""
    if "." in separator or ":" in separator:
        return "sentence boundary"
    return "repetition"


def find_repeated_word_matches(
    text: str,
    context_size: int = 60,
) -> list[tuple[int, str, str, str, int, str]]:
    pattern = re.compile(cfg.REPEATED_WORD_REGEX_PATTERN, re.IGNORECASE)
    matches = []
    for rank, match in enumerate(pattern.finditer(text), start=1):
        start, end = match.start(), match.end()
        first_word_end = match.start(1) + len(match.group(1))
        separator = text[first_word_end : end - len(match.group(1))]
        category = classify_repeated_match(separator)
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        matches.append(
            (
                rank,
                collapse_whitespace(match.group(0)),
                match.group(1),
                category,
                start,
                collapse_whitespace(text[context_start:context_end]),
            )
        )
    return matches



def build_cleaning_summary(raw_text: str, cleaned_text: str) -> str:
    cleaned_upper = cleaned_text.upper()
    return "\n".join(
        [
            f"Book title: {cfg.BOOK_TITLE}",
            f"Author: {cfg.AUTHOR}",
            f"Source URL: {cfg.BOOK_URL}",
            f"Raw character count: {len(raw_text)}",
            f"Cleaned character count: {len(cleaned_text)}",
            "",
            'Cleaned text contains "PROJECT GUTENBERG": '
            f"{'PROJECT GUTENBERG' in cleaned_upper}",
            'Cleaned text starts with "CHAPTER I.": '
            f"{cleaned_text.startswith('CHAPTER I.')}",
        ]
    )