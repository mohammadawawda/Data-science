
import settings as cfg
from io_utils import (
    collapse_whitespace,
    create_folders,
    create_log_frequency_rank_plot,
    decode_text,
    obtain_raw_bytes,
    remove_gutenberg_header_footer,
    write_ranked_count_csv,
    write_repeated_word_matches_csv,
    write_text_file,
)
from nlp_analysis import (
    build_cleaning_summary,
    build_pos_tagged_sample_text,
    build_ranked_counts,
    create_proper_noun_wordcloud,
    extract_proper_nouns,
    find_pos_tagging_mistake,
    find_repeated_word_matches,
    load_english_stopwords,
    pos_tag_sentences,
    remove_stopwords,
    stem_tokens,
    tokenize_text,
)


def run_download_and_cleaning() -> str:
    raw_bytes = obtain_raw_bytes()
    raw_text = decode_text(raw_bytes)
    cleaned_text = remove_gutenberg_header_footer(raw_text)
    write_text_file(cfg.CLEAN_TEXT_PATH, cleaned_text + "\n")
    write_text_file(cfg.CLEANING_SUMMARY_PATH, build_cleaning_summary(raw_text, cleaned_text) + "\n")
    return cleaned_text


def run_original_counts(cleaned_text: str):
    tokens = tokenize_text(cleaned_text)
    ranked = build_ranked_counts(tokens)
    top20 = ranked[:20]
    write_ranked_count_csv(cfg.ORIGINAL_COUNTS_PATH, ranked)
    write_ranked_count_csv(cfg.TOP20_ORIGINAL_PATH, top20)
    create_log_frequency_rank_plot(
        ranked,
        "Original token frequencies in Alice's Adventures in Wonderland",
        cfg.ORIGINAL_LOG_PLOT_PATH,
    )
    print(f"[b/c] original: {len(tokens)} tokens, {len(ranked)} unique")
    return tokens, ranked, top20


def run_stopword_counts(tokens: list[str]):
    stopwords, source = load_english_stopwords()
    filtered = remove_stopwords(tokens, stopwords)
    ranked = build_ranked_counts(filtered)
    top20 = ranked[:20]
    write_ranked_count_csv(cfg.STOPWORD_REMOVED_COUNTS_PATH, ranked)
    write_ranked_count_csv(cfg.TOP20_STOPWORD_REMOVED_PATH, top20)
    create_log_frequency_rank_plot(
        ranked,
        "Token frequencies after stopword removal",
        cfg.STOPWORD_REMOVED_LOG_PLOT_PATH,
    )
    print(f"[d] stopwords removed ({source}): {len(filtered)} tokens, {len(ranked)} unique")
    return stopwords, filtered, ranked, top20


def run_stemming(original_tokens: list[str]):
    # FIX: stem the ORIGINAL tokens, i.e. "repeat (b) with stemmed text".
    stemmed = stem_tokens(original_tokens)
    ranked = build_ranked_counts(stemmed)
    top20 = ranked[:20]
    write_ranked_count_csv(cfg.STEMMED_COUNTS_PATH, ranked, "stem")
    write_ranked_count_csv(cfg.TOP20_STEMMED_PATH, top20, "stem")
    create_log_frequency_rank_plot(
        ranked,
        "Stemmed token frequencies (whole text)",
        cfg.STEMMED_LOG_PLOT_PATH,
    )
    print(f"[e] stemmed whole text: {len(stemmed)} tokens, {len(ranked)} unique stems")
    return stemmed, ranked, top20


def run_pos_tagging(cleaned_text: str):
    tagged_sentences = pos_tag_sentences(cleaned_text)
    write_text_file(cfg.POS_TAGGED_SAMPLE_PATH, build_pos_tagged_sample_text(tagged_sentences))
    found, mistake_text = find_pos_tagging_mistake(tagged_sentences)
    write_text_file(cfg.POS_TAGGING_MISTAKE_PATH, mistake_text + "\n")
    print(f"[f] sentences: {len(tagged_sentences)}, mistake found automatically: {found}")
    return tagged_sentences


def run_proper_nouns(tagged_sentences):
    proper_nouns = extract_proper_nouns(tagged_sentences)
    ranked = build_ranked_counts(proper_nouns)
    top30 = ranked[:30]
    write_ranked_count_csv(cfg.PROPER_NOUN_COUNTS_PATH, ranked, "proper_noun")
    write_ranked_count_csv(cfg.TOP30_PROPER_NOUNS_PATH, top30, "proper_noun")
    create_proper_noun_wordcloud(ranked)
    print(f"[g] proper nouns: {len(proper_nouns)} tokens, {len(ranked)} unique")
    return ranked, top30


def run_repeated_word_regex(cleaned_text: str):
    matches = find_repeated_word_matches(cleaned_text)
    write_repeated_word_matches_csv(cfg.REPEATED_WORD_MATCHES_PATH, matches)
    repetition = sum(1 for m in matches if m[3] == "repetition")
    boundary = len(matches) - repetition
    print(f"[h] repeated-word matches: {len(matches)} ({repetition} real, {boundary} boundary)")
    return matches


def main() -> None:
    create_folders()
    cleaned_text = run_download_and_cleaning()

    tokens, _, _ = run_original_counts(cleaned_text)
    run_stopword_counts(tokens)
    run_stemming(tokens)

    tagged_sentences = run_pos_tagging(cleaned_text)
    run_proper_nouns(tagged_sentences)
    run_repeated_word_regex(cleaned_text)
    print("Done.")


if __name__ == "__main__":
    main()