from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError as exc:
    print("Missing dependency: nltk. Install it with `pip install nltk tqdm`.", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    print("Missing dependency: tqdm. Install it with `pip install tqdm`.", file=sys.stderr)
    raise SystemExit(1) from exc


RESOURCE_PATHS = {
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip/wordnet/"),
    "omw-1.4": ("corpora/omw-1.4", "corpora/omw-1.4.zip/omw-1.4/"),
}
WORD_PATTERN = re.compile(r"^[a-z][a-z'-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill weights.json with real English words and WordNet definitions."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10_000,
        help="Number of word-definition pairs to write. Default: 10000.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("weights.json"),
        help="Output JSON path. Default: weights.json next to this script.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=4,
        help="Minimum word length to include. Default: 4.",
    )
    return parser.parse_args()


def ensure_corpora() -> None:
    for resource_name, resource_paths in RESOURCE_PATHS.items():
        if any(resource_exists(path) for path in resource_paths):
            continue

        print(f"Downloading NLTK resource: {resource_name}")
        nltk.download(resource_name, quiet=False)


def resource_exists(resource_path: str) -> bool:
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        return False


def normalize_word(raw_word: str) -> str | None:
    word = unicodedata.normalize("NFC", raw_word).strip().lower()
    if not word:
        return None
    if "_" in word or " " in word:
        return None
    if not WORD_PATTERN.fullmatch(word):
        return None
    return word


def normalize_definition(raw_definition: str) -> str:
    normalized = unicodedata.normalize("NFC", raw_definition)
    return " ".join(normalized.split())


def is_candidate_word(word: str, min_length: int) -> bool:
    return len(word) >= min_length


@lru_cache(maxsize=None)
def best_definition(word: str) -> str | None:
    synsets = wn.synsets(word)
    if not synsets:
        return None

    best_synset = None
    best_count = -1

    for synset in synsets:
        counts = [
            lemma.count()
            for lemma in synset.lemmas()
            if normalize_word(lemma.name()) == word
        ]
        if counts and max(counts) > best_count:
            best_count = max(counts)
            best_synset = synset

    if best_synset is None:
        best_synset = synsets[0]

    return normalize_definition(best_synset.definition())


def rank_wordnet_words(min_length: int) -> list[str]:
    best_entries: dict[str, tuple[int, str]] = {}
    synsets = list(wn.all_synsets())

    for synset in tqdm(synsets, desc="Scanning WordNet", unit="synset"):
        definition = normalize_definition(synset.definition())
        for lemma in synset.lemmas():
            word = normalize_word(lemma.name())
            if not word or not is_candidate_word(word, min_length):
                continue

            score = lemma.count()
            current = best_entries.get(word)
            if current is None or score > current[0]:
                best_entries[word] = (score, definition)

    if not best_entries:
        return []

    ranked_items = sorted(best_entries.items(), key=lambda item: (-item[1][0], item[0]))
    return [word for word, _entry in ranked_items]


def collect_weights(target_count: int, min_length: int) -> dict[str, str]:
    weights: dict[str, str] = {}
    ranked_words = rank_wordnet_words(min_length)

    with tqdm(total=target_count, desc="Collecting definitions", unit="word") as progress:
        for word in ranked_words:
            definition = best_definition(word)
            if not definition or word in weights:
                continue

            weights[word] = definition
            progress.update(1)

            if len(weights) >= target_count:
                return weights

    raise RuntimeError(
        f"Could not find {target_count} unique English words with definitions in NLTK."
    )


def write_weights(weights: dict[str, str], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_weights = dict(sorted(weights.items()))

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
        newline="\n",
    ) as handle:
        json.dump(ordered_weights, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)

    temp_path.replace(output_path)


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        print("The word count must be greater than zero.", file=sys.stderr)
        return 1
    if args.min_length <= 0:
        print("The minimum word length must be greater than zero.", file=sys.stderr)
        return 1

    ensure_corpora()
    weights = collect_weights(args.count, args.min_length)
    write_weights(weights, args.output)

    print(f"Wrote {len(weights):,} words to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
