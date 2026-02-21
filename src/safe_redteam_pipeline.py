"""Safe adversarial red-team case generator.

This script implements a small-data, category-conditioned generation pipeline for
safety testing. It intentionally keeps explicit hateful/profane lexicon abstracted
with placeholders.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

TOKEN_RE = re.compile(r"\[?[A-Za-z_']+\]?|[.,!?]")


@dataclass(frozen=True)
class SeedRow:
    id: str
    category: str
    text: str


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def detokenize(tokens: Sequence[str]) -> str:
    out: List[str] = []
    for t in tokens:
        if t in {".", ",", "!", "?"} and out:
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return " ".join(out)


def load_seed_jsonl(path: Path) -> List[SeedRow]:
    rows: List[SeedRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                rows.append(
                    SeedRow(
                        id=str(obj["id"]),
                        category=str(obj["category"]).strip().lower(),
                        text=str(obj["text"]).strip(),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing key {exc} in seed file line {line_no}") from exc
    if not rows:
        raise ValueError(f"No valid rows loaded from {path}")
    return rows


class CategoryMarkovGenerator:
    """Category-specific bigram Markov generator with template mutation fallback."""

    def __init__(self, seed_rows: Iterable[SeedRow], n: int = 2) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.chain: Dict[str, Dict[Tuple[str, ...], List[str]]] = defaultdict(lambda: defaultdict(list))
        self.starts: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
        self.templates: Dict[str, List[str]] = defaultdict(list)

        for row in seed_rows:
            toks = tokenize(row.text)
            if len(toks) < self.n:
                continue
            self.templates[row.category].append(row.text)
            self.starts[row.category].append(tuple(toks[: self.n]))
            for i in range(len(toks) - self.n):
                state = tuple(toks[i : i + self.n])
                nxt = toks[i + self.n]
                self.chain[row.category][state].append(nxt)

    def sample_markov(self, category: str, max_len: int = 30) -> str:
        start_pool = self.starts.get(category, [])
        if not start_pool:
            return ""

        state = random.choice(start_pool)
        out = list(state)
        for _ in range(max_len - self.n):
            next_candidates = self.chain[category].get(state)
            if not next_candidates:
                break
            nxt = random.choice(next_candidates)
            out.append(nxt)
            state = tuple(out[-self.n :])
            if nxt in {".", "!", "?"}:
                break

        return detokenize(out)

    def sample_template_mutation(self, category: str) -> str:
        templates = self.templates.get(category)
        if not templates:
            return ""
        text = random.choice(templates).lower()

        swaps = {
            "you are": ["you seem", "you sound", "you come across as"],
            "this is": ["that is", "this feels", "this looks"],
            "nobody": ["no one", "not one person"],
            "always": ["consistently", "every time"],
            "keep": ["continue", "go on and"],
        }
        for src, candidates in swaps.items():
            if src in text and random.random() < 0.45:
                text = text.replace(src, random.choice(candidates), 1)

        if random.random() < 0.35:
            text += random.choice([" today", " right now", " at this point"])
        return text


class PlaceholderFiller:
    MAP = {
        "[INSULT]": ["clown", "fool", "jerk", "buffoon", "menace"],
        "[PROFANITY]": ["[EXPLETIVE]", "[OBSCENITY]", "[VULGARITY]"],
        "[TARGET_GROUP]": ["[GROUP_A]", "[GROUP_B]", "[GROUP_C]"],
        "[DEMEANING_ADJ]": ["[DEROGATORY_ADJ]", "[DEHUMANIZING_ADJ]"],
    }

    @classmethod
    def fill(cls, text: str) -> str:
        out = text
        for key, values in cls.MAP.items():
            while re.search(re.escape(key), out, flags=re.IGNORECASE):
                out = re.sub(re.escape(key), random.choice(values), out, count=1, flags=re.IGNORECASE)
        return out


class Evaluator:
    STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "at",
        "as",
        "by",
        "from",
        "you",
        "your",
        "i",
        "we",
        "they",
        "he",
        "she",
        "them",
        "their",
        "our",
        "my",
        "me",
        "us",
        "do",
        "does",
        "did",
        "not",
        "no",
        "so",
        "if",
        "then",
        "than",
        "just",
        "all",
        "any",
        "can",
        "could",
        "should",
        "would",
        "will",
        "about",
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "but",
        "because",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "again",
        "more",
        "most",
        "very",
        "here",
        "there",
        "now",
        "too",
    }

    def __init__(self, seed_rows: Sequence[SeedRow], top_k: int = 80, min_freq: int = 5) -> None:
        self.markers = self._build_markers(seed_rows, top_k=top_k, min_freq=min_freq)

    def _valid_marker_token(self, tok: str) -> bool:
        if tok in {".", ",", "!", "?"}:
            return False
        if tok in self.STOPWORDS:
            return False
        if len(tok) <= 2:
            return False
        return True

    def _build_markers(
        self, seed_rows: Sequence[SeedRow], top_k: int = 80, min_freq: int = 5
    ) -> Dict[str, set]:
        by_cat_counts: Dict[str, Counter] = defaultdict(Counter)
        by_cat_totals: Counter = Counter()
        all_categories = sorted({r.category for r in seed_rows})

        for row in seed_rows:
            toks = tokenize(row.text)
            filtered = [t for t in toks if self._valid_marker_token(t)]
            by_cat_counts[row.category].update(filtered)
            by_cat_totals[row.category] += len(filtered)

        markers: Dict[str, set] = {}
        for cat in all_categories:
            cat_counts = by_cat_counts[cat]
            cat_total = max(1, by_cat_totals[cat])

            other_counts = Counter()
            other_total = 0
            for other_cat in all_categories:
                if other_cat == cat:
                    continue
                other_counts.update(by_cat_counts[other_cat])
                other_total += by_cat_totals[other_cat]
            other_total = max(1, other_total)

            scored: List[Tuple[str, float, int]] = []
            for tok, c_cat in cat_counts.items():
                if c_cat < min_freq:
                    continue
                p_cat = c_cat / cat_total
                p_other = other_counts.get(tok, 0) / other_total
                score = p_cat - p_other
                if score > 0:
                    scored.append((tok, score, c_cat))

            scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            markers[cat] = {tok for tok, _, _ in scored[:top_k]}

        return markers

    @staticmethod
    def distinct_n(texts: Sequence[str], n: int) -> float:
        uniq = set()
        total = 0
        for t in texts:
            toks = tokenize(t)
            if len(toks) < n:
                continue
            grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
            total += len(grams)
            uniq.update(grams)
        return len(uniq) / total if total else 0.0

    @staticmethod
    def novelty_rate(seed_texts: Sequence[str], gen_texts: Sequence[str]) -> float:
        seed_norm = {" ".join(tokenize(s)) for s in seed_texts}
        gen_norm = [" ".join(tokenize(s)) for s in gen_texts]
        novel = [x for x in gen_norm if x not in seed_norm]
        return len(novel) / max(1, len(gen_norm))

    def category_hit_rate(self, rows: Sequence[dict]) -> float:
        hits = 0
        for row in rows:
            cat = row["category"]
            toks = set(tokenize(row["text"]))
            if toks & self.markers.get(cat, set()):
                hits += 1
        return hits / max(1, len(rows))

    def evaluate(self, seed_rows: Sequence[SeedRow], generated_rows: Sequence[dict]) -> Dict[str, float]:
        seed_texts = [r.text for r in seed_rows]
        gen_texts = [r["text"] for r in generated_rows]
        count_by_cat = Counter(r["category"] for r in generated_rows)
        total = max(1, len(generated_rows))

        return {
            "distinct_1": round(self.distinct_n(gen_texts, 1), 4),
            "distinct_2": round(self.distinct_n(gen_texts, 2), 4),
            "novelty_rate": round(self.novelty_rate(seed_texts, gen_texts), 4),
            "category_hit_rate": round(self.category_hit_rate(generated_rows), 4),
            "offensive_share": round(count_by_cat.get("offensive", 0) / total, 4),
            "harassment_share": round(count_by_cat.get("harassment", 0) / total, 4),
            "hate_share": round(count_by_cat.get("hate", 0) / total, 4),
        }


def generate_conditioned(seed_rows: Sequence[SeedRow], num_cases: int, rng_seed: int) -> List[dict]:
    random.seed(rng_seed)
    model = CategoryMarkovGenerator(seed_rows, n=2)
    categories = sorted({r.category for r in seed_rows}) or ["offensive", "harassment", "hate"]

    outputs: List[dict] = []
    for i in range(num_cases):
        cat = categories[i % len(categories)]
        if random.random() < 0.6:
            raw = model.sample_markov(cat)
        else:
            raw = model.sample_template_mutation(cat)
        text = PlaceholderFiller.fill(raw)
        outputs.append({"id": i + 1, "category": cat, "text": text})
    return outputs


def generate_generic_baseline(seed_rows: Sequence[SeedRow], num_cases: int, rng_seed: int) -> List[dict]:
    random.seed(rng_seed)
    pool = [r.text for r in seed_rows]
    cats = sorted({r.category for r in seed_rows}) or ["offensive", "harassment", "hate"]

    outputs: List[dict] = []
    for i in range(num_cases):
        a, b = random.sample(pool, 2)
        left = tokenize(a)[: random.randint(5, 9)]
        right = tokenize(b)[-random.randint(4, 8) :]
        mixed = detokenize((left + right)[:24])
        text = PlaceholderFiller.fill(mixed)
        outputs.append({"id": i + 1, "category": random.choice(cats), "text": text})
    return outputs


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe category-conditioned red-team generator")
    parser.add_argument("--seed", default="data/hatexplain_seed.jsonl", help="Seed JSONL path")
    parser.add_argument("--count", type=int, default=90, help="Number of generated cases")
    parser.add_argument("--seed_value", type=int, default=13, help="Random seed")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    seed_rows = load_seed_jsonl(Path(args.seed))
    conditioned = generate_conditioned(seed_rows, args.count, args.seed_value)
    baseline = generate_generic_baseline(seed_rows, args.count, args.seed_value)

    evaluator = Evaluator(seed_rows)
    conditioned_metrics = evaluator.evaluate(seed_rows, conditioned)
    baseline_metrics = evaluator.evaluate(seed_rows, baseline)

    summary = {
        "meta": {
            "seed_path": args.seed,
            "seed_size": len(seed_rows),
            "generated_count": args.count,
            "random_seed": args.seed_value,
            "safety_note": "Generated cases keep explicit hate/profanity abstracted by placeholders.",
            "marker_sizes": {k: len(v) for k, v in evaluator.markers.items()},
        },
        "conditioned_metrics": conditioned_metrics,
        "generic_baseline_metrics": baseline_metrics,
        "delta_vs_generic": {
            k: round(conditioned_metrics[k] - baseline_metrics[k], 4)
            for k in conditioned_metrics
        },
    }

    out_dir = Path(args.out_dir)
    conditioned_path = out_dir / "conditioned_outputs.json"
    baseline_path = out_dir / "generic_outputs.json"
    report_path = out_dir / "metrics_report.json"

    save_json(conditioned_path, conditioned)
    save_json(baseline_path, baseline)
    save_json(report_path, summary)

    print("Wrote:")
    print(f"- {conditioned_path}")
    print(f"- {baseline_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
