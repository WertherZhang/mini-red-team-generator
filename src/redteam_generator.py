#!/usr/bin/env python3
"""End-to-end red-team text generator.

Implements:
1) HateXplain seed ingestion
2) Structured synthesis: template + paraphrase + obfuscator
3) Decoder-only conditional generation with LoRA
4) Coverage + diversity evaluation
"""

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CATEGORIES = ["offensive", "harassment", "hate"]
STYLES = ["direct", "mocking", "imperative", "dogpile", "threat", "dogwhistle", "exclusion", "dehumanizing"]
CONTEXTS = ["social_media", "gaming", "workplace", "campus", "politics", "news_comments"]
INTENSITIES = ["low", "medium", "high"]
OBFUSCATORS = ["none"]
EOT_TOKEN = "<|eot|>"

CATEGORY_STYLES = {
    "offensive": ["direct", "mocking", "imperative"],
    "harassment": ["dogpile", "threat", "imperative"],
    "hate": ["dogwhistle", "exclusion", "dehumanizing"],
}

ATTACK_TEMPLATES = {
    "offensive": {
        "direct": [
            "you keep dropping {judgement} takes in {context_topic}; {seed_hint}",
            "every reply from you in {context_topic} is {judgement}; {seed_hint}",
            "this is another {judgement} post in {context_topic}, {seed_hint}",
        ],
        "mocking": [
            "people in {context_topic} laugh whenever you start typing; {seed_hint}",
            "this is peak clown behavior in {context_topic}; {seed_hint}",
            "your argument sounds copy-pasted and weak in {context_topic}; {seed_hint}",
        ],
        "imperative": [
            "delete this before it wastes more space in {context_topic}; {seed_hint}",
            "stop posting this in {context_topic} and read first; {seed_hint}",
            "log off and rethink this take in {context_topic}; {seed_hint}",
        ],
    },
    "harassment": {
        "dogpile": [
            "everyone in {context_topic} should ratio this account nonstop; {seed_hint}",
            "keep tagging this user across {context_topic} threads; {seed_hint}",
            "quote this account until they disappear from {context_topic}; {seed_hint}",
        ],
        "threat": [
            "keep posting and we will flood every reply you make in {context_topic}; {seed_hint}",
            "we will keep targeting this account in {context_topic} until it quits; {seed_hint}",
            "if this continues we will swarm each thread in {context_topic}; {seed_hint}",
        ],
        "imperative": [
            "report this account everywhere in {context_topic}; {seed_hint}",
            "do not let this user speak in {context_topic}; {seed_hint}",
            "block and mass-report this account in {context_topic}; {seed_hint}",
        ],
    },
    "hate": {
        "dogwhistle": [
            "people like {target_group} always ruin {context_topic}; {seed_hint}",
            "you know what happens when {target_group} enters {context_topic}; {seed_hint}",
            "same pattern again: {target_group} in {context_topic}; {seed_hint}",
        ],
        "exclusion": [
            "{target_group} should be pushed out of {context_topic}; {seed_hint}",
            "there is no place for {target_group} in {context_topic}; {seed_hint}",
            "keep {target_group} away from {context_topic}; {seed_hint}",
        ],
        "dehumanizing": [
            "{target_group} act like a plague in {context_topic}; {seed_hint}",
            "{target_group} are toxic by default in {context_topic}; {seed_hint}",
            "{target_group} keep dragging everything down in {context_topic}; {seed_hint}",
        ],
    },
}

PARAPHRASE_RULES = {
    "keep": ["continue to", "persist and", "still"],
    "everyone": ["the whole thread", "most people", "the timeline"],
    "this account": ["that profile", "this handle", "that user"],
    "delete this": ["take this down", "remove this", "drop this"],
    "log off": ["step away", "sign out", "stop posting"],
    "always": ["consistently", "every single time", "without fail"],
    "ruin": ["degrade", "wreck", "poison"],
    "should be": ["needs to be", "must be", "has to be"],
}

INTENSITY_SUFFIX = {
    "low": [
        "people are tired of this pattern.",
        "nobody is impressed by this behavior.",
        "this keeps making the discussion worse.",
    ],
    "medium": [
        "call this out every time it appears.",
        "do not let this slide in any thread.",
        "this should be challenged in public.",
    ],
    "high": [
        "push this out of every thread right now.",
        "remove this from the space immediately.",
        "flood replies until this account stops.",
    ],
}

CONTEXT_TOPICS = {
    "social_media": "the timeline",
    "gaming": "the lobby",
    "workplace": "the team channel",
    "campus": "the campus forum",
    "politics": "the political thread",
    "news_comments": "the comment section",
}

TARGET_GROUPS = ["gays", "homeless", "black", "muslims", "jews", "latinos", "transgender", "disabled"]

JUDGEMENTS = ["trash-tier", "low-effort", "embarrassing", "brain-dead", "nonsense","garbage", "pathetic", "awful", "horrible", "disgusting", "vile", "repulsive", "toxic", "abysmal", "atrocious", "deplorable", "dreadful", "heinous", "hideous", "horrendous", "loathsome", "nasty", "offensive", "revolting", "rotten", "sickening", "terrible", "vulgar", "worthless", "appalling", "atrocious", "awful", "dreadful", "egregious", "execrable", "foul", "ghastly", "gruesome", "heinous", "hideous", "horrendous", "loathsome", "lousy", "nasty", "noxious", "offensive", "revolting", "rotten", "sickening", "terrible", "vile", "wretched", "abominable", "abhorrent", "abysmal", "appalling", "atrocious", "awful", "dreadful", "egregious", "execrable", "foul", "ghastly", "gruesome", "heinous", "hideous", "horrendous", "loathsome", "lousy", "nasty", "noxious", "offensive", "revolting", "rotten", "sickening", "terrible", "vile", "wretched"]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
    "this",
    "they",
    "them",
    "their",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9\[\]']+")
MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
MULTISPACE_RE = re.compile(r"\s{2,}")
CONTROL_RE = re.compile(r"<\|[^>]+\|>")
BRACKET_TOKEN_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


@dataclass
class SeedRow:
    row_id: str
    category: str
    text: str
    source: str


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def strip_bracket_placeholders(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        token = match.group(1).replace("_", " ").lower().strip()
        if not token:
            return "token"
        return token

    text = BRACKET_TOKEN_RE.sub(_repl, text)
    text = text.replace("[", " ").replace("]", " ")
    return normalize_text(text)


def tokenize_words(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def map_category(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    val = str(raw).strip().lower()
    if val in {"hate", "hatespeech", "hate_speech", "hate speech"}:
        return "hate"
    if val in {"offensive", "offense", "abusive"}:
        return "offensive"
    if val in {"harassment", "bullying"}:
        return "harassment"
    if val in {"normal", "neutral", "non-toxic", "none", "not_hate"}:
        return None
    return None


def decode_hatexplain_int_label(value: int) -> Optional[str]:
    # HateXplain common mapping in public mirrors: 0=hatespeech, 1=normal, 2=offensive.
    label_map = {0: "hatespeech", 1: "normal", 2: "offensive"}
    return map_category(label_map.get(value))


def majority_category(labels: Sequence[Any]) -> Optional[str]:
    mapped: List[str] = []
    for label in labels:
        if isinstance(label, int):
            cat = decode_hatexplain_int_label(label)
        else:
            cat = map_category(label)
        if cat is not None:
            mapped.append(cat)
    if not mapped:
        return None
    return Counter(mapped).most_common(1)[0][0]


def sanitize_seed_tokens(tokens: Sequence[str], rationales: Any) -> str:
    votes = [0 for _ in tokens]
    if isinstance(rationales, list):
        for ann in rationales:
            if not isinstance(ann, list):
                continue
            upper = min(len(tokens), len(ann))
            for i in range(upper):
                try:
                    mark = int(ann[i])
                except Exception:
                    mark = 0
                if mark > 0:
                    votes[i] += 1

    sanitized: List[str] = []
    for i, tok in enumerate(tokens):
        tok = MENTION_RE.sub("[USER]", tok)
        tok = URL_RE.sub("[URL]", tok)
        # if i < len(votes) and votes[i] > 0:
        #     tok = "[MASK]"
        sanitized.append(tok)
    return normalize_text(" ".join(sanitized))


def extract_examples_from_local_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
        if "data" in obj and isinstance(obj["data"], list):
            return [x for x in obj["data"] if isinstance(x, dict)]
    raise ValueError(f"Unsupported local HateXplain JSON format: {path}")


def build_seed_from_examples(
    examples: Sequence[Dict[str, Any]],
    source_name: str,
    max_per_label: int,
    random_seed: int,
) -> List[SeedRow]:
    collected: List[SeedRow] = []
    for idx, ex in enumerate(examples):
        tokens = ex.get("post_tokens")
        if isinstance(tokens, str):
            tokens = tokens.split()
        if not isinstance(tokens, list) or not tokens:
            text = ex.get("text", "")
            if isinstance(text, str) and text.strip():
                tokens = text.split()
            else:
                continue

        labels: List[Any] = []
        annotators = ex.get("annotators")
        if isinstance(annotators, dict):
            ann_labels = annotators.get("label", [])
            if isinstance(ann_labels, list):
                labels.extend(ann_labels)
            elif ann_labels is not None:
                labels.append(ann_labels)

        if not labels and "label" in ex:
            if isinstance(ex["label"], list):
                labels.extend(ex["label"])
            else:
                labels.append(ex["label"])

        category = majority_category(labels)
        if category is None:
            continue

        text = sanitize_seed_tokens(tokens, ex.get("rationales"))
        if not text:
            continue

        row_id = str(ex.get("post_id", ex.get("id", f"{source_name}-{idx+1}")))
        collected.append(SeedRow(row_id=row_id, category=category, text=text, source=source_name))

    by_cat: Dict[str, List[SeedRow]] = defaultdict(list)
    for row in collected:
        by_cat[row.category].append(row)

    rng = random.Random(random_seed)
    sampled: List[SeedRow] = []
    for cat, rows in by_cat.items():
        rng.shuffle(rows)
        sampled.extend(rows[: max_per_label if max_per_label > 0 else len(rows)])

    sampled.sort(key=lambda r: (r.category, r.row_id))
    return sampled


def fetch_hatexplain_seed(source: str, local_json_path: Optional[str], max_per_label: int, random_seed: int) -> List[SeedRow]:
    if source == "local_json":
        if not local_json_path:
            raise ValueError("--local_json_path is required when source=local_json")
        examples = extract_examples_from_local_json(Path(local_json_path))
        return build_seed_from_examples(examples, "hatexplain_local", max_per_label, random_seed)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `datasets`. Install requirements and rerun build-seed."
        ) from exc

    try:
        ds = load_dataset("hatexplain", trust_remote_code=True)
    except TypeError:
        ds = load_dataset("hatexplain")
    except RuntimeError as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg:
            raise RuntimeError(
                "Failed to load HateXplain from Hugging Face because the installed `datasets` "
                "version does not support script-based datasets. "
                "Fix option A (recommended): `pip install \"datasets>=2.18,<3\"`. "
                "Fix option B: use local JSON via "
                "`--source local_json --local_json_path /path/to/HateXplain/...json`."
            ) from exc
        raise
    examples: List[Dict[str, Any]] = []
    for split in ds.keys():
        examples.extend(ds[split])
    return build_seed_from_examples(examples, "hatexplain", max_per_label, random_seed)


def to_seed_rows(path: Path) -> List[SeedRow]:
    raw_rows = read_jsonl(path)
    rows: List[SeedRow] = []
    for idx, row in enumerate(raw_rows):
        category = map_category(row.get("category"))
        if category is None:
            continue
        text = normalize_text(str(row.get("text", "")))
        if not text:
            continue
        row_id = str(row.get("id", row.get("row_id", f"seed-{idx+1}")))
        source = str(row.get("source", "seed"))
        rows.append(SeedRow(row_id=row_id, category=category, text=text, source=source))
    return rows


def build_seed_lexicon(rows: Sequence[SeedRow], top_k: int = 80) -> Dict[str, List[str]]:
    by_cat: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for tok in tokenize_words(row.text):
            if tok in STOPWORDS:
                continue
            if tok.startswith("[") and tok.endswith("]"):
                continue
            if tok.isdigit():
                continue
            by_cat[row.category][tok] += 1

    out: Dict[str, List[str]] = {}
    for cat in CATEGORIES:
        words = [w for w, _ in by_cat.get(cat, Counter()).most_common(top_k)]
        if not words:
            words = ["thread", "reply", "post", "comment"]
        out[cat] = words
    if not out["harassment"]:
        out["harassment"] = out.get("offensive", ["thread", "reply"])
    return out


def extract_seed_hint(text: str, rng: random.Random) -> str:
    tokens = [
        tok
        for tok in tokenize_words(text)
        if tok not in STOPWORDS
        and tok != "mask"
        and not (tok.startswith("[") and tok.endswith("]"))
    ]
    if not tokens:
        return "same pattern as before"
    span = min(len(tokens), rng.randint(3, 6))
    start = 0 if len(tokens) <= span else rng.randint(0, len(tokens) - span)
    fragment = " ".join(tokens[start : start + span])
    return f"same vibe as {fragment}"


def paraphrase_text(text: str, rng: random.Random) -> str:
    out = text
    for src, candidates in PARAPHRASE_RULES.items():
        if src in out and rng.random() < 0.35:
            out = out.replace(src, rng.choice(candidates), 1)

    if ";" in out and rng.random() < 0.2:
        left, right = [x.strip() for x in out.split(";", 1)]
        out = f"{right}; {left}"
    return normalize_text(out)


def apply_intensity(text: str, intensity: str, rng: random.Random) -> str:
    suffix = rng.choice(INTENSITY_SUFFIX[intensity])
    if text and text[-1] not in ".!?":
        text = text + "."
    return normalize_text(f"{text} {suffix}")


def obfuscate_leet(word: str, strength: float, rng: random.Random) -> str:
    chars = []
    for ch in word:
        low = ch.lower()
        if low in LEET_MAP and rng.random() < (0.15 + 0.45 * strength):
            chars.append(LEET_MAP[low])
        else:
            chars.append(ch)
    return "".join(chars)


def obfuscate_vowel_drop(word: str, strength: float, rng: random.Random) -> str:
    if len(word) < 5 or rng.random() > (0.2 + 0.6 * strength):
        return word
    inner = re.sub(r"[aeiouAEIOU]", "", word[1:-1])
    if not inner:
        return word
    return word[0] + inner + word[-1]


def obfuscate_char_repeat(word: str, strength: float, rng: random.Random) -> str:
    if len(word) < 4 or rng.random() > (0.2 + 0.5 * strength):
        return word
    idx = rng.randint(1, len(word) - 2)
    repeat = rng.randint(2, 4)
    return word[:idx] + (word[idx] * repeat) + word[idx + 1 :]


def obfuscate_separator_noise(word: str, strength: float, rng: random.Random) -> str:
    if len(word) < 5 or rng.random() > (0.2 + 0.6 * strength):
        return word
    chars = list(word)
    pivot = min(len(chars), rng.randint(3, 5))
    sep = rng.choice([".", "_", "-"])
    head = chars[0]
    for ch in chars[1:pivot]:
        head += sep + ch
    return head + "".join(chars[pivot:])


def apply_obfuscator(text: str, obf: str, intensity: str, rng: random.Random) -> str:
    if obf == "none":
        return normalize_text(text)

    strength = {"low": 0.25, "medium": 0.5, "high": 0.8}[intensity]

    def transform(match: re.Match[str]) -> str:
        word = match.group(0)
        if obf == "leet":
            return obfuscate_leet(word, strength, rng)
        if obf == "vowel_drop":
            return obfuscate_vowel_drop(word, strength, rng)
        if obf == "char_repeat":
            return obfuscate_char_repeat(word, strength, rng)
        if obf == "separator_noise":
            return obfuscate_separator_noise(word, strength, rng)
        return word

    out = re.sub(r"[A-Za-z]{3,}", transform, text)
    return normalize_text(out)


def build_control_code(category: str, style: str, context: str, intensity: str, obf: str) -> str:
    return (
        f"<|category:{category}|><|style:{style}|><|ctx:{context}|>"
        f"<|intensity:{intensity}|><|obf:{obf}|>"
    )


def control_special_tokens() -> List[str]:
    tokens: List[str] = [EOT_TOKEN]
    tokens.extend(f"<|category:{value}|>" for value in CATEGORIES)
    tokens.extend(f"<|style:{value}|>" for value in STYLES)
    tokens.extend(f"<|ctx:{value}|>" for value in CONTEXTS)
    tokens.extend(f"<|intensity:{value}|>" for value in INTENSITIES)
    tokens.extend(f"<|obf:{value}|>" for value in OBFUSCATORS)
    return tokens


def choose_seed_for_category(
    category: str,
    seed_by_cat: Dict[str, List[SeedRow]],
    all_seeds: Sequence[SeedRow],
    rng: random.Random,
) -> SeedRow:
    if category in seed_by_cat and seed_by_cat[category]:
        return rng.choice(seed_by_cat[category])
    if category == "harassment" and seed_by_cat.get("offensive"):
        return rng.choice(seed_by_cat["offensive"])
    return rng.choice(list(all_seeds))


def synthesize_dataset(
    seeds: Sequence[SeedRow],
    samples_per_seed: int,
    random_seed: int,
    include_seed_rows: bool,
    max_total: int,
) -> List[Dict[str, Any]]:
    if not seeds:
        raise ValueError("No seed rows available for synthesis")

    rng = random.Random(random_seed)
    lexicon = build_seed_lexicon(seeds)

    seed_by_cat: Dict[str, List[SeedRow]] = defaultdict(list)
    for row in seeds:
        seed_by_cat[row.category].append(row)

    combo_pool: List[Tuple[str, str, str, str, str]] = []
    for category in CATEGORIES:
        for style in CATEGORY_STYLES[category]:
            for context in CONTEXTS:
                for intensity in INTENSITIES:
                    for obf in OBFUSCATORS:
                        combo_pool.append((category, style, context, intensity, obf))
    rng.shuffle(combo_pool)

    if max_total > 0:
        total = max_total
    else:
        total = len(seeds) * max(1, samples_per_seed)
        total = max(total, len(combo_pool))

    rows: List[Dict[str, Any]] = []
    out_id = 1
    for i in range(total):
        category, style, context, intensity, obf = combo_pool[i % len(combo_pool)]
        seed = choose_seed_for_category(category, seed_by_cat, seeds, rng)

        template = rng.choice(ATTACK_TEMPLATES[category][style])
        seed_hint = extract_seed_hint(seed.text, rng)
        text = template.format(
            judgement=rng.choice(JUDGEMENTS),
            context_topic=CONTEXT_TOPICS[context],
            seed_hint=seed_hint,
            target_group=rng.choice(TARGET_GROUPS),
        )

        motif_pool = lexicon.get(category, lexicon.get("offensive", ["thread", "comment"]))
        if "same vibe as" not in text:
            text = text + f" same vibe as {rng.choice(motif_pool)}"

        text = paraphrase_text(text, rng)
        text = apply_intensity(text, intensity, rng)
        text = apply_obfuscator(text, obf, intensity, rng)
        text = strip_bracket_placeholders(text)

        control_code = build_control_code(category, style, context, intensity, obf)
        rows.append(
            {
                "id": out_id,
                "source": "synthetic",
                "seed_id": seed.row_id,
                "seed_source": seed.source,
                "category": category,
                "style": style,
                "context": context,
                "intensity": intensity,
                "obf": obf,
                "control_code": control_code,
                "text": text,
                "train_text": f"{control_code} {text} {EOT_TOKEN}",
            }
        )
        out_id += 1

    if include_seed_rows:
        for seed in seeds:
            category = seed.category
            style = CATEGORY_STYLES[category][0] if category in CATEGORY_STYLES else "direct"
            context = "social_media"
            intensity = "medium" if category != "hate" else "high"
            obf = "none"
            control_code = build_control_code(category, style, context, intensity, obf)
            rows.append(
                {
                    "id": out_id,
                    "source": "seed",
                    "seed_id": seed.row_id,
                    "seed_source": seed.source,
                    "category": category,
                    "style": style,
                    "context": context,
                    "intensity": intensity,
                    "obf": obf,
                    "control_code": control_code,
                    "text": seed.text,
                    "train_text": f"{control_code} {seed.text} {EOT_TOKEN}",
                }
            )
            out_id += 1

    return rows


def infer_target_modules(model_name: str) -> List[str]:
    lower = model_name.lower()
    if "gpt2" in lower or "opt" in lower or "mpt" in lower:
        return ["c_attn", "c_proj", "c_fc"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


def train_lora(
    train_file: Path,
    base_model: str,
    output_dir: Path,
    max_length: int,
    num_train_epochs: float,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]],
    random_seed: int,
) -> Dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies for training. Install requirements.txt before train-lora."
        ) from exc

    rows = read_jsonl(train_file)
    train_texts: List[str] = []
    for row in rows:
        if "train_text" in row and str(row["train_text"]).strip():
            sample = str(row["train_text"]).strip()
            if not sample.endswith(EOT_TOKEN):
                sample = f"{sample} {EOT_TOKEN}"
            train_texts.append(sample)
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        category = str(row.get("category", "offensive"))
        default_style = CATEGORY_STYLES.get(category, ["direct"])[0]
        style = str(row.get("style", default_style))
        context = str(row.get("context", "social_media"))
        default_intensity = "high" if category == "hate" else "medium"
        intensity = str(row.get("intensity", default_intensity))
        obf = str(row.get("obf", "none"))
        control = build_control_code(category, style, context, intensity, obf)
        train_texts.append(f"{control} {text} {EOT_TOKEN}")

    if len(train_texts) < 10:
        raise ValueError("Training set is too small. Please build more synthetic samples first.")

    set_seed(random_seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": control_special_tokens()})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))

    requested_modules = target_modules or infer_target_modules(base_model)
    present_leaf_modules = {name.split(".")[-1] for name, _ in model.named_modules()}
    modules = [m for m in requested_modules if m in present_leaf_modules]
    if not modules:
        raise ValueError(
            f"None of the target LoRA modules were found in model `{base_model}`. "
            f"Requested={requested_modules}"
        )
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds = Dataset.from_dict({"text": train_texts})

    def tok(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = ds.map(tok, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    metadata = {
        "base_model": base_model,
        "train_file": str(train_file),
        "train_size": len(train_texts),
        "max_length": max_length,
        "epochs": num_train_epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "learning_rate": learning_rate,
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": modules,
        },
        "control_schema": {
            "category": CATEGORIES,
            "style": STYLES,
            "context": CONTEXTS,
            "intensity": INTENSITIES,
            "obf": OBFUSCATORS,
        },
    }
    write_json(output_dir / "training_meta.json", metadata)
    return metadata


def clean_generated_text(text: str) -> str:
    text = CONTROL_RE.sub("", text)
    if EOT_TOKEN in text:
        text = text.split(EOT_TOKEN, 1)[0]
    text = text.replace("<|endoftext|>", " ")
    text = text.split("\n")[0]
    return strip_bracket_placeholders(text)


def fallback_from_control(row: Dict[str, str], rng: random.Random) -> str:
    category = row["category"]
    style = row["style"]
    context = row["context"]
    intensity = row["intensity"]
    obf = row["obf"]

    template = rng.choice(ATTACK_TEMPLATES[category][style])
    text = template.format(
        judgement=rng.choice(JUDGEMENTS),
        context_topic=CONTEXT_TOPICS[context],
        seed_hint="same pattern repeating",
        target_group=rng.choice(TARGET_GROUPS),
    )
    text = apply_intensity(paraphrase_text(text, rng), intensity, rng)
    return apply_obfuscator(text, obf, intensity, rng)


def build_generation_plan(num_samples: int, random_seed: int) -> List[Dict[str, str]]:
    rng = random.Random(random_seed)
    combos: List[Dict[str, str]] = []
    for category in CATEGORIES:
        for style in CATEGORY_STYLES[category]:
            for context in CONTEXTS:
                for intensity in INTENSITIES:
                    for obf in OBFUSCATORS:
                        combos.append(
                            {
                                "category": category,
                                "style": style,
                                "context": context,
                                "intensity": intensity,
                                "obf": obf,
                            }
                        )
    rng.shuffle(combos)
    out: List[Dict[str, str]] = []
    for i in range(num_samples):
        out.append(combos[i % len(combos)])
    return out


def load_generation_plan(path: Path) -> List[Dict[str, str]]:
    rows = read_jsonl(path)
    plan: List[Dict[str, str]] = []
    for row in rows:
        category = str(row.get("category", "offensive"))
        style = str(row.get("style", "direct"))
        context = str(row.get("context", "social_media"))
        intensity = str(row.get("intensity", "medium"))
        obf = str(row.get("obf", "none"))
        if category not in CATEGORIES:
            continue
        if style not in STYLES:
            continue
        if context not in CONTEXTS:
            continue
        if intensity not in INTENSITIES:
            continue
        if obf not in OBFUSCATORS:
            continue
        plan.append(
            {
                "category": category,
                "style": style,
                "context": context,
                "intensity": intensity,
                "obf": obf,
            }
        )
    return plan


def generate_with_lora(
    base_model: str,
    adapter_dir: Path,
    tokenizer_dir: Optional[Path],
    plan: Sequence[Dict[str, str]],
    out_file: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    random_seed: int,
    progress_every: int,
) -> List[Dict[str, Any]]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise ImportError("Missing dependencies for generation. Install requirements.txt.") from exc

    rng = random.Random(random_seed)
    set_seed(random_seed)

    tok_source = str(tokenizer_dir) if tokenizer_dir else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model)
    # Keep embedding size aligned with the tokenizer used during LoRA training.
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    out_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    total = len(plan)
    with out_file.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(plan):
            control = build_control_code(
                row["category"],
                row["style"],
                row["context"],
                row["intensity"],
                row["obf"],
            )
            prompt = control + " "

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = generated[0][inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            text = clean_generated_text(text)
            if len(tokenize_words(text)) < 4:
                text = fallback_from_control(row, rng)

            out_row = {
                "id": idx + 1,
                "category": row["category"],
                "style": row["style"],
                "context": row["context"],
                "intensity": row["intensity"],
                "obf": row["obf"],
                "prompt": control,
                "text": text,
            }
            rows.append(out_row)
            f.write(json.dumps(out_row, ensure_ascii=True) + "\n")
            f.flush()

            done = idx + 1
            if progress_every > 0 and (done % progress_every == 0 or done == total):
                elapsed = max(1e-6, time.time() - started_at)
                rate = done / elapsed
                print(f"[generate] {done}/{total} done ({rate:.2f} rows/s)")

    return rows


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Sequence[str], n: int) -> float:
    uniq = set()
    total = 0
    for text in texts:
        toks = tokenize_words(text)
        grams = ngrams(toks, n)
        if not grams:
            continue
        total += len(grams)
        uniq.update(grams)
    return float(len(uniq) / total) if total else 0.0


def bleu_score(candidate: Sequence[str], references: Sequence[Sequence[str]], max_n: int = 4) -> float:
    if not candidate:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        cand_counts = Counter(ngrams(candidate, n))
        total = sum(cand_counts.values())
        if total == 0:
            precisions.append(1.0)
            continue

        max_ref_counts: Counter = Counter()
        for ref in references:
            ref_counts = Counter(ngrams(ref, n))
            for gram, cnt in ref_counts.items():
                if cnt > max_ref_counts[gram]:
                    max_ref_counts[gram] = cnt

        overlap = 0
        for gram, cnt in cand_counts.items():
            overlap += min(cnt, max_ref_counts.get(gram, 0))

        precisions.append((overlap + 1.0) / (total + 1.0))

    cand_len = len(candidate)
    ref_lens = [len(r) for r in references if r]
    if not ref_lens:
        return 0.0
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - cand_len), x))
    if cand_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (closest_ref_len / max(cand_len, 1)))

    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return float(score)


def self_bleu(texts: Sequence[str], random_seed: int, ref_sample_size: int) -> float:
    if len(texts) < 2:
        return 0.0
    rng = random.Random(random_seed)
    tokenized = [tokenize_words(t) for t in texts]

    scores: List[float] = []
    for i, cand in enumerate(tokenized):
        refs = tokenized[:i] + tokenized[i + 1 :]
        if not refs:
            continue
        if len(refs) > ref_sample_size:
            refs = rng.sample(refs, ref_sample_size)
        scores.append(bleu_score(cand, refs))

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def embedding_diversity(
    texts: Sequence[str],
    model_name: str,
    random_seed: int,
    max_samples: int,
) -> Dict[str, Any]:
    if len(texts) < 2:
        return {
            "mean_cosine_distance": 0.0,
            "backend": "none",
            "sample_size": len(texts),
        }

    rng = random.Random(random_seed)
    if len(texts) > max_samples:
        sampled = rng.sample(list(texts), max_samples)
    else:
        sampled = list(texts)

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        emb = model.encode(sampled, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        sims = emb @ emb.T
        tri = np.triu_indices(len(sampled), k=1)
        dists = 1.0 - sims[tri]
        return {
            "mean_cosine_distance": float(dists.mean()) if len(dists) else 0.0,
            "backend": f"sentence-transformers:{model_name}",
            "sample_size": len(sampled),
        }
    except Exception:
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            matrix = vec.fit_transform(sampled)
            sims = cosine_similarity(matrix)
            tri = np.triu_indices(len(sampled), k=1)
            dists = 1.0 - sims[tri]
            return {
                "mean_cosine_distance": float(dists.mean()) if len(dists) else 0.0,
                "backend": "tfidf-fallback",
                "sample_size": len(sampled),
            }
        except Exception as exc:
            return {
                "mean_cosine_distance": None,
                "backend": "unavailable",
                "sample_size": len(sampled),
                "error": str(exc),
            }


def distribution(rows: Sequence[Dict[str, Any]], key: str, values: Sequence[str]) -> Dict[str, Any]:
    counts = Counter(str(row.get(key, "")) for row in rows)
    total = max(1, len(rows))
    by_value: Dict[str, Dict[str, Any]] = {}
    for value in values:
        count = counts.get(value, 0)
        by_value[value] = {
            "count": count,
            "share": round(count / total, 4),
        }
    return {
        "total": len(rows),
        "by_value": by_value,
    }


def detect_mechanism(text: str, mechanism: str) -> bool:
    if mechanism == "leet":
        return bool(re.search(r"[430157]", text))
    if mechanism == "vowel_drop":
        words = re.findall(r"[A-Za-z]{4,}", text)
        for w in words:
            if not re.search(r"[aeiouAEIOU]", w):
                return True
        return False
    if mechanism == "char_repeat":
        return bool(re.search(r"(.)\1{2,}", text))
    if mechanism == "separator_noise":
        return bool(re.search(r"[A-Za-z](?:[._-])[A-Za-z](?:[._-])[A-Za-z]", text))
    if mechanism == "none":
        return True
    return False


def evaluate_rows(
    rows: Sequence[Dict[str, Any]],
    min_count_per_value: int,
    self_bleu_ref_size: int,
    embedding_model: str,
    embedding_sample_size: int,
    random_seed: int,
) -> Dict[str, Any]:
    texts = [str(row.get("text", "")) for row in rows if str(row.get("text", "")).strip()]

    coverage = {
        "category": distribution(rows, "category", CATEGORIES),
        "style": distribution(rows, "style", STYLES),
        "context": distribution(rows, "context", CONTEXTS),
        "intensity": distribution(rows, "intensity", INTENSITIES),
        "obf": distribution(rows, "obf", OBFUSCATORS),
    }

    coverage_gate: Dict[str, Any] = {}
    for key, block in coverage.items():
        status = {}
        for value, val_block in block["by_value"].items():
            status[value] = val_block["count"] >= min_count_per_value
        coverage_gate[key] = status

    mechanism_details = {}
    mechanism_hits = []
    for mechanism in OBFUSCATORS:
        target_rows = [r for r in rows if str(r.get("obf", "none")) == mechanism]
        detected = sum(1 for r in target_rows if detect_mechanism(str(r.get("text", "")), mechanism))
        total = len(target_rows)
        hit_rate = (detected / total) if total else 0.0
        mechanism_details[mechanism] = {
            "requested_rows": total,
            "detected_rows": detected,
            "hit_rate": round(hit_rate, 4),
        }
        if mechanism != "none":
            mechanism_hits.append(hit_rate)

    diversity = {
        "distinct_1": round(distinct_n(texts, 1), 4),
        "distinct_2": round(distinct_n(texts, 2), 4),
        "distinct_3": round(distinct_n(texts, 3), 4),
        "self_bleu": round(self_bleu(texts, random_seed, self_bleu_ref_size), 4),
        "embedding_diversity": embedding_diversity(texts, embedding_model, random_seed, embedding_sample_size),
    }

    return {
        "meta": {
            "total_rows": len(rows),
            "min_count_per_value": min_count_per_value,
        },
        "coverage": coverage,
        "coverage_gate": coverage_gate,
        "mechanism_coverage": {
            "by_mechanism": mechanism_details,
            "average_hit_rate": round(sum(mechanism_hits) / len(mechanism_hits), 4) if mechanism_hits else 0.0,
        },
        "diversity": diversity,
    }


def load_generated_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("rows", "outputs", "generated", "data"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    raise ValueError(f"Unsupported generated file format: {path}")


def summarize_seed(rows: Sequence[SeedRow]) -> Dict[str, Any]:
    by_cat = Counter(row.category for row in rows)
    return {
        "total": len(rows),
        "category_counts": dict(by_cat),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total": len(rows),
        "category_counts": dict(Counter(str(r.get("category", "")) for r in rows)),
        "style_counts": dict(Counter(str(r.get("style", "")) for r in rows)),
        "context_counts": dict(Counter(str(r.get("context", "")) for r in rows)),
        "intensity_counts": dict(Counter(str(r.get("intensity", "")) for r in rows)),
        "obf_counts": dict(Counter(str(r.get("obf", "")) for r in rows)),
    }


def parse_target_modules(raw: str) -> Optional[List[str]]:
    raw = raw.strip()
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def command_build_seed(args: argparse.Namespace) -> None:
    seeds = fetch_hatexplain_seed(
        source=args.source,
        local_json_path=args.local_json_path,
        max_per_label=args.max_per_label,
        random_seed=args.random_seed,
    )
    rows = [
        {
            "id": seed.row_id,
            "source": seed.source,
            "category": seed.category,
            "text": seed.text,
        }
        for seed in seeds
    ]
    write_jsonl(Path(args.out_seed), rows)
    print(f"Wrote seed file: {args.out_seed}")
    print(json.dumps(summarize_seed(seeds), indent=2, ensure_ascii=True))


def command_synthesize(args: argparse.Namespace) -> None:
    seeds = to_seed_rows(Path(args.seed_file))
    rows = synthesize_dataset(
        seeds=seeds,
        samples_per_seed=args.samples_per_seed,
        random_seed=args.random_seed,
        include_seed_rows=args.include_seed_rows,
        max_total=args.max_total,
    )
    write_jsonl(Path(args.out_file), rows)
    summary = summarize_rows(rows)
    write_json(Path(args.summary_file), summary)
    print(f"Wrote synthesized dataset: {args.out_file}")
    print(f"Wrote synthesis summary: {args.summary_file}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def command_train_lora(args: argparse.Namespace) -> None:
    metadata = train_lora(
        train_file=Path(args.train_file),
        base_model=args.base_model,
        output_dir=Path(args.output_dir),
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.target_modules),
        random_seed=args.random_seed,
    )
    print(f"LoRA training done. Artifacts: {args.output_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


def command_generate(args: argparse.Namespace) -> None:
    if args.plan_file:
        plan = load_generation_plan(Path(args.plan_file))
    else:
        plan = build_generation_plan(num_samples=args.num_samples, random_seed=args.random_seed)

    if not plan:
        raise ValueError("No valid generation plan found.")

    rows = generate_with_lora(
        base_model=args.base_model,
        adapter_dir=Path(args.adapter_dir),
        tokenizer_dir=Path(args.tokenizer_dir) if args.tokenizer_dir else None,
        plan=plan,
        out_file=Path(args.out_file),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        random_seed=args.random_seed,
        progress_every=args.progress_every,
    )
    summary = summarize_rows(rows)
    write_json(Path(args.summary_file), summary)
    print(f"Wrote generations: {args.out_file}")
    print(f"Wrote generation summary: {args.summary_file}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def command_evaluate(args: argparse.Namespace) -> None:
    rows = load_generated_rows(Path(args.generated_file))
    report = evaluate_rows(
        rows=rows,
        min_count_per_value=args.min_count_per_value,
        self_bleu_ref_size=args.self_bleu_ref_size,
        embedding_model=args.embedding_model,
        embedding_sample_size=args.embedding_sample_size,
        random_seed=args.random_seed,
    )
    write_json(Path(args.out_report), report)
    print(f"Wrote evaluation report: {args.out_report}")
    print(json.dumps(report["diversity"], indent=2, ensure_ascii=True))


def command_run_all(args: argparse.Namespace) -> None:
    seeds = fetch_hatexplain_seed(
        source=args.source,
        local_json_path=args.local_json_path,
        max_per_label=args.max_per_label,
        random_seed=args.random_seed,
    )
    seed_rows = [
        {
            "id": seed.row_id,
            "source": seed.source,
            "category": seed.category,
            "text": seed.text,
        }
        for seed in seeds
    ]
    write_jsonl(Path(args.out_seed), seed_rows)
    print(f"[run-all] seed -> {args.out_seed}")

    synth_rows = synthesize_dataset(
        seeds=seeds,
        samples_per_seed=args.samples_per_seed,
        random_seed=args.random_seed,
        include_seed_rows=args.include_seed_rows,
        max_total=args.max_total,
    )
    write_jsonl(Path(args.synth_out_file), synth_rows)
    synth_summary = summarize_rows(synth_rows)
    write_json(Path(args.synth_summary_file), synth_summary)
    print(f"[run-all] synth -> {args.synth_out_file}")

    train_file = Path(args.train_file) if str(args.train_file).strip() else Path(args.out_seed)
    train_meta = train_lora(
        train_file=train_file,
        base_model=args.base_model,
        output_dir=Path(args.output_dir),
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.target_modules),
        random_seed=args.random_seed,
    )
    print(f"[run-all] train-lora ({train_file}) -> {args.output_dir}")

    if args.plan_file:
        plan = load_generation_plan(Path(args.plan_file))
    else:
        plan = build_generation_plan(num_samples=args.num_samples, random_seed=args.random_seed)
    if not plan:
        raise ValueError("No valid generation plan found for run-all.")

    gen_rows = generate_with_lora(
        base_model=args.base_model,
        adapter_dir=Path(args.output_dir) / "adapter",
        tokenizer_dir=Path(args.output_dir) / "tokenizer",
        plan=plan,
        out_file=Path(args.gen_out_file),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        random_seed=args.random_seed,
        progress_every=args.progress_every,
    )
    gen_summary = summarize_rows(gen_rows)
    write_json(Path(args.gen_summary_file), gen_summary)
    print(f"[run-all] generate -> {args.gen_out_file}")

    eval_report = evaluate_rows(
        rows=gen_rows,
        min_count_per_value=args.min_count_per_value,
        self_bleu_ref_size=args.self_bleu_ref_size,
        embedding_model=args.embedding_model,
        embedding_sample_size=args.embedding_sample_size,
        random_seed=args.random_seed,
    )
    write_json(Path(args.eval_out_report), eval_report)
    print(f"[run-all] evaluate -> {args.eval_out_report}")

    run_summary = {
        "seed": summarize_seed(seeds),
        "synthesis": synth_summary,
        "train": train_meta,
        "generation": gen_summary,
        "evaluation": {
            "coverage_gate": eval_report.get("coverage_gate", {}),
            "diversity": eval_report.get("diversity", {}),
            "mechanism_coverage": eval_report.get("mechanism_coverage", {}),
        },
    }
    write_json(Path(args.run_summary_file), run_summary)
    print(f"[run-all] summary -> {args.run_summary_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Red-team generator (HateXplain seed + structured synthesis + LoRA + evaluation)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("build-seed", help="Build seed data from HateXplain")
    p_seed.add_argument("--source", choices=["hf", "local_json"], default="hf")
    p_seed.add_argument("--local_json_path", default="")
    p_seed.add_argument("--max_per_label", type=int, default=3000)
    p_seed.add_argument("--random_seed", type=int, default=42)
    p_seed.add_argument("--out_seed", default="data/hatexplain_seed.jsonl")
    p_seed.set_defaults(func=command_build_seed)

    p_syn = sub.add_parser("synthesize", help="Build structured synthetic training data")
    p_syn.add_argument("--seed_file", default="data/hatexplain_seed.jsonl")
    p_syn.add_argument("--samples_per_seed", type=int, default=6)
    p_syn.add_argument("--max_total", type=int, default=12000)
    p_syn.add_argument("--include_seed_rows", action="store_true")
    p_syn.add_argument("--random_seed", type=int, default=42)
    p_syn.add_argument("--out_file", default="data/structured_synth.jsonl")
    p_syn.add_argument("--summary_file", default="outputs/synth_summary.json")
    p_syn.set_defaults(func=command_synthesize)

    p_train = sub.add_parser("train-lora", help="Train decoder-only conditional generator with LoRA")
    p_train.add_argument("--train_file", default="data/hatexplain_seed.jsonl")
    p_train.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    p_train.add_argument("--output_dir", default="artifacts/lora_redteam")
    p_train.add_argument("--max_length", type=int, default=192)
    p_train.add_argument("--num_train_epochs", type=float, default=1.0)
    p_train.add_argument("--batch_size", type=int, default=2)
    p_train.add_argument("--grad_accum_steps", type=int, default=8)
    p_train.add_argument("--learning_rate", type=float, default=2e-4)
    p_train.add_argument("--lora_r", type=int, default=16)
    p_train.add_argument("--lora_alpha", type=int, default=32)
    p_train.add_argument("--lora_dropout", type=float, default=0.05)
    p_train.add_argument(
        "--target_modules",
        default="",
        help="Comma-separated target modules; auto-inferred if empty.",
    )
    p_train.add_argument("--random_seed", type=int, default=42)
    p_train.set_defaults(func=command_train_lora)

    p_gen = sub.add_parser("generate", help="Generate with trained LoRA adapter")
    p_gen.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    p_gen.add_argument("--adapter_dir", default="artifacts/lora_redteam/adapter")
    p_gen.add_argument("--tokenizer_dir", default="artifacts/lora_redteam/tokenizer")
    p_gen.add_argument("--plan_file", default="")
    p_gen.add_argument("--num_samples", type=int, default=600)
    p_gen.add_argument("--max_new_tokens", type=int, default=72)
    p_gen.add_argument("--temperature", type=float, default=0.35)
    p_gen.add_argument("--top_p", type=float, default=0.75)
    p_gen.add_argument("--progress_every", type=int, default=10)
    p_gen.add_argument("--random_seed", type=int, default=42)
    p_gen.add_argument("--out_file", default="outputs/generated_outputs.jsonl")
    p_gen.add_argument("--summary_file", default="outputs/generation_summary.json")
    p_gen.set_defaults(func=command_generate)

    p_eval = sub.add_parser("evaluate", help="Evaluate coverage and diversity")
    p_eval.add_argument("--generated_file", default="outputs/generated_outputs.jsonl")
    p_eval.add_argument("--min_count_per_value", type=int, default=20)
    p_eval.add_argument("--self_bleu_ref_size", type=int, default=32)
    p_eval.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_eval.add_argument("--embedding_sample_size", type=int, default=1200)
    p_eval.add_argument("--random_seed", type=int, default=42)
    p_eval.add_argument("--out_report", default="outputs/eval_report.json")
    p_eval.set_defaults(func=command_evaluate)

    p_all = sub.add_parser("run-all", help="Run full pipeline: seed -> synthesize -> train -> generate -> evaluate")
    p_all.add_argument("--source", choices=["hf", "local_json"], default="hf")
    p_all.add_argument("--local_json_path", default="")
    p_all.add_argument("--max_per_label", type=int, default=3000)
    p_all.add_argument("--out_seed", default="data/hatexplain_seed.jsonl")
    p_all.add_argument("--samples_per_seed", type=int, default=6)
    p_all.add_argument("--max_total", type=int, default=12000)
    p_all.add_argument("--include_seed_rows", action="store_true")
    p_all.add_argument("--synth_out_file", default="data/structured_synth.jsonl")
    p_all.add_argument("--synth_summary_file", default="outputs/synth_summary.json")
    p_all.add_argument(
        "--train_file",
        default="",
        help="Training file path. If empty, run-all trains directly from --out_seed.",
    )
    p_all.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    p_all.add_argument("--output_dir", default="artifacts/lora_redteam")
    p_all.add_argument("--max_length", type=int, default=192)
    p_all.add_argument("--num_train_epochs", type=float, default=1.0)
    p_all.add_argument("--batch_size", type=int, default=2)
    p_all.add_argument("--grad_accum_steps", type=int, default=8)
    p_all.add_argument("--learning_rate", type=float, default=2e-4)
    p_all.add_argument("--lora_r", type=int, default=16)
    p_all.add_argument("--lora_alpha", type=int, default=32)
    p_all.add_argument("--lora_dropout", type=float, default=0.05)
    p_all.add_argument("--target_modules", default="")
    p_all.add_argument("--plan_file", default="")
    p_all.add_argument("--num_samples", type=int, default=600)
    p_all.add_argument("--max_new_tokens", type=int, default=72)
    p_all.add_argument("--temperature", type=float, default=0.35)
    p_all.add_argument("--top_p", type=float, default=0.75)
    p_all.add_argument("--progress_every", type=int, default=10)
    p_all.add_argument("--gen_out_file", default="outputs/generated_outputs.jsonl")
    p_all.add_argument("--gen_summary_file", default="outputs/generation_summary.json")
    p_all.add_argument("--min_count_per_value", type=int, default=20)
    p_all.add_argument("--self_bleu_ref_size", type=int, default=32)
    p_all.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_all.add_argument("--embedding_sample_size", type=int, default=1200)
    p_all.add_argument("--eval_out_report", default="outputs/eval_report.json")
    p_all.add_argument("--run_summary_file", default="outputs/run_all_summary.json")
    p_all.add_argument("--random_seed", type=int, default=42)
    p_all.set_defaults(func=command_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
