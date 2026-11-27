#!/usr/bin/env python3
"""quiz_tool.py — extract multiple-choice quiz questions from screenshots and
answer them, using the Anthropic Claude API (Python SDK).

Two-stage pipeline:
  1. EXTRACT  (vision model, default claude-haiku-5-5): image + instructions ->
     structured JSON {question_number, question_type, question, options[]}.
  2. ANSWER   (text-only model, default claude-sonnet-5-5): question text +
     options -> {correct_labels[], explanation, confidence} (no image re-sent,
     to save tokens).

Usage:
  python quiz_tool.py <file|folder> [--answer] [--out results.json]
                      [--extract-model ...] [--answer-model ...] [--crop l,t,r,b]

Requires ANTHROPIC_API_KEY in the environment (or a .env file).
Works on Windows and macOS (pure stdlib + anthropic/pydantic/pillow/dotenv).
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field, field_validator

try:
    import anthropic
except ImportError:  # pragma: no cover
    print("anthropic SDK missing — run: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

EXTRACT_MODEL_DEFAULT = "claude-haiku-5-5"
ANSWER_MODEL_DEFAULT = "claude-sonnet-5-5"

# USD per million tokens, used for the end-of-run cost estimate.
PRICES_PER_MTOK = {
    "haiku-5-5": {"in": 0.10, "out": 0.50},
    "sonnet-5-5": {"in": 2.00, "out": 10.00},
    "opus-5-5": {"in": 4.00, "out": 20.00},
}

MAX_LONG_EDGE = 1568          # Anthropic's recommended max for image detail
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
RETRIES = 3                   # exponential backoff on 429/5xx/overloaded

EXTRACT_SYSTEM = """\
You extract exam questions from screenshots of an online quiz.
Copy the question text VERBATIM (fix only obvious OCR errors).
Keep every option with the letter shown on screen; assign A, B, C... if none are labelled.
IGNORE UI elements: buttons (Save, Note, Submit, Next), tags, progress bars, timers, headers, watermarks.
Detect "select two" / "choose all that apply" as multiple_choice_multi_select; true/false statements as true_false.
Reply in strict JSON only."""

ANSWER_SYSTEM = """\
You answer one multiple-choice quiz question from its text and options only (the image is not shown, to save tokens).
Pick the correct option label(s). For multi-select questions return every correct label.
Rely on your own knowledge. Reply in strict JSON only."""


# --------------------------------------------------------------------------
# Pydantic schemas (used with messages.parse structured outputs)
# --------------------------------------------------------------------------

class QuizOption(BaseModel):
    label: str = Field(description="Option letter exactly as shown on screen (A, B, C...)")
    text: str = Field(description="Option text, copied verbatim")


class ExtractedQuestion(BaseModel):
    question_number: Optional[str] = Field(default=None, description="e.g. '3' or '3 of 10', null if absent")
    question_type: Literal["single_choice", "multiple_choice_multi_select", "true_false", "other"]
    question: str
    options: list[QuizOption] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def _assign_missing_labels(cls, v: list[QuizOption]) -> list[QuizOption]:
        if any(not o.label.strip() for o in v):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i, o in enumerate(v):
                if not o.label.strip():
                    o.label = letters[i % 26]
        return v


class AnsweredQuestion(BaseModel):
    correct_labels: list[str] = Field(description="One or more option labels that are correct")
    explanation: str = Field(description="Brief (1-2 sentences)")
    confidence: Literal["high", "medium", "low"]


# --------------------------------------------------------------------------
# Image pre-processing
# --------------------------------------------------------------------------

def prepare_image(path: Path, crop: Optional[tuple[int, int, int, int]] = None) -> tuple[bytes, str]:
    """Open, optionally crop, convert to RGB, downscale (long edge <= 1568 px),
    re-encode as PNG and return (bytes, media_type)."""
    with Image.open(path) as im:
        if crop:
            w, h = im.size
            l, t, r, b = crop
            # Fraction (0..1] trims that share of each edge; larger values are pixels.
            left = round(l * w) if l <= 1 else int(l)
            top = round(t * h) if t <= 1 else int(t)
            right = w - round(r * w) if r <= 1 else w - int(r)
            bottom = h - round(b * h) if b <= 1 else h - int(b)
            left, top = max(0, min(left, w - 1)), max(0, min(top, h - 1))
            right, bottom = max(left + 1, min(right, w)), max(top + 1, min(bottom, h))
            im = im.crop((left, top, right, bottom))
        im = im.convert("RGB")
        scale = MAX_LONG_EDGE / max(im.size)
        if scale < 1:
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
        from io import BytesIO
        buf = BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Anthropic call helpers (retry / usage / parse)
# --------------------------------------------------------------------------

def price_for(model: str) -> dict[str, float]:
    m = model.lower()
    for key, val in PRICES_PER_MTOK.items():
        if key in m:
            return val
    return PRICES_PER_MTOK["sonnet-5-5"]


class UsageTracker:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, model: str, stage: str, usage, path: str) -> None:
        tin = getattr(usage, "input_tokens", 0) or 0
        tout = getattr(usage, "output_tokens", 0) or 0
        p = price_for(model)
        cost = tin / 1e6 * p["in"] + tout / 1e6 * p["out"]
        self.rows.append({"model": model, "stage": stage, "file": path,
                          "input_tokens": tin, "output_tokens": tout, "usd": cost})
        print(f"  [{stage}] {path}: in={tin} out={tout} (~${cost:.4f})", file=sys.stderr)

    def summary(self) -> str:
        tin = sum(r["input_tokens"] for r in self.rows)
        tout = sum(r["output_tokens"] for r in self.rows)
        total = sum(r["usd"] for r in self.rows)
        return (f"Token usage: {tin:,} input / {tout:,} output across "
                f"{len(self.rows)} calls. Estimated cost: ${total:.4f}")


def _is_retryable(err: Exception) -> bool:
    if isinstance(err, anthropic.APIStatusError):
        return err.status_code == 429 or err.status_code >= 500 or "overloaded" in str(err).lower()
    return isinstance(err, anthropic.APIConnectionError)


def call_with_retry(fn, *, what: str):
    """Run fn(attempt) with exponential backoff on 429/5xx/overloaded (max RETRIES)."""
    delay = 1.0
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 — narrowed below
            last = err
            if attempt >= RETRIES or not _is_retryable(err):
                raise
            print(f"  [{what}] retry {attempt + 1}/{RETRIES} after {delay:.0f}s: {err}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise last  # unreachable


def parse_with_max_tokens_retry(client: anthropic.Anthropic, *, model: str, system: str,
                                content: list, output_format, max_tokens: int, what: str):
    """Structured-output call; if stop_reason == 'max_tokens', retry once with a
    higher budget so long option lists are not truncated."""
    for budget in (max_tokens, max_tokens * 2):
        resp = call_with_retry(
            lambda b=budget: client.messages.parse(
                model=model, system=system, messages=[{"role": "user", "content": content}],
                max_tokens=b, output_format=output_format),
            what=what)
        if getattr(resp, "stop_reason", None) != "max_tokens":
            return resp
        print(f"  [{what}] hit max_tokens={budget}, retrying with {budget * 2}", file=sys.stderr)
    return resp


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def extract_question(client: anthropic.Anthropic, model: str, img_b64: str, media_type: str,
                     usage: UsageTracker, path: str) -> ExtractedQuestion:
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
        {"type": "text", "text": "Extract the quiz question from this screenshot."},
    ]
    resp = parse_with_max_tokens_retry(client, model=model, system=EXTRACT_SYSTEM,
                                       content=content, output_format=ExtractedQuestion,
                                       max_tokens=1024, what="extract")
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError("model refused to process this image")
    usage.record(model, "extract", resp.usage, path)
    parsed = resp.parsed_output
    if parsed is None:
        blocks = getattr(resp, "content", None) or []
        kinds = ", ".join(getattr(b, "type", type(b).__name__) for b in blocks) or "empty"
        raise RuntimeError(f"model returned no structured output (stop_reason={getattr(resp, 'stop_reason', '?')}, content blocks: {kinds})")
    return parsed


def answer_question(client: anthropic.Anthropic, model: str, q: ExtractedQuestion,
                    usage: UsageTracker, path: str) -> AnsweredQuestion:
    opts = "\n".join(f"{o.label}. {o.text}" for o in q.options)
    user = (f"Question type: {q.question_type}\n\n{q.question}\n\nOptions:\n{opts}\n\n"
            "Return the correct label(s), a brief explanation and your confidence.")
    resp = parse_with_max_tokens_retry(client, model=model, system=ANSWER_SYSTEM,
                                       content=[{"type": "text", "text": user}],
                                       output_format=AnsweredQuestion,
                                       max_tokens=512, what="answer")
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError("model refused to answer this question")
    usage.record(model, "answer", resp.usage, path)
    parsed = resp.parsed_output
    if parsed is None:
        blocks = getattr(resp, "content", None) or []
        kinds = ", ".join(getattr(b, "type", type(b).__name__) for b in blocks) or "empty"
        raise RuntimeError(f"model returned no structured output (stop_reason={getattr(resp, 'stop_reason', '?')}, content blocks: {kinds})")
    # Case-insensitive validation of returned labels against the options shown.
    valid = {o.label.upper() for o in q.options}
    kept = [l for l in (x.strip().upper() for x in parsed.correct_labels) if l in valid]
    if not kept and parsed.correct_labels:
        print(f"  [answer] warning: model labels {parsed.correct_labels} do not match any option "
              f"({sorted(valid)}); keeping as-is", file=sys.stderr)
    parsed.correct_labels = kept or parsed.correct_labels
    return parsed


# --------------------------------------------------------------------------
# Result cache (SHA-256 of original image bytes)
# --------------------------------------------------------------------------

CACHE_FILE = Path(__file__).resolve().parent / ".quiz_cache.json"


def cache_load() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def cache_save(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as err:
        print(f"warning: could not write cache: {err}", file=sys.stderr)


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------

def write_outputs(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "question", "A", "B", "C", "D", "answer", "confidence"])
        for r in results:
            q = r.get("extraction") or {}
            opts = {o["label"]: o["text"] for o in q.get("options", [])}
            ans = r.get("answer") or {}
            w.writerow([
                r.get("file", ""), q.get("question", r.get("error", "")),
                opts.get("A", ""), opts.get("B", ""), opts.get("C", ""), opts.get("D", ""),
                ",".join(ans.get("correct_labels", [])), ans.get("confidence", ""),
            ])
    print(f"Wrote {out_path} and {csv_path}", file=sys.stderr)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        files = [p for p in target.iterdir() if p.suffix.lower() in IMAGE_TYPES and p.is_file()]
        return sorted(files, key=lambda p: p.name)
    raise SystemExit(f"No such file or folder: {target}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract & answer quiz questions from screenshots (Claude).")
    ap.add_argument("target", help="one screenshot or a folder of screenshots (PNG/JPG/WEBP)")
    ap.add_argument("--answer", action="store_true", help="run stage 2 (answer) as well as extraction")
    ap.add_argument("--out", default="results.json", help="output JSON path (CSV written alongside)")
    ap.add_argument("--extract-model", default=os.environ.get("QUIZ_EXTRACT_MODEL", EXTRACT_MODEL_DEFAULT))
    ap.add_argument("--answer-model", default=os.environ.get("QUIZ_ANSWER_MODEL", ANSWER_MODEL_DEFAULT))
    ap.add_argument("--crop", default=None, metavar="L,T,R,B",
                    help="trim UI margins: pixels (e.g. 0,80,0,120) or fractions of the edge (0..1)")
    ap.add_argument("--no-cache", action="store_true", help="disable the SHA-256 result cache")
    args = ap.parse_args(argv)

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (export it or put it in a .env file).", file=sys.stderr)
        return 2

    crop = None
    if args.crop:
        try:
            crop = tuple(float(x) for x in args.crop.split(","))
            assert len(crop) == 4
        except Exception:
            print("--crop must be four comma-separated numbers L,T,R,B", file=sys.stderr)
            return 2

    images = collect_images(Path(args.target))
    if not images:
        print("No PNG/JPG/WEBP files found.", file=sys.stderr)
        return 2

    client = anthropic.Anthropic()
    usage = UsageTracker()
    cache = {} if args.no_cache else cache_load()
    results: list[dict] = []
    failures = 0

    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}", file=sys.stderr)
        rec: dict = {"file": str(path)}
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            rec["sha256"] = digest
            cached = None if args.no_cache else cache.get(digest)
            if cached and (cached.get("answer") or not args.answer):
                print("  cache hit — skipping API calls", file=sys.stderr)
                results.append({**cached, "cached": True})
                continue

            img_bytes, media_type = prepare_image(path, crop)
            img_b64 = base64.standard_b64encode(img_bytes).decode()
            q = extract_question(client, args.extract_model, img_b64, media_type, usage, path.name)
            rec["extraction"] = q.model_dump()
            if args.answer and q.options:
                a = answer_question(client, args.answer_model, q, usage, path.name)
                rec["answer"] = a.model_dump()
            if not args.no_cache:
                cache[digest] = rec
        except Exception as err:  # noqa: BLE001 — batch continues on single-file failure
            failures += 1
            rec["error"] = f"{type(err).__name__}: {err}"
            print(f"  FAILED, continuing batch: {rec['error']}", file=sys.stderr)
        results.append(rec)

    write_outputs(results, Path(args.out))
    if not args.no_cache:
        cache_save(cache)
    print(usage.summary())
    if failures:
        print(f"{failures} file(s) failed out of {len(images)}.", file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
