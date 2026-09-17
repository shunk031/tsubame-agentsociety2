#!/usr/bin/env python3
"""Measure what the codegen template cache actually sees.

Why this exists: two 128-agent runs logged zero template-cache hits, and eight
in ten misses gave ``below_similarity_threshold`` — including for instruction
strings that were byte-identical to one already cached. The lookup itself is
implemented correctly (``IndexFlatIP`` over L2-normalised vectors, so the score
is a cosine similarity), the embedding server was healthy, and nothing errored.
That leaves the embeddings themselves, which can only be queried from the node
serving them.

Reports the similarity an identical string gets, which must be ~1.0, and the
similarities between the instructions these runs really produced, which is what
decides whether the 0.85 threshold is the wrong number or the wrong lever.
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.request

# Taken verbatim from a run's "Template cache miss" lines, so the numbers below
# describe the traffic that actually missed rather than a synthetic sample.
INSTRUCTIONS = [
    "refresh_feed user_id={user_id} algorithm={algorithm} limit={limit}",
    "refresh_feed user_id={user_id} algorithm={algorithm} limit={limit}",
    "create a post using author_id and content from ctx['variables']",
    "refresh the feed using user_id from ctx['variables']",
    "view_post user_id={user_id} post_id={post_id}",
    "observe social media profile for user {user_id}",
    "search_posts keyword={keyword} tags={tags} limit={limit} sort_by={sort_by}",
]


def embed(endpoint: str, model: str, text: str) -> list[float]:
    payload = json.dumps({"model": model, "input": text}).encode()
    request = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["data"][0]["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def main() -> int:
    endpoint = os.environ["EMBEDDING_ENDPOINT"].rstrip("/") + "/v1/embeddings"
    model = os.environ["EMBEDDING_MODEL"]
    threshold = float(os.environ.get("TEMPLATE_CACHE_THRESHOLD", "0.85"))

    vectors = [embed(endpoint, model, text) for text in INSTRUCTIONS]

    # The one result that decides everything else. An identical string scoring
    # below the threshold would mean no threshold can ever help.
    identical = cosine(vectors[0], vectors[1])
    print(f"identical string   : {identical:.4f}")
    if identical < 0.999:
        print(
            "  the same text does not embed to the same vector; the cache "
            "cannot hit at any threshold",
            file=sys.stderr,
        )

    print(f"\nthreshold in force : {threshold}")
    print("\npairwise cosine similarity:")
    for i in range(len(INSTRUCTIONS)):
        for j in range(i + 1, len(INSTRUCTIONS)):
            sim = cosine(vectors[i], vectors[j])
            verdict = "hit" if sim >= threshold else "miss"
            print(f"  {sim:.4f}  {verdict:4}  {INSTRUCTIONS[i][:38]!r}")
            print(f"                  vs {INSTRUCTIONS[j][:38]!r}")

    return 0 if identical >= 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
