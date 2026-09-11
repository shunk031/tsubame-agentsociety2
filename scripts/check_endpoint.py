#!/usr/bin/env python3
"""Check that the local vLLM endpoint behaves the way a run depends on.

Reads ENDPOINT, MODEL and ENABLE_THINKING from the environment. Prints every
finding rather than stopping at the first, then exits non-zero if any matter.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# Thinking models spend their first few hundred tokens inside the reasoning
# block. Asking for 32 truncates mid-thought, and the reasoning parser emits
# nothing for an unterminated block, so both content and reasoning_content come
# back empty and the endpoint looks broken when it is merely cut short.
MAX_TOKENS_THINKING = 1024
MAX_TOKENS_PLAIN = 32


def _get(endpoint: str, path: str) -> dict:
    with urllib.request.urlopen(f"{endpoint}{path}", timeout=30) as response:
        return json.load(response)


def _post(endpoint: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{endpoint}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> int:
    endpoint = os.environ["ENDPOINT"]
    model = os.environ["MODEL"]
    thinking = os.environ.get("ENABLE_THINKING", "1") != "0"

    max_tokens = MAX_TOKENS_THINKING if thinking else MAX_TOKENS_PLAIN
    print(f"thinking: {thinking} (max_tokens {max_tokens})")

    failures = []

    served = [entry["id"] for entry in _get(endpoint, "/v1/models")["data"]]
    print(f"served models: {served}")
    if model not in served:
        failures.append(f"{model} is not served; got {served}")

    result = _post(
        endpoint,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Reply with exactly one word: ready"}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
    )

    choice = result["choices"][0]
    message = choice["message"]
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    finish_reason = choice.get("finish_reason")

    print(f"finish_reason: {finish_reason}")
    print(f"content:       {content!r}")
    print(f"reasoning:     {reasoning[:200]!r}{'...' if len(reasoning) > 200 else ''}")
    print(f"usage:         {result.get('usage', {})}")

    if not content:
        if finish_reason == "length":
            failures.append(
                f"the reply hit the {max_tokens}-token cap before producing any "
                "content; raise MAX_TOKENS_THINKING or set ENABLE_THINKING=0"
            )
        else:
            failures.append("chat completion returned empty content")

    # See vLLM issue #35574: the chat-template kwarg has been reported as
    # ignored, which would show up here as a reasoning trace that was asked to
    # be absent.
    if not thinking and reasoning:
        failures.append(
            "thinking mode is still active; enable_thinking=false did not take "
            "effect, so output tokens will be inflated"
        )

    # The inverse is worth knowing too, though it is not fatal: a model that
    # ignores a request to reason will quietly behave differently from the
    # experiment's intent.
    if thinking and not reasoning:
        print(
            "note: thinking was requested but no reasoning trace came back; "
            "this model may not support it",
            file=sys.stderr,
        )

    failures.extend(_check_tool_choice_auto(endpoint, model, max_tokens))

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1

    print("all endpoint checks passed")
    return 0


def _check_tool_choice_auto(endpoint: str, model: str, max_tokens: int) -> list[str]:
    """Send the request shape AgentSociety uses for its agent steps.

    Every agent step goes out with tool_choice="auto". A server started without
    --enable-auto-tool-choice rejects those with a 400, litellm retries, the
    retries fail identically, and the run finishes with a thin replay and no
    obvious cause. Reproducing the shape here surfaces it in seconds.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }

    try:
        result = _post(endpoint, "/v1/chat/completions", payload)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:400]
        return [
            'tool_choice="auto" was rejected, which is how AgentSociety calls '
            f"the model on every agent step: HTTP {error.code} {detail}"
        ]

    message = result["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    print(f"tool_calls:    {len(tool_calls)}")
    if tool_calls:
        print(f"  first:       {tool_calls[0].get('function', {}).get('name')}")
    else:
        # Not fatal. The model is free to answer directly, and the point of the
        # check is that the request was accepted at all.
        print("  none returned; the request was accepted, which is what matters")

    return []


if __name__ == "__main__":
    sys.exit(main())
