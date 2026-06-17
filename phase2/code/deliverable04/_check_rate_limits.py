#!/usr/bin/env python
"""Diagnostic: report the API key's *actual effective* rate limits via response headers.

Makes one tiny (max_tokens=1) call and prints every `anthropic-ratelimit-*` header. These
reflect the limits applied to THIS key, including any per-workspace override that can cap you
below your org's account tier. Use to confirm tier rather than infer it from 429 behavior.

Usage (needs ANTHROPIC_API_KEY):
  ANTHROPIC_API_KEY=$(security find-generic-password -s nepa-anthropic -w) \
      conda run -n nepa python code/deliverable04/_check_rate_limits.py
"""
import os
import sys

import anthropic

# Rough RPM ladder for sanity-mapping headers -> tier (Anthropic standard tiers; verify in Console).
TIER_BY_RPM = [(50, "Tier 1"), (1000, "Tier 2"), (2000, "Tier 3"), (4000, "Tier 4")]


def _tier_for(rpm: int) -> str:
    for lim, name in TIER_BY_RPM:
        if rpm <= lim:
            return name
    return ">= Tier 4"


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5-20251001"
    client = anthropic.Anthropic()
    try:
        resp = client.messages.with_raw_response.create(
            model=model, max_tokens=1, messages=[{"role": "user", "content": "hi"}])
        h = resp.headers
    except anthropic.APIStatusError as e:
        # Even a 429/billing error carries the ratelimit headers — use them.
        print(f"(call returned {e.status_code}: {type(e).__name__}; reading headers anyway)")
        h = e.response.headers
    print(f"model: {model}")
    print(f"organization: {h.get('anthropic-organization-id', '(none)')}")
    rl = {k: v for k, v in h.items() if k.lower().startswith("anthropic-ratelimit") or k.lower() == "retry-after"}
    for k in sorted(rl):
        print(f"  {k}: {rl[k]}")
    rpm = rl.get("anthropic-ratelimit-requests-limit")
    if rpm:
        print(f"\n=> requests/min = {rpm}  ->  {_tier_for(int(rpm))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
