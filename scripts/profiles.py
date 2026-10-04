"""Workload profiles — defined once, imported everywhere.

The plan's rule: do not redefine these per experiment. Every ablation from W10 onward
reads these same numbers so its results stay comparable. Change a value here and you
have changed every experiment that follows, so don't tune them to make a run look good.

Token counts are SERVER-REPORTED totals (`usage.prompt_tokens` / `usage.completion_tokens`),
not raw user-message length. For a chat model the prompt total includes the chat template
wrapper, which on Qwen2.5-Instruct is a couple dozen tokens on its own — so `input_tokens`
is not something you can eyeball from the prompt string. Use `loadgen.py --calibrate` to
find the filler-word count that lands on the target.

All values here are synthetic: chosen to match the W0 Session 1 baseline run, not derived
from any production traffic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    input_tokens: int
    output_tokens: int
    note: str = ""


# W1-chat: matches the single request measured in W0 Session 1 (prompt_tokens 33,
# completion_tokens 49), so the first load-test numbers are directly comparable to
# that run. Short on both ends — this profile exercises decode far more than prefill.
W1_CHAT = Profile(
    name="W1-chat",
    input_tokens=33,
    output_tokens=49,
    note="matches W0S1 baseline; short prompt, decode-dominated",
)

# W2-rag / W3-agent: not yet defined. Fill these in before any experiment that claims
# to cover them, and keep them synthetic.
W2_RAG = None
W3_AGENT = None

PROFILES = {p.name: p for p in (W1_CHAT,) if p is not None}


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise SystemExit(
            f"unknown profile {name!r}; defined: {', '.join(sorted(PROFILES)) or '(none)'}"
        )
