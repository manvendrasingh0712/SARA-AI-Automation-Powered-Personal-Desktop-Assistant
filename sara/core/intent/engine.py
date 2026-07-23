"""
sara.core.intent.engine
Pattern compilation (with the groupless-merge optimization) and the
public detect_intent() entry point, including its LRU cache wrapper.
"""

import re
from functools import lru_cache
from typing import Optional, Tuple

# ── Pattern table ──────────────────────────────────────────────────────
# Each entry: (intent_name, [pattern_strings])
# Order matters: more specific patterns must come before broad fallbacks.
from .patterns import _INTENT_PATTERNS, _INTENT_GATES



def _merge_groupless(patterns):
    """
    Collapse a multi-pattern intent group into a single compiled
    alternation regex when it's provably safe to do so — i.e. when
    NONE of its patterns contain a capturing group. Cuts N separate
    .search() engine invocations down to 1 for pure keyword/phrase
    toggle intents. Intents with any capturing pattern are compiled
    individually, unchanged, so match.group(1) semantics never shift.
    """
    compiled_each = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
    if len(patterns) > 1 and all(c.groups == 0 for c in compiled_each):
        merged = "|".join(f"(?:{p})" for p in patterns)
        return [re.compile(merged, re.IGNORECASE | re.UNICODE)]
    return compiled_each


# Pre-compile all patterns.
# Groupless multi-pattern intents are auto-merged into a single
# alternation regex via _merge_groupless() to cut the number of regex
# engine invocations per detect_intent() call (see docstring above).
# Intents needing capture groups are compiled individually, exactly as
# before — order, patterns, and flags all unchanged from the source
# _INTENT_PATTERNS table.
_COMPILED_PATTERNS = [
    (name, _merge_groupless(patterns))
    for name, patterns in _INTENT_PATTERNS
]

# ── Hot-path route table (Phase 3) ──────────────────────────────────────
# Joins (intent_name, compiled_patterns, gate) into one pre-built tuple so
# the per-call hot loop never has to do a dict lookup
# (_INTENT_GATES.get(...)) while iterating — the gate for each intent
# already sits right next to its compiled patterns. Built once at import
# time from the exact same _INTENT_GATES / _COMPILED_PATTERNS data, so it
# cannot drift out of sync with them and changes no matching behavior
# whatsoever — same names, same patterns, same order, same gates.
_ROUTES = tuple(
    (name, compiled_list, _INTENT_GATES.get(name))
    for name, compiled_list in _COMPILED_PATTERNS
)


def _validate_intent_tables():
    """
    One-time startup sanity checks (debug builds only — stripped when
    Python is run with -O / -OO, so this costs nothing in such a build).
    Verifies every _INTENT_GATES key corresponds to a real intent (catches
    a typo that would silently disable a gate forever) and flags any exact
    duplicate pattern string compiled within the same intent's own
    alternation (a duplicate there is dead/unreachable). Never fires in
    normal operation against the current tables — it exists purely to
    catch a future editing mistake before it ships, and does not alter
    matching behavior in any way.
    """
    intent_names = {name for name, _ in _INTENT_PATTERNS}
    for gated_name in _INTENT_GATES:
        assert gated_name in intent_names, (
            f"_INTENT_GATES has an entry for unknown intent {gated_name!r}"
        )
    for name, patterns in _INTENT_PATTERNS:
        assert len(patterns) == len(set(patterns)), (
            f"duplicate pattern string(s) detected in intent {name!r}"
        )


if __debug__:
    _validate_intent_tables()

# Bounded size for the repeated-command memoization cache below. Voice
# commands repeat often ("what time is it", "open chrome", "play music"),
# but arbitrary "chat" fallback text (ordinary conversation) also gets
# cached and mostly never repeats — bounding the cache keeps memory flat
# over a long-running session instead of growing without limit.
_INTENT_CACHE_SIZE = 256


@lru_cache(maxsize=_INTENT_CACHE_SIZE)
def _detect_intent_cached(text: str) -> Tuple[str, Optional[re.Match]]:
    """
    Core matching routine behind detect_intent(), memoized with an LRU
    cache keyed on the exact (already-stripped) input string.

    Regex matching over an immutable string is a pure function of that
    string's contents, so an identical repeated command can only ever
    produce the same (intent_name, match) result — caching it is safe
    and lets repeats skip regex evaluation over all ~100 pattern groups
    entirely instead of re-running them.
    """
    text_lower = text.lower()

    for intent_name, compiled_list, gate in _ROUTES:
        if gate is not None:
            hit = False
            for kw in gate:
                if kw in text_lower:
                    hit = True
                    break
            if not hit:
                continue
        for pattern in compiled_list:
            match = pattern.search(text)
            if match:
                return intent_name, match

    return "chat", None


def detect_intent(text: str) -> Tuple[str, Optional[re.Match]]:
    """
    Detect the intent of a user command via fast local keyword matching.

    Returns:
        (intent_name, match_object)
        Falls through to ("chat", None) when no intent matches.
    """
    return _detect_intent_cached(text.strip())


# Convenience passthroughs for tests/tools — do not affect matching
# behavior. cache_clear() resets memoized state; cache_info() reports
# hits/misses/maxsize/currsize for observability.
detect_intent.cache_clear = _detect_intent_cached.cache_clear
detect_intent.cache_info = _detect_intent_cached.cache_info