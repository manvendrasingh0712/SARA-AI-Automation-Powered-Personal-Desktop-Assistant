"""
sara.core.intent
Public package API for the deterministic regex intent router. External
code keeps using this exactly as before:

    from sara.core.intent import detect_intent

Internal layout:
    patterns.py  - _INTENT_PATTERNS / _INTENT_GATES, the pattern data table
    engine.py    - pattern compilation + detect_intent() + its LRU cache
"""
from .engine import detect_intent

__all__ = ["detect_intent"]
