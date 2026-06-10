"""Dialogue script parser.

Supports:
  - single dialogue:           "Hello, my name is Timmy."
  - multi-character scripts:   [SQUIRREL]\nHey elephant!\n\n[ELEPHANT]\nHello!
  - inline form:               SQUIRREL: Hey elephant!
  - timestamps are stripped:   [0:03] Zumboo: "..."  ->  speaker Zumboo
"""
import re

_TAG_RE = re.compile(r"^\s*\[([^\]\d][^\]]*)\]\s*$")          # [SQUIRREL]
_TIME_RE = re.compile(r"^\s*\[\d+[:.]\d+\]\s*")               # [0:03]
_INLINE_RE = re.compile(r"^\s*([A-Za-z][\w \-']{0,30})\s*:\s*(.+)$")  # SQUIRREL: hi


def parse_dialogue(script: str):
    """Returns list of (speaker_or_None, line_text)."""
    lines = []
    current = None
    for raw in script.splitlines():
        raw = _TIME_RE.sub("", raw)          # strip leading [0:03] timestamps
        text = raw.strip()
        if not text:
            continue
        m = _TAG_RE.match(text)
        if m:
            current = m.group(1).strip()
            continue
        m = _INLINE_RE.match(text)
        if m and len(m.group(1).split()) <= 3:
            speaker, rest = m.group(1).strip(), m.group(2).strip()
            lines.append((speaker, _clean(rest)))
            current = speaker
            continue
        lines.append((current, _clean(text)))
    # merge consecutive same-speaker lines into one utterance per source line
    return lines


def _clean(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    return text
