from enum import Enum
from dataclasses import dataclass
import re

class BlockReason(str, Enum):
    NONE = "none"
    POLITICS = "politics"
    RELIGION = "religion"
    PERSONAL_ADVICE = "personal_advice"

@dataclass
class RuleMatch:
    allowed: bool
    reason: BlockReason = BlockReason.NONE
    details: str = ""

POLITICS_KEYS = [
    r"\b(prim(e|) minister|president|election|party|politic(al|s)|parliament|loksabha|rajyasabha|mp|mla)\b",
]
RELIGION_KEYS = [
    r"\b(hindu(ism|)|muslim|christian(ity|)|sikh(ism|)|buddh(ism|)|jain(ism|)|religion|god|allah|bhagwan|jesus)\b",
]
PERSONAL_ADVICE_KEYS = [
    r"\b(relationship advice|dating|breakup|therapy on my life|should i marry|love advice)\b",
]

def _match_any(text: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        if re.search(pat, text, flags=re.I):
            return pat
    return None

def is_allowed(text: str) -> RuleMatch:
    txt = text.strip()
    if not txt:
        return RuleMatch(False, BlockReason.PERSONAL_ADVICE, "empty_text")

    m = _match_any(txt, POLITICS_KEYS)
    if m: return RuleMatch(False, BlockReason.POLITICS, m)

    m = _match_any(txt, RELIGION_KEYS)
    if m: return RuleMatch(False, BlockReason.RELIGION, m)

    m = _match_any(txt, PERSONAL_ADVICE_KEYS)
    if m: return RuleMatch(False, BlockReason.PERSONAL_ADVICE, m)

    return RuleMatch(True, BlockReason.NONE, "")
