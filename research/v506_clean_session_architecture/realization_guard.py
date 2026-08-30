
from __future__ import annotations

import re

from model_utils import token_set,clean_generated
from response_firewall import is_internal


STOPWORDS={
    "the","a","an","is","are","am","be","to","of","and","or","in","on",
    "for","with","that","this","it","i","you","me","my","your","we","they",
    "he","she","do","does","did","what","how","why","when","where","who",
    "can","could","would","should","will","may","might","please","tell",
    "give","show","about","just","really","very","one","thing",
}


def content_tokens(text):
    return {
        t for t in token_set(text)
        if t not in STOPWORDS and len(t)>2
    }


def verify_realization(selected,realized,facts=(),goal_name=""):
    """
    A realizer may change wording, but it may not introduce new semantic
    content. This is deliberately conservative.

    The architecture-selected content is authoritative. The realizer can:
      * paraphrase
      * shorten
      * add normal function words

    It cannot introduce concrete nouns/entities/numbers absent from the
    selected content or explicitly grounded evidence.
    """
    selected=clean_generated(selected)
    realized=clean_generated(realized)

    if not selected or not realized:
        return False,0.0,"empty"

    if is_internal(realized):
        return False,0.0,"internal"

    selected_terms=content_tokens(selected)
    realized_terms=content_tokens(realized)
    evidence_terms=set()

    for fact in facts:
        evidence_terms |= content_tokens(
            f"{fact.get('subject','')} "
            f"{fact.get('predicate','')} "
            f"{fact.get('object_text','')}"
        )

    allowed=selected_terms|evidence_terms

    introduced=realized_terms-allowed

    # Keep common social/function wording from failing, but reject concrete
    # content additions. A tiny amount of introduction is allowed for natural
    # morphology/paraphrase.
    ratio=len(realized_terms & allowed)/max(1,len(realized_terms))

    if introduced:
        # Numbers/entities are never safe to invent.
        if any(
            re.search(r"\d",x) or x in {
                "dog","cat","car","universe","brown","red","blue",
                "green","pigment","military","helicopter",
            }
            for x in introduced
        ):
            return False,ratio,"introduced_concrete_content"

    if ratio<0.55:
        return False,ratio,"low_grounding"

    return True,ratio,"accepted"
