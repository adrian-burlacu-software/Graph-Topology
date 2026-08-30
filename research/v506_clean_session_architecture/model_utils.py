
import re


def token_set(text):
    return {
        x for x in re.findall(r"[a-z0-9']+",str(text or "").lower())
        if len(x)>1
    }


def clean_generated(text):
    text=re.sub(r"\s+"," ",str(text or "").strip())
    text=text.strip("` ")
    return text


def looks_meta(text):
    low=str(text or "").lower()
    return any(x in low for x in (
        "the architecture","candidate proposition",
        "useful candidate","the participant",
        "the user is asking","as an ai",
    ))
