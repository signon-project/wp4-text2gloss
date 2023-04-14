import re


def standardize_gloss(gloss: str) -> str:
    gloss = re.sub(r"-[A-Z]$", "", gloss)
    gloss = re.sub(r"\([^)]+\)$", "", gloss)
    return gloss

