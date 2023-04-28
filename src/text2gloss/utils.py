import re
from typing import Dict, Optional

import requests
from requests import Session


def standardize_gloss(gloss: str) -> str:
    gloss = re.sub(r"-[A-Z]$", "", gloss)
    gloss = re.sub(r"\([^)]+\)$", "", gloss)
    return gloss


def send_request(endpoint: str, session: Optional[Session] = None, port: int = 5000, params: Optional[Dict] = None):
    if session is None:
        response = requests.get(rf"http://127.0.0.1:{port}/{endpoint}/", params=params)
    else:
        response = session.get(rf"http://127.0.0.1:{port}/{endpoint}/", params=params)

    if response:
        return response.json()
    else:
        return None
