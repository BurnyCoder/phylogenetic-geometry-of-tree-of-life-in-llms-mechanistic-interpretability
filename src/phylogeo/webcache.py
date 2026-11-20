"""HTTP helpers with retries and an on-disk JSON cache.

Global context: species selection (Wikidata via QLever), tree building (Open Tree
of Life API) and dating (TimeTree API) all hit public web APIs. Caching every
response under ``DATA_DIR/cache`` makes reruns reproducible and offline-capable,
and keeps the load on those services minimal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import requests

from phylogeo.config import CACHE_DIR

log = logging.getLogger("phylogeo.webcache")
# Generic, non-identifying client name (no personal details in requests).
HEADERS = {"User-Agent": "phylogeo-research/0.1 (academic research script)"}


def _cache_path(namespace: str, key: str) -> Path:
    """Map (namespace, request key) -> cache file path via sha256 of the key."""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    p = CACHE_DIR / namespace
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{h}.json"


def request_json(
    method: str,
    url: str,
    *,
    namespace: str,
    params: dict | None = None,
    payload: dict | None = None,
    retries: int = 5,
    timeout: float = 120.0,
    accept_error_json: bool = False,
):
    """GET/POST ``url`` and return parsed JSON, using the disk cache if present.

    Local: exponential backoff on network errors and 429/5xx. If
    ``accept_error_json`` is True, a 4xx response with a JSON body is returned
    (and cached) instead of raised; Open Tree reports unknown ids that way.
    Global: the single choke point for all web access in the project.
    """
    key = json.dumps([method, url, params, payload], sort_keys=True)
    path = _cache_path(namespace, key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    delay = 2.0
    for attempt in range(retries):
        try:
            r = requests.request(
                method, url, params=params, json=payload, headers=HEADERS, timeout=timeout
            )
            if r.status_code == 200 or (accept_error_json and 400 <= r.status_code < 500):
                data = r.json()
                path.write_text(json.dumps(data), encoding="utf-8")
                return data
            log.warning("HTTP %s from %s (attempt %d)", r.status_code, url, attempt + 1)
        except (requests.RequestException, ValueError) as e:  # network error or bad JSON
            log.warning("request error %s on %s (attempt %d)", e, url, attempt + 1)
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"failed after {retries} attempts: {method} {url}")


def request_text(url: str, *, namespace: str, params: dict | None = None, retries: int = 5) -> str:
    """GET ``url`` and return the body as text (cached); used for CSV SPARQL results."""
    key = json.dumps(["GET", url, params], sort_keys=True)
    path = _cache_path(namespace, key).with_suffix(".txt")
    if path.exists():
        return path.read_text(encoding="utf-8")
    delay = 2.0
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers={**HEADERS, "Accept": "text/csv"}, timeout=300)
            if r.status_code == 200:
                path.write_text(r.text, encoding="utf-8")
                return r.text
            log.warning("HTTP %s from %s (attempt %d)", r.status_code, url, attempt + 1)
        except requests.RequestException as e:
            log.warning("request error %s on %s (attempt %d)", e, url, attempt + 1)
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"failed after {retries} attempts: GET {url}")
