"""Canonical identities shared by pure kernels, profiles and artifact producers."""

import hashlib
import json


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def content_id(value):
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()
