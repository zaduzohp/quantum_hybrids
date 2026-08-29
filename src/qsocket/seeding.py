"""Seeding contract — pairing of delta = acc(A) - acc(B) depends on it.

Uses blake2b rather than the builtin hash(), which is randomised across processes.
Socket and head init keys exclude the arm, so arms differ only in socket contents;
batch order excludes arm, R and dilution, so every arm sees the same sample order.
Every stream, its key and the resulting integer are recorded in the results row.

Streams and their keys:

    quantum socket init   derive(seed, ansatz_level, R)   no arm
    head init             derive(seed, dilution)          no arm
    arm D projection      derive(seed, "D", scale)
    batch order           derive(seed)                    no arm, R or dilution
    dataset shuffle       derive(dataset_seed, "shuffle")
"""

from __future__ import annotations

import hashlib

DIGEST_BYTES = 8


def derive(*parts) -> int:
    """Deterministic seed, stable across processes, runs and Python versions.

    Keyed on repr(parts), so part order matters and derive(1, "a") differs from
    derive((1, "a")).
    """
    digest = hashlib.blake2b(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:DIGEST_BYTES], "big")
