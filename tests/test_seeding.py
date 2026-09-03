"""derive must be stable across processes — the paired design rests on it."""

from __future__ import annotations

import subprocess
import sys

from qsocket.core import derive


def test_derive_is_deterministic_in_process():
    assert derive(1, "L1", 2) == derive(1, "L1", 2)


def test_derive_separates_keys():
    assert derive(1, "L1", 2) != derive(1, "L2", 2)
    assert derive(1, "L1", 2) != derive(1, "L1", 3)
    assert derive(1, "L1", 2) != derive(2, "L1", 2)
    assert derive("a", 1) != derive(1, "a")


def test_derive_is_a_64_bit_unsigned_int():
    value = derive(7, "linear")
    assert isinstance(value, int)
    assert 0 <= value < 2**64


def _derive_in_subprocess(expression: str) -> int:
    """Fresh interpreter: the builtin hash() would disagree with itself here."""
    completed = subprocess.run(
        [sys.executable, "-c", f"from qsocket.core import derive; print({expression})"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(completed.stdout.strip())


def test_derive_matches_across_two_separate_processes():
    expression = 'derive(3, "L2", 2)'
    first = _derive_in_subprocess(expression)
    second = _derive_in_subprocess(expression)
    assert first == second == derive(3, "L2", 2)
