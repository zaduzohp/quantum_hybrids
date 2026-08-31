"""Code copied verbatim from other repositories.

Sources: the read-only qbanknote repo and the qml-benchmarks project. Rules for
anything added here:
  1. every copied unit carries a header with its source, commit, date and a
     "Changes vs upstream" field,
  2. every copied unit gets its own test pinning its behaviour — a copy loses the
     upstream tests, so without one nothing would catch a regression,
  3. copy verbatim; describe anything you had to change, and never refactor along
     the way,
  4. a copy that stops having a caller is DELETED, not kept "in case". Rule 3 keeps a
     copy faithful to upstream; it does not make an unused one worth carrying. The
     Wilcoxon/sign enumeration went that way once qsocket.stats replaced it — its
     oracle now lives in tests/test_exact_tests_equivalence.py, written independently of
     the implementation it checks.

The dataset generators are deliberately NOT re-exported here — importing through
__init__ is exactly what pulls jax into the upstream package.
"""
