"""Code copied verbatim from other repositories.

Sources: the read-only qbanknote repo and the qml-benchmarks project. Rules for
anything added here:
  1. every copied unit carries a header with its source, commit, date and a
     "Changes vs upstream" field,
  2. every copied unit gets its own test pinning its behaviour — a copy loses the
     upstream tests, so without one nothing would catch a regression,
  3. copy verbatim; describe anything you had to change, and never refactor along
     the way.

The dataset generators are deliberately NOT re-exported here — importing through
__init__ is exactly what pulls jax into the upstream package.
"""
