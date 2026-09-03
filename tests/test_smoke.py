"""Environment smoke test: every module imports and the installed versions fall inside the pinned ranges."""

from packaging.version import Version


def test_qsocket_imports():
    import qsocket

    assert qsocket.__version__ == "0.1.0"


def test_qiskit_version_in_contract_range():
    import qiskit

    v = Version(qiskit.__version__)
    assert Version("2.0") <= v < Version("3.0"), qiskit.__version__


def test_qiskit_machine_learning_version_in_contract_range():
    import qiskit_machine_learning

    v = Version(qiskit_machine_learning.__version__)
    assert Version("0.9") <= v < Version("0.10"), qiskit_machine_learning.__version__


def test_torch_imports_and_reports_version():
    import torch

    assert Version(torch.__version__.split("+")[0]) >= Version("2.0"), torch.__version__


def test_contract_dependencies_importable():
    """The remaining declared dependencies — no version ranges pinned for these."""
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import qiskit_algorithms  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    import statsmodels  # noqa: F401
    from iqm.qiskit_iqm import IQMProvider  # noqa: F401


def test_all_stub_modules_importable():
    """Every module of the planned package layout exists and imports."""
    import importlib

    modules = [
        "qsocket.ansatzes",
        "qsocket.core",
        "qsocket.hardware",
        "qsocket.hardware",
        "qsocket.socket",
        "qsocket.head",
        "qsocket.training",
        "qsocket.datasets",
        "qsocket.gates",
        "qsocket.core",
        "qsocket.results",
        "qsocket.vendored",
        "qsocket.vendored.counts",
        "qsocket.vendored.circuit_stats",
        "qsocket.vendored.entanglement",
        "qsocket.vendored.expressibility",
        "qsocket.vendored.metrics_cls",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        assert mod.__doc__, f"{name} has no docstring"
