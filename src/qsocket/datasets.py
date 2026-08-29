"""Dataset generation and freezing, deterministic, hashed, replayable.

generate_and_freeze runs one fixed order of operations:

    1. np.random.seed(dataset_seed)
    2. ONE call to the generator, arguments by name
    3. shuffle with derive(dataset_seed, "shuffle")
    4. split 4200 train / 600 val / 1200 test
    5. PCA to 5 components, whiten=False, fitted on the 4200 training rows ONLY
    6. transform all three splits with that frozen PCA
    7. scale to FEATURE_RANGE, scaler fitted on the training rows ONLY
    8. write arrays + manifest

Three properties of the vendored generators drive steps 1-3:

  * no seed argument, so reproducibility runs through a global np.random.seed set
    immediately before a single call;
  * generate_two_curves takes offset before noise, unlike the upstream README example,
    so arguments are always passed by name;
  * labels come out sorted, so without the shuffle the split would be class-pure.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from qsocket.encoding import FEATURE_RANGE
from qsocket.seeding import derive
from qsocket.vendored.hidden_manifold import generate_hidden_manifold_model
from qsocket.vendored.hyperplanes import generate_hyperplanes_parity
from qsocket.vendored.two_curves import generate_two_curves

# frozen dataset constants

SPLIT_SIZES: dict[str, int] = {"train": 4200, "val": 600, "test": 1200}
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
N_SAMPLES_TOTAL = sum(SPLIT_SIZES.values())
N_COMPONENTS = 5
PCA_WHITEN = False

# Commit the generators were copied at; the manifest records this rather than
# qml_benchmarks.__version__, because the package is not a dependency.
VENDORED_GENERATORS_COMMIT = "95e5a07e8e9e75ba7e24e67fb32b030112a1309a"

GENERATORS = {
    "two_curves": generate_two_curves,
    "hidden_manifold": generate_hidden_manifold_model,
    "hyperplanes": generate_hyperplanes_parity,
}

# Every generator argument, all required. Listed here rather than taken from the
# generator signature so a knob cannot silently fall back to its default.
REQUIRED_GENERATOR_KWARGS: dict[str, tuple[str, ...]] = {
    "two_curves": ("n_features", "degree", "offset", "noise"),
    "hidden_manifold": ("n_features", "manifold_dimension"),
    "hyperplanes": ("n_features", "n_hyperplanes", "dim_hyperplanes"),
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PRODUCTION_DATASET = "hyperplanes_n_features20_n_hyperplanes3_dim_hyperplanes5_seed11"

RETIRED_TWO_CURVES = "two_curves_n_features20_degree3_offset0p05_noise0p1_seed11"

# name -> (dataset_hash prefix, pca_hash prefix), both from the freezing report.
FROZEN_DATASET_HASH_PREFIXES: dict[str, tuple[str, str]] = {
    PRODUCTION_DATASET: ("43605086", "2bf856a6"),
    RETIRED_TWO_CURVES: ("f758dfd8", "124e24b7"),
}


def expected_hash_prefixes(name: str) -> tuple[str, str]:
    """The (dataset_hash, pca_hash) prefixes a frozen dataset of this name must have."""
    try:
        return FROZEN_DATASET_HASH_PREFIXES[name]
    except KeyError:
        known = ", ".join(sorted(FROZEN_DATASET_HASH_PREFIXES))
        raise KeyError(
            f"{name!r} is not a frozen dataset of this project; known: {known}. "
            "A dataset added to data/ must be registered here with the hashes from its "
            "freezing report before any measurement may run on it."
        ) from None


# Numerical slack on the FEATURE_RANGE assertion.
_RANGE_TOL = 1e-9


# --- hashing ------------------------------------------------------------------------


def _digest(items: list[tuple[str, np.ndarray]]) -> str:
    """SHA-256 over (name, dtype, shape, bytes) of each array, in the given order."""
    hasher = hashlib.sha256()
    for name, array in items:
        array = np.ascontiguousarray(array)
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- paths --------------------------------------------------------------------------


def dataset_paths(name: str, *, out_dir=DEFAULT_DATA_DIR) -> tuple[Path, Path]:
    """(array file, manifest file) for a frozen dataset called `name`."""
    out_dir = Path(out_dir)
    return out_dir / f"{name}.npz", out_dir / f"{name}.manifest.json"


# --- generation ---------------------------------------------------------------------


def _call_generator(name: str, *, n_samples: int, generator_kwargs: dict):
    """One call, arguments by name, after the global seed has been set by the caller."""
    if name not in GENERATORS:
        raise ValueError(f"unknown generator {name!r}; expected one of {sorted(GENERATORS)}")

    required = REQUIRED_GENERATOR_KWARGS[name]
    missing = [key for key in required if key not in generator_kwargs]
    if missing:
        raise ValueError(
            f"generator {name!r} requires {list(required)}; missing {missing}. "
            "These have no defaults on purpose (open decision D-3): a knob that "
            "silently defaults produces a dataset nobody can reconstruct."
        )
    unexpected = sorted(set(generator_kwargs) - set(required))
    if unexpected:
        raise ValueError(f"generator {name!r} got unexpected kwargs {unexpected}")

    return GENERATORS[name](n_samples=n_samples, **{k: generator_kwargs[k] for k in required})


def _split_indices(n_samples: int) -> dict[str, slice]:
    if n_samples != N_SAMPLES_TOTAL:
        raise ValueError(
            f"n_samples must be {N_SAMPLES_TOTAL} to fill the frozen "
            f"{SPLIT_SIZES['train']}/{SPLIT_SIZES['val']}/{SPLIT_SIZES['test']} split, got {n_samples}"
        )
    start = 0
    bounds = {}
    for split in SPLIT_NAMES:
        stop = start + SPLIT_SIZES[split]
        bounds[split] = slice(start, stop)
        start = stop
    return bounds


def generate_and_freeze(
    name: str,
    *,
    n_samples: int,
    generator_kwargs: dict,
    dataset_seed: int,
    out_dir=DEFAULT_DATA_DIR,
    frozen_name: str | None = None,
    n_components: int = N_COMPONENTS,
) -> dict:
    """Generate, shuffle, split, reduce, scale, write. Returns the manifest."""
    frozen_name = frozen_name or name
    out_dir = Path(out_dir)

    # 1-2. Global seed, then exactly one generator call.
    np.random.seed(dataset_seed)
    X_generated, y_generated = _call_generator(
        name, n_samples=n_samples, generator_kwargs=generator_kwargs
    )
    X_generated = np.asarray(X_generated, dtype=np.float64)
    y_generated = np.asarray(y_generated).reshape(-1)

    # 3. Shuffle before splitting.
    shuffle_seed = derive(dataset_seed, "shuffle")
    order = np.random.default_rng(shuffle_seed).permutation(len(X_generated))
    X_shuffled, y_shuffled = X_generated[order], y_generated[order]

    # 4. Split.
    bounds = _split_indices(len(X_shuffled))
    raw = {split: X_shuffled[bounds[split]] for split in SPLIT_NAMES}
    labels = {split: y_shuffled[bounds[split]] for split in SPLIT_NAMES}

    # 5-6. PCA fitted on the training rows only.
    pca_random_state = derive(dataset_seed, "pca") % (2**31)
    pca = PCA(n_components=n_components, whiten=PCA_WHITEN, random_state=pca_random_state)
    pca.fit(raw["train"])
    projected = {split: pca.transform(raw[split]) for split in SPLIT_NAMES}

    # 7. Scale to FEATURE_RANGE, again fitted on the training rows only.
    scaler = MinMaxScaler(feature_range=FEATURE_RANGE).fit(projected["train"])
    scaled = {split: scaler.transform(projected[split]) for split in SPLIT_NAMES}

    # Train lands inside FEATURE_RANGE by construction; val and test can fall outside it
    # because the scaler never saw them. Clipped rather than rescaled — rescaling on
    # val/test would leak. The clipped fraction is recorded per split.
    low, high = FEATURE_RANGE
    clipped_fraction = {}
    max_overshoot = {}
    for split in SPLIT_NAMES:
        values = scaled[split]
        overshoot = np.maximum(low - values, values - high)
        clipped_fraction[split] = float(np.count_nonzero(overshoot > _RANGE_TOL)) / values.size
        max_overshoot[split] = float(max(overshoot.max(), 0.0))
        scaled[split] = np.clip(values, low, high)
        assert np.all(scaled[split] >= low - _RANGE_TOL) and np.all(
            scaled[split] <= high + _RANGE_TOL
        ), f"{split} split leaves FEATURE_RANGE after clipping — impossible unless NaN"

    arrays: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        arrays[f"X_{split}"] = scaled[split].astype(np.float64)
        arrays[f"y_{split}"] = labels[split].astype(np.int64)
        arrays[f"X_raw_{split}"] = raw[split]
    arrays["pca_components"] = pca.components_
    arrays["pca_mean"] = pca.mean_
    arrays["scaler_min"] = scaler.min_
    arrays["scaler_scale"] = scaler.scale_

    dataset_hash = _digest(
        [(key, arrays[key]) for key in sorted(arrays) if key.startswith(("X_", "y_"))]
    )
    pca_hash = _digest([("components", pca.components_), ("mean", pca.mean_)])

    evr = pca.explained_variance_ratio_
    retained_share = evr / evr.sum()

    manifest = {
        "frozen_name": frozen_name,
        "generator": name,
        "generator_commit": VENDORED_GENERATORS_COMMIT,
        "generator_kwargs": {k: _jsonable(v) for k, v in generator_kwargs.items()},
        "n_samples": int(n_samples),
        "dataset_seed": int(dataset_seed),
        "shuffle_key": f'derive({dataset_seed}, "shuffle")',
        "shuffle_seed": int(shuffle_seed),
        "split_sizes": {split: int(SPLIT_SIZES[split]) for split in SPLIT_NAMES},
        "class_balance": {
            split: {
                "n": int(labels[split].size),
                "fraction_positive": float(np.mean(labels[split] > 0)),
                "counts": {
                    "-1": int(np.sum(labels[split] < 0)),
                    "+1": int(np.sum(labels[split] > 0)),
                },
            }
            for split in SPLIT_NAMES
        },
        "pca": {
            "n_components": int(n_components),
            "whiten": PCA_WHITEN,
            "random_state": int(pca_random_state),
            "fitted_on": "train split only (4200 rows)",
            "explained_variance_ratio_": [float(v) for v in evr],
            "explained_variance_ratio_of_retained": [float(v) for v in retained_share],
            "total_variance_explained": float(evr.sum()),
        },
        "scaling": {
            "feature_range": [float(low), float(high)],
            "fitted_on": "train split only (4200 rows), after PCA",
            "clipped_fraction": {k: float(v) for k, v in clipped_fraction.items()},
            "max_overshoot_before_clipping": {k: float(v) for k, v in max_overshoot.items()},
            "clipping_note": (
                "val/test values outside FEATURE_RANGE are clipped; refitting the scaler "
                "on them would be a leak"
            ),
        },
        "dataset_hash": dataset_hash,
        "pca_hash": pca_hash,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    data_path, manifest_path = dataset_paths(frozen_name, out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_path, **arrays)
    manifest["file_sha256"] = _file_sha256(data_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return manifest


def _jsonable(value):
    """numpy scalars are not JSON-serialisable; the manifest must round-trip."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


# --- loading ------------------------------------------------------------------------


def load_manifest(name: str, *, out_dir=DEFAULT_DATA_DIR) -> dict:
    _, manifest_path = dataset_paths(name, out_dir=out_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_frozen_identity(
    name: str = PRODUCTION_DATASET, *, out_dir=DEFAULT_DATA_DIR
) -> dict:
    """Assert the manifest of `name` carries the hashes the registry records for it.

    load_frozen checks that the bytes match the manifest beside them; this checks that
    the manifest is the artefact the project means by that name. A run against the wrong
    dataset yields a plausible table answering the wrong question.
    """
    # Registry first: an unregistered name is refused before any file is opened.
    dataset_prefix, pca_prefix = expected_hash_prefixes(name)
    manifest = load_manifest(name, out_dir=out_dir)
    assert manifest["dataset_hash"].startswith(dataset_prefix), (
        f"dataset_hash {manifest['dataset_hash']} of {name} does not start with "
        f"{dataset_prefix}"
    )
    assert manifest["pca_hash"].startswith(pca_prefix), (
        f"pca_hash {manifest['pca_hash']} of {name} does not start with {pca_prefix}"
    )
    return manifest


def load_frozen(
    name: str,
    split: str,
    *,
    out_dir=DEFAULT_DATA_DIR,
    raw: bool = False,
    verify: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one split, verifying both hashes recorded in the manifest.

    `raw=True` returns the pre-PCA features of the same rows.

    Verification is loud and on by default: a dataset whose bytes changed after freezing
    invalidates every result computed from it, and must fail here rather than surface as
    an unexplained shift in accuracy.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")

    data_path, _ = dataset_paths(name, out_dir=out_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"no frozen dataset at {data_path}")
    manifest = load_manifest(name, out_dir=out_dir)

    if verify:
        actual_file = _file_sha256(data_path)
        if actual_file != manifest["file_sha256"]:
            raise ValueError(
                f"{data_path} was modified after freezing: sha256 {actual_file} != "
                f"{manifest['file_sha256']} in the manifest. Refusing to load."
            )

    with np.load(data_path) as archive:
        arrays = {key: archive[key] for key in archive.files}

    if verify:
        actual_content = _digest(
            [(key, arrays[key]) for key in sorted(arrays) if key.startswith(("X_", "y_"))]
        )
        if actual_content != manifest["dataset_hash"]:
            raise ValueError(
                f"{data_path} content digest {actual_content} != {manifest['dataset_hash']} "
                "in the manifest. Refusing to load."
            )
        actual_pca = _digest(
            [("components", arrays["pca_components"]), ("mean", arrays["pca_mean"])]
        )
        if actual_pca != manifest["pca_hash"]:
            raise ValueError(
                f"{data_path} PCA digest {actual_pca} != {manifest['pca_hash']} in the "
                "manifest. Refusing to load."
            )

    key = f"X_raw_{split}" if raw else f"X_{split}"
    return arrays[key], arrays[f"y_{split}"]


def load_splits(name: str, *, out_dir=DEFAULT_DATA_DIR, raw: bool = False) -> dict:
    """All three splits as {"train": (X, y), ...}."""
    return {
        split: load_frozen(name, split, out_dir=out_dir, raw=raw) for split in SPLIT_NAMES
    }
