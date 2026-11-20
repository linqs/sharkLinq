"""Feature engineering and normalization utilities for media metadata."""

import dataclasses
import datetime
import json
import math
import os
import typing

import torch

from src.core import constants
import src.multimodal.multimodal_config as multimodal_config

_SOFT_TYPE_KNOWN = ("still", "video")
_SOFT_QUALITY_KNOWN = ("good", "medium", "poor")
_FEATURE_NAMES = ("sex", "location", "type", "quality", "length", "time")


@dataclasses.dataclass
class MetadataFeatureSpec:
    """Selects which metadata features to include and how to encode them across representations."""
    use_soft_encoding: bool = False
    include_sex: bool = True
    include_type: bool = True
    include_quality: bool = True
    include_location: bool = True
    include_length: bool = True
    include_time: bool = True
    use_representation_b: bool = False
    use_representation_c: bool = False
    rbf_n_bins: int = 5
    # Per-feature repC overrides that swap individual feature groups from repB to repC encoding for ablation decomposition.
    repc_categoricals: bool = False
    repc_length: bool = False
    repc_time_year: bool = False

    def __post_init__(self):
        if self.use_representation_b and self.use_representation_c:
            raise ValueError(
                "use_representation_b and use_representation_c are mutually "
                "exclusive; set at most one."
            )
        # Per-feature flags only compose with repB or repC (need the
        # split layout / centered-encoding stats).
        per_feat_set = (self.repc_categoricals or self.repc_length
                        or self.repc_time_year)
        if per_feat_set and not (self.use_representation_b
                                  or self.use_representation_c):
            raise ValueError(
                "repc_categoricals / repc_length / repc_time_year require "
                "use_representation_b=True (or use_representation_c=True). "
                "They swap one feature from repB encoding to repC encoding."
            )

    def cat_uses_repc(self) -> bool:
        return self.use_representation_c or self.repc_categoricals

    def length_uses_repc(self) -> bool:
        return self.use_representation_c or self.repc_length

    def time_year_uses_repc(self) -> bool:
        return self.use_representation_c or self.repc_time_year

    @classmethod
    def from_dict(cls, data: typing.Optional[dict]) -> "MetadataFeatureSpec":
        """Build a spec from a config dict (missing keys take defaults)."""
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON config persistence."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MetadataNormStats:
    """Statistics fit once on the corpus and reused for every per-sample call."""
    sex_vocab: typing.List[str] = dataclasses.field(default_factory=list)
    location_vocab: typing.List[str] = dataclasses.field(default_factory=list)
    type_vocab: typing.List[str] = dataclasses.field(default_factory=list)
    quality_vocab: typing.List[str] = dataclasses.field(default_factory=list)
    length_min: typing.Optional[float] = None
    length_max: typing.Optional[float] = None
    year_min: typing.Optional[int] = None
    year_max: typing.Optional[int] = None

    # representation_c: known-class vocabularies and known proportions used for centered+L2-normed categorical encoding.
    sex_known_vocab: typing.List[str] = dataclasses.field(default_factory=list)
    sex_known_proportions: typing.List[float] = dataclasses.field(default_factory=list)
    location_known_vocab: typing.List[str] = dataclasses.field(default_factory=list)
    location_known_proportions: typing.List[float] = dataclasses.field(default_factory=list)
    type_known_vocab: typing.List[str] = dataclasses.field(default_factory=list)
    type_known_proportions: typing.List[float] = dataclasses.field(default_factory=list)
    quality_known_vocab: typing.List[str] = dataclasses.field(default_factory=list)
    quality_known_proportions: typing.List[float] = dataclasses.field(default_factory=list)
    # representation_c: RBF centers, sigma, and population means used for centered+L2 continuous feature encoding.
    length_rbf_centers: typing.List[float] = dataclasses.field(default_factory=list)
    length_rbf_sigma: typing.Optional[float] = None
    length_rbf_population_mean: typing.List[float] = dataclasses.field(default_factory=list)
    year_rbf_centers: typing.List[float] = dataclasses.field(default_factory=list)
    year_rbf_sigma: typing.Optional[float] = None
    year_rbf_population_mean: typing.List[float] = dataclasses.field(default_factory=list)


def fit_norm_stats(medias: typing.Iterable, spec: MetadataFeatureSpec) -> MetadataNormStats:
    """Compute categorical vocabularies and numeric min/max from the corpus."""
    medias = list(medias)
    stats = MetadataNormStats()

    # Categorical vocabularies — only fit those we'll one-hot encode.
    # In soft mode, sex/type/quality have hard-coded rules and need no vocab,
    # but location still uses one-hot regardless of mode.
    if spec.use_soft_encoding:
        if spec.include_location:
            stats.location_vocab = _fit_vocab(_str_or_unknown(m.location) for m in medias)
    else:
        if spec.include_sex:
            stats.sex_vocab = _fit_vocab(_str_or_unknown(_sex(m)) for m in medias)
        if spec.include_location:
            stats.location_vocab = _fit_vocab(_str_or_unknown(m.location) for m in medias)
        if spec.include_type:
            stats.type_vocab = _fit_vocab(_str_or_unknown(m.type) for m in medias)
        if spec.include_quality:
            stats.quality_vocab = _fit_vocab(_str_or_unknown(m.quality) for m in medias)

    # Length min/max.
    if spec.include_length:
        lengths = [v for v in (_length(m) for m in medias) if v is not None]
        if lengths:
            stats.length_min = float(min(lengths))
            stats.length_max = float(max(lengths))

    # Year min/max.
    if spec.include_time:
        years = []
        for m in medias:
            year, _ = _year_and_day(m)
            if year is not None:
                years.append(year)
        if years:
            stats.year_min = int(min(years))
            stats.year_max = int(max(years))

    # Per-feature repC stats: fit only the components needed by the
    # active flags. ``use_representation_c=True`` implicitly enables all
    # three; the per-feature flags allow partial swaps for ablations.
    if spec.cat_uses_repc():
        if spec.include_sex:
            stats.sex_known_vocab, stats.sex_known_proportions = \
                _fit_known_proportions(_str_or_unknown(_sex(m)) for m in medias)
        if spec.include_location:
            stats.location_known_vocab, stats.location_known_proportions = \
                _fit_known_proportions(_str_or_unknown(m.location) for m in medias)
        if spec.include_type:
            stats.type_known_vocab, stats.type_known_proportions = \
                _fit_known_proportions(_str_or_unknown(m.type) for m in medias)
        if spec.include_quality:
            stats.quality_known_vocab, stats.quality_known_proportions = \
                _fit_known_proportions(_str_or_unknown(m.quality) for m in medias)
    if spec.length_uses_repc() and spec.include_length \
            and stats.length_min is not None:
        stats.length_rbf_centers, stats.length_rbf_sigma = \
            _rbf_centers(stats.length_min, stats.length_max, spec.rbf_n_bins)
        length_values = [_length(m) for m in medias]
        stats.length_rbf_population_mean = _rbf_population_mean(
            [v for v in length_values if v is not None],
            stats.length_rbf_centers, stats.length_rbf_sigma,
        )
    if spec.time_year_uses_repc() and spec.include_time \
            and stats.year_min is not None:
        stats.year_rbf_centers, stats.year_rbf_sigma = \
            _rbf_centers(float(stats.year_min), float(stats.year_max),
                         spec.rbf_n_bins)
        year_values = []
        for m in medias:
            year, _ = _year_and_day(m)
            if year is not None:
                year_values.append(float(year))
        stats.year_rbf_population_mean = _rbf_population_mean(
            year_values, stats.year_rbf_centers, stats.year_rbf_sigma,
        )

    return stats


def engineer_features(
        media,
        spec: MetadataFeatureSpec,
        stats: MetadataNormStats
) -> torch.Tensor:
    """Return the 1-D feature vector for a single Media with individual metadata attached."""
    parts: typing.List[float] = []

    # --- categoricals ---
    if spec.cat_uses_repc():
        # Centered + L2-normed categoricals, drop the Unknown bin entirely.
        # Same-class match cos sim = +1; conflict < 0; vs Unknown = 0.
        if spec.include_sex:
            parts.extend(_centered_l2_categorical(
                _str_or_unknown(_sex(media)),
                stats.sex_known_vocab, stats.sex_known_proportions,
            ))
        if spec.include_location:
            parts.extend(_centered_l2_categorical(
                _str_or_unknown(media.location),
                stats.location_known_vocab, stats.location_known_proportions,
            ))
        if spec.include_type:
            parts.extend(_centered_l2_categorical(
                _str_or_unknown(media.type),
                stats.type_known_vocab, stats.type_known_proportions,
            ))
        if spec.include_quality:
            parts.extend(_centered_l2_categorical(
                _str_or_unknown(media.quality),
                stats.quality_known_vocab, stats.quality_known_proportions,
            ))
    elif spec.use_soft_encoding:
        if spec.include_sex:
            parts.extend(_soft_sex(_sex(media)))
        if spec.include_type:
            parts.extend(_soft_type(media.type))
        if spec.include_quality:
            parts.extend(_soft_quality(media.quality))
        if spec.include_location:
            parts.extend(_one_hot(_str_or_unknown(media.location), stats.location_vocab))
    else:
        # Default one-hot path (repA, or repB without categorical swap).
        if spec.include_sex:
            parts.extend(_one_hot(_str_or_unknown(_sex(media)), stats.sex_vocab))
        if spec.include_location:
            parts.extend(_one_hot(_str_or_unknown(media.location), stats.location_vocab))
        if spec.include_type:
            parts.extend(_one_hot(_str_or_unknown(media.type), stats.type_vocab))
        if spec.include_quality:
            parts.extend(_one_hot(_str_or_unknown(media.quality), stats.quality_vocab))

    # --- length ---
    if spec.include_length:
        if spec.length_uses_repc():
            parts.extend(_l2_rbf(
                _length(media), stats.length_rbf_centers,
                stats.length_rbf_sigma or 1.0,
                stats.length_rbf_population_mean,
            ))
        elif spec.use_representation_b:
            raw = _length(media)
            if raw is None:
                parts.append(0.0)
                parts.append(0.0)
            else:
                parts.append(_norm_length(raw, stats))
                parts.append(1.0)
        else:
            parts.append(_norm_length(_length(media), stats))

    # --- time ---
    if spec.include_time:
        if spec.use_representation_b or spec.use_representation_c:
            # Split layout: time_day (sin, cos) + time_year.
            year, day = _year_and_day(media)
            day_sin, day_cos = _day_features(day)
            parts.append(day_sin)
            parts.append(day_cos)
            if spec.time_year_uses_repc():
                parts.extend(_l2_rbf(
                    float(year) if year is not None else None,
                    stats.year_rbf_centers,
                    stats.year_rbf_sigma or 1.0,
                    stats.year_rbf_population_mean,
                ))
            else:
                # repB: [year_norm, present]
                if year is None:
                    parts.append(0.0)
                    parts.append(0.0)
                else:
                    parts.append(_norm_year(year, stats))
                    parts.append(1.0)
        else:
            # repA: single 3-dim block (day_sin, day_cos, year_norm).
            parts.extend(_time_features(media, stats))

    return torch.tensor(parts, dtype=torch.float32)


def feature_dim(spec: MetadataFeatureSpec, stats: MetadataNormStats) -> int:
    """Return the dimensionality of vectors produced by ``engineer_features``."""
    dim = 0
    if spec.cat_uses_repc():
        if spec.include_sex:
            dim += len(stats.sex_known_vocab)
        if spec.include_location:
            dim += len(stats.location_known_vocab)
        if spec.include_type:
            dim += len(stats.type_known_vocab)
        if spec.include_quality:
            dim += len(stats.quality_known_vocab)
    elif spec.use_soft_encoding:
        if spec.include_sex:
            dim += 2
        if spec.include_type:
            dim += 2
        if spec.include_quality:
            dim += 3
        if spec.include_location:
            dim += len(stats.location_vocab)
    else:
        if spec.include_sex:
            dim += len(stats.sex_vocab)
        if spec.include_location:
            dim += len(stats.location_vocab)
        if spec.include_type:
            dim += len(stats.type_vocab)
        if spec.include_quality:
            dim += len(stats.quality_vocab)
    if spec.include_length:
        if spec.length_uses_repc():
            dim += len(stats.length_rbf_centers)
        elif spec.use_representation_b:
            dim += 2
        else:
            dim += 1
    if spec.include_time:
        if spec.use_representation_b or spec.use_representation_c:
            # time_day (sin, cos) + time_year (RBF or [value,present])
            dim += 2
            if spec.time_year_uses_repc():
                dim += len(stats.year_rbf_centers)
            else:
                dim += 2
        else:
            dim += 3  # repA single 3-dim block
    return dim


def _sex(media) -> typing.Optional[str]:
    if media.individual is None:
        return None
    return media.individual.sex


def _length(media) -> typing.Optional[float]:
    if media.individual is None or media.individual.length is None:
        return None
    return float(media.individual.length)


def _year_and_day(media) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:
    """Return (year, day_of_year) from the media's UNIX timestamp."""
    if media.time is None:
        return None, None
    try:
        dt = datetime.datetime.fromtimestamp(int(media.time))
    except (ValueError, OSError, OverflowError):
        return None, None
    return dt.year, dt.timetuple().tm_yday


def _str_or_unknown(value: typing.Optional[str]) -> str:
    """Mirror ``df[col].fillna('Unknown')`` for None/empty inputs."""
    if value is None or value == "":
        return "Unknown"
    return value


def _fit_vocab(values: typing.Iterable[str]) -> typing.List[str]:
    """Return alphabetically sorted unique values (matches pd.get_dummies)."""
    return sorted(set(values))


def _fit_known_proportions(
        values: typing.Iterable[str],
) -> typing.Tuple[typing.List[str], typing.List[float]]:
    """Return sorted known_vocab excluding "Unknown" and class frequency proportions."""
    counts: typing.Dict[str, int] = {}
    for v in values:
        if v == "Unknown":
            continue
        counts[v] = counts.get(v, 0) + 1
    known_vocab = sorted(counts.keys())
    total = sum(counts.values())
    if total == 0:
        return known_vocab, [0.0] * len(known_vocab)
    proportions = [counts[v] / total for v in known_vocab]
    return known_vocab, proportions


def _rbf_centers(
        v_min: float, v_max: float, n_bins: int,
) -> typing.Tuple[typing.List[float], float]:
    """Compute evenly spaced RBF centers across [v_min, v_max] and inter-center sigma spacing."""
    if n_bins < 2 or v_max <= v_min:
        return [float(v_min)], 1.0
    step = (v_max - v_min) / (n_bins - 1)
    centers = [v_min + i * step for i in range(n_bins)]
    return centers, step


def _rbf_encoding(
        value: float, centers: typing.Sequence[float], sigma: float,
) -> typing.List[float]:
    """RBF encoding: ``exp(-(value - center)^2 / (2 * sigma^2))`` per center."""
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma) if sigma > 0 else 0.0
    return [math.exp(-(value - c) * (value - c) * inv_two_sigma_sq)
            for c in centers]


def _rbf_population_mean(
        values: typing.Sequence[float],
        centers: typing.Sequence[float],
        sigma: float,
) -> typing.List[float]:
    """Mean RBF encoding over the population. Empty population -> zeros."""
    if not values:
        return [0.0] * len(centers)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma) if sigma > 0 else 0.0
    sums = [0.0] * len(centers)
    for v in values:
        for k, c in enumerate(centers):
            sums[k] += math.exp(-(v - c) * (v - c) * inv_two_sigma_sq)
    return [s / len(values) for s in sums]


def _l2_normalize_list(vec: typing.Sequence[float]) -> typing.List[float]:
    """L2-normalize a Python float list. Zero vector stays zero."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-8:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def _centered_l2_categorical(
        value: str, known_vocab: typing.Sequence[str],
        known_proportions: typing.Sequence[float],
) -> typing.List[float]:
    """Encode categorical value by centering against population proportions and L2-normalizing, or return zero if unknown."""
    if value == "Unknown" or value not in known_vocab:
        return [0.0] * len(known_vocab)
    onehot = [0.0] * len(known_vocab)
    onehot[known_vocab.index(value)] = 1.0
    centered = [a - b for a, b in zip(onehot, known_proportions)]
    return _l2_normalize_list(centered)


def _l2_rbf(
        raw_value: typing.Optional[float],
        centers: typing.Sequence[float],
        sigma: float,
        population_mean: typing.Sequence[float],
) -> typing.List[float]:
    """Compute L2-normed RBF encoding for a continuous value, mapping missing values to the L2-normed population mean."""
    if not centers:
        return []
    if raw_value is None:
        return _l2_normalize_list(list(population_mean))
    rbf = _rbf_encoding(raw_value, centers, sigma)
    return _l2_normalize_list(rbf)


def _one_hot(value: str, vocab: typing.Sequence[str]) -> typing.List[float]:
    """Return a one-hot row over ``vocab``; out-of-vocab -> all zeros."""
    row = [0.0] * len(vocab)
    if value in vocab:
        row[vocab.index(value)] = 1.0
    return row


def _soft_sex(value: typing.Optional[str]) -> typing.List[float]:
    """Probabilistic sex encoding. M->[1,0], F->[0,1], U->[0.5,0.5], else [0,0]."""
    male = 1.0 if value == "M" else (0.5 if value == "U" else 0.0)
    female = 1.0 if value == "F" else (0.5 if value == "U" else 0.0)
    return [male, female]


def _soft_type(value: typing.Optional[str]) -> typing.List[float]:
    """Probabilistic type encoding. Anything not in {still, video} -> [0.5, 0.5]."""
    if value == "still":
        return [1.0, 0.0]
    if value == "video":
        return [0.0, 1.0]
    return [0.5, 0.5]


def _soft_quality(value: typing.Optional[str]) -> typing.List[float]:
    """Probabilistic quality encoding. Out-of-set -> uniform 1/3 across known."""
    if value in _SOFT_QUALITY_KNOWN:
        return [1.0 if value == q else 0.0 for q in _SOFT_QUALITY_KNOWN]
    unknown_prob = 1.0 / len(_SOFT_QUALITY_KNOWN)
    return [unknown_prob] * len(_SOFT_QUALITY_KNOWN)


def _norm_length(value: typing.Optional[float], stats: MetadataNormStats) -> float:
    """Min-max normalize length; degenerate stats or missing value -> 0.5."""
    if (stats.length_min is None or stats.length_max is None
            or stats.length_min == stats.length_max):
        return 0.5
    if value is None:
        return 0.5
    return (value - stats.length_min) / (stats.length_max - stats.length_min)


def _day_features(day: typing.Optional[int]) -> typing.Tuple[float, float]:
    """Cyclical day-of-year (sin, cos). Missing day -> (0, 0)."""
    if day is None:
        return 0.0, 0.0
    angle = 2.0 * math.pi * day / 366.0
    return math.sin(angle), math.cos(angle)


def _norm_year(year: typing.Optional[int], stats: MetadataNormStats) -> float:
    """Min-max normalized year. Degenerate stats or missing -> 0.5."""
    if (stats.year_min is None or stats.year_max is None
            or stats.year_min == stats.year_max or year is None):
        return 0.5
    return (year - stats.year_min) / (stats.year_max - stats.year_min)


def _time_features(media, stats: MetadataNormStats) -> typing.List[float]:
    """Cyclical day-of-year (sin, cos) plus min-max normalized year."""
    year, day = _year_and_day(media)
    day_sin, day_cos = _day_features(day)
    year_norm = _norm_year(year, stats)
    return [day_sin, day_cos, year_norm]


def _resolve_metadata_spec(
        spec_from: str,
        representation_b: bool = False,
        representation_c: bool = False,
        repc_categoricals: bool = False,
        repc_length: bool = False,
        repc_time_year: bool = False,
        only_feature: str = None,
) -> MetadataFeatureSpec:
    """Build a MetadataFeatureSpec from a model config directory or defaults, applying representation overrides."""
    if representation_b and representation_c:
        raise ValueError("--representation-b and --representation-c are "
                         "mutually exclusive.")
    if spec_from:
        config_path = os.path.join(
            constants.TRAINED_MODELS_DIR, spec_from, "config.json"
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        spec = MetadataFeatureSpec.from_dict(
            saved.get("metadata_features")
        )
    else:
        spec = MetadataFeatureSpec.from_dict(
            multimodal_config.DEFAULT_METADATA_FEATURES
        )
    if representation_b:
        spec.use_representation_b = True
        spec.use_representation_c = False
    if representation_c:
        spec.use_representation_c = True
        spec.use_representation_b = False
    if repc_categoricals:
        spec.repc_categoricals = True
    if repc_length:
        spec.repc_length = True
    if repc_time_year:
        spec.repc_time_year = True
    if only_feature is not None:
        if only_feature not in _FEATURE_NAMES:
            raise ValueError(
                f"only_feature must be one of {_FEATURE_NAMES}, "
                f"got {only_feature!r}.")
        for f in _FEATURE_NAMES:
            setattr(spec, f"include_{f}", f == only_feature)
    spec.__post_init__()
    return spec


def _group_slices(spec, stats) -> dict:
    """Compute (start, end) dimension ranges for each metadata group inside the engineered feature vector."""
    slices = {}
    offset = 0
    if spec.cat_uses_repc():
        if spec.include_sex:
            n = len(stats.sex_known_vocab)
            slices["sex"] = (offset, offset + n); offset += n
        if spec.include_location:
            n = len(stats.location_known_vocab)
            slices["location"] = (offset, offset + n); offset += n
        if spec.include_type:
            n = len(stats.type_known_vocab)
            slices["type"] = (offset, offset + n); offset += n
        if spec.include_quality:
            n = len(stats.quality_known_vocab)
            slices["quality"] = (offset, offset + n); offset += n
    elif spec.use_soft_encoding:
        if spec.include_sex:
            slices["sex"] = (offset, offset + 2); offset += 2
        if spec.include_type:
            slices["type"] = (offset, offset + 2); offset += 2
        if spec.include_quality:
            slices["quality"] = (offset, offset + 3); offset += 3
        if spec.include_location:
            n = len(stats.location_vocab)
            slices["location"] = (offset, offset + n); offset += n
    else:
        if spec.include_sex:
            n = len(stats.sex_vocab)
            slices["sex"] = (offset, offset + n); offset += n
        if spec.include_location:
            n = len(stats.location_vocab)
            slices["location"] = (offset, offset + n); offset += n
        if spec.include_type:
            n = len(stats.type_vocab)
            slices["type"] = (offset, offset + n); offset += n
        if spec.include_quality:
            n = len(stats.quality_vocab)
            slices["quality"] = (offset, offset + n); offset += n
    if spec.include_length:
        if spec.length_uses_repc():
            n = len(stats.length_rbf_centers)
        elif spec.use_representation_b:
            n = 2
        else:
            n = 1
        slices["length"] = (offset, offset + n); offset += n
    if spec.include_time:
        if spec.use_representation_b or spec.use_representation_c:
            slices["time_day"] = (offset, offset + 2); offset += 2
            if spec.time_year_uses_repc():
                n = len(stats.year_rbf_centers)
            else:
                n = 2
            slices["time_year"] = (offset, offset + n); offset += n
        else:
            slices["time"] = (offset, offset + 3); offset += 3
    return slices
