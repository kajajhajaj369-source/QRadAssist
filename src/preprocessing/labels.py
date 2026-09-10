"""Target label handling and policy encoding (Phase 2A).

Adheres to Phase 1D dataset protocol:
- Target labels: Pneumonia, Atelectasis, Pleural Effusion.
- Preserves raw values: '1.0' (positive), '0.0' (negative), '-1.0' (uncertain), '' (unmentioned/blank).
- Does not silently overwrite raw labels in metadata.
- Implements Phase 1D baseline policy: U-Ignore (uncertain labels masked out with weight 0 for loss).
- Supports evaluation of alternative policies (U-Zeros, U-Ones).
"""

from __future__ import annotations

from typing import Mapping

TARGET_LABELS: tuple[str, ...] = ("Pneumonia", "Atelectasis", "Pleural Effusion")
SUPPORTED_POLICIES: tuple[str, ...] = ("u_ignore", "u_zeros", "u_ones")


def classify_raw_label(val: str | float | None) -> str:
    """Classify raw string or float representation into its canonical category."""
    if val is None:
        return "blank"
    text = str(val).strip()
    if not text:
        return "blank"
    try:
        fval = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid non-numeric label value: {val!r}") from exc

    if fval == 1.0:
        return "positive"
    if fval == 0.0:
        return "negative"
    if fval == -1.0:
        return "uncertain"
    raise ValueError(f"Unexpected label numeric value: {val!r}")


def encode_labels(
    raw_labels: Mapping[str, str | float | None],
    policy: str = "u_ignore",
) -> tuple[dict[str, float], dict[str, float]]:
    """Encode raw labels into numerical targets and loss masks under a given policy.

    Args:
        raw_labels: Mapping of disease name to raw value (e.g. {'Pneumonia': '1.0', ...}).
        policy: One of 'u_ignore', 'u_zeros', 'u_ones'.
            - 'u_ignore' (Phase 1D baseline): target=0.0, mask=0.0 for uncertain labels.
            - 'u_zeros': target=0.0, mask=1.0 for uncertain labels.
            - 'u_ones': target=1.0, mask=1.0 for uncertain labels.

    Returns:
        (targets, loss_mask) dictionaries keyed by disease name.
    """
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported uncertainty policy: {policy!r}. Must be one of {SUPPORTED_POLICIES}")

    targets: dict[str, float] = {}
    masks: dict[str, float] = {}

    for disease in TARGET_LABELS:
        raw_val = raw_labels.get(disease)
        cat = classify_raw_label(raw_val)

        if cat == "positive":
            targets[disease] = 1.0
            masks[disease] = 1.0
        elif cat == "negative":
            targets[disease] = 0.0
            masks[disease] = 1.0
        elif cat == "blank":
            # Clinical assumption documented in Protocol Section 5.3:
            # Unmentioned finding is assumed negative under negative reporting bias.
            targets[disease] = 0.0
            masks[disease] = 1.0
        elif cat == "uncertain":
            if policy == "u_ignore":
                # Masked out from loss computation; loss weight is 0.0.
                targets[disease] = 0.0
                masks[disease] = 0.0
            elif policy == "u_zeros":
                targets[disease] = 0.0
                masks[disease] = 1.0
            elif policy == "u_ones":
                targets[disease] = 1.0
                masks[disease] = 1.0

    return targets, masks
