"""Strict compatibility checks for persisted sklearn model artifacts.

This module is deliberately independent of Streamlit and training logic so its
contract can be tested without importing the UI application.
"""

from numbers import Integral


def model_matches_contract(
    model,
    *,
    expected_features,
    allowed_classes,
    model_type,
    require_all_classes=True,
):
    """Return True only when a fitted model matches its feature/class contract.

    Missing feature_names_in_ is accepted for models fitted from NumPy arrays;
    when names are present they must match exactly and in order.
    """
    if not isinstance(model, model_type):
        return False

    try:
        features = list(expected_features)
        classes = list(model.classes_)
        feature_count = model.n_features_in_
    except Exception:
        return False

    if isinstance(feature_count, bool) or not isinstance(feature_count, Integral):
        return False
    if int(feature_count) != len(features):
        return False
    if len(classes) < 2:
        return False

    try:
        allowed = set(allowed_classes)
        if any(isinstance(value, bool) or value not in allowed for value in classes):
            return False
        if require_all_classes and set(classes) != allowed:
            return False
    except Exception:
        return False

    fitted_names = getattr(model, "feature_names_in_", None)
    if fitted_names is not None:
        try:
            if list(fitted_names) != features:
                return False
        except Exception:
            return False

    return True
