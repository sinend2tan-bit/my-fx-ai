"""Unit tests for strict persisted-model compatibility checks."""

import unittest

from model_artifacts import model_matches_contract


FEATURES = ["f1", "f2", "f3"]
CLASSES = (-1, 0, 1)


class FakeModel:
    def __init__(self, classes=(-1, 0, 1), feature_count=3, names=None):
        self.classes_ = classes
        self.n_features_in_ = feature_count
        if names is not None:
            self.feature_names_in_ = names


def check(model, *, require_all=True):
    return model_matches_contract(
        model,
        expected_features=FEATURES,
        allowed_classes=CLASSES,
        model_type=FakeModel,
        require_all_classes=require_all,
    )


class ModelArtifactContractTests(unittest.TestCase):
    def test_accepts_exact_contract(self):
        self.assertTrue(check(FakeModel(names=FEATURES)))

    def test_accepts_models_without_feature_names(self):
        self.assertTrue(check(FakeModel()))

    def test_rejects_feature_count_mismatch(self):
        self.assertFalse(check(FakeModel(feature_count=2)))

    def test_rejects_boolean_feature_count(self):
        self.assertFalse(check(FakeModel(feature_count=True)))

    def test_rejects_feature_name_order_mismatch(self):
        self.assertFalse(check(FakeModel(names=["f2", "f1", "f3"])))

    def test_rejects_unknown_class(self):
        self.assertFalse(check(FakeModel(classes=(-1, 0, 2))))

    def test_rejects_missing_required_class_when_full_contract_required(self):
        self.assertFalse(check(FakeModel(classes=(-1, 1))))

    def test_allows_two_of_three_classes_for_save_compatibility(self):
        self.assertTrue(check(FakeModel(classes=(-1, 1)), require_all=False))

    def test_rejects_single_class_model(self):
        self.assertFalse(check(FakeModel(classes=(1,))))

    def test_rejects_wrong_model_type(self):
        self.assertFalse(model_matches_contract(
            object(),
            expected_features=FEATURES,
            allowed_classes=CLASSES,
            model_type=FakeModel,
        ))


if __name__ == "__main__":
    unittest.main()
