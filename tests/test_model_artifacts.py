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

    def test_rejects_models_without_feature_names(self):
        self.assertFalse(check(FakeModel()))

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
        self.assertTrue(
            check(FakeModel(classes=(-1, 1), names=FEATURES), require_all=False)
        )

    def test_rejects_single_class_model(self):
        self.assertFalse(check(FakeModel(classes=(1,))))

    def test_rejects_wrong_model_type(self):
        self.assertFalse(model_matches_contract(
            object(),
            expected_features=FEATURES,
            allowed_classes=CLASSES,
            model_type=FakeModel,
        ))

    def test_random_forest_joblib_round_trip_preserves_feature_contract(self):
        import tempfile

        import joblib
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        frame = pd.DataFrame(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.5, 0.5, 0.5],
                [0.55, 0.5, 0.45],
                [0.8, 0.7, 0.6],
                [0.9, 0.8, 0.7],
                [1.0, 0.9, 0.8],
            ],
            columns=FEATURES,
        )
        model = RandomForestClassifier(n_estimators=4, random_state=0)
        model.fit(frame, [-1, -1, -1, 0, 0, 1, 1, 1])

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/model.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)

        self.assertTrue(model_matches_contract(
            loaded,
            expected_features=FEATURES,
            allowed_classes=CLASSES,
            model_type=RandomForestClassifier,
        ))

    def test_random_forest_joblib_round_trip_rejects_reordered_features(self):
        import tempfile

        import joblib
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        frame = pd.DataFrame(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.5, 0.5, 0.5],
                [0.55, 0.5, 0.45],
                [0.8, 0.7, 0.6],
                [0.9, 0.8, 0.7],
                [1.0, 0.9, 0.8],
            ],
            columns=FEATURES,
        )
        model = RandomForestClassifier(n_estimators=4, random_state=0)
        model.fit(frame, [-1, -1, -1, 0, 0, 1, 1, 1])

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/model.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)

        self.assertFalse(model_matches_contract(
            loaded,
            expected_features=["f2", "f1", "f3"],
            allowed_classes=CLASSES,
            model_type=RandomForestClassifier,
        ))


if __name__ == "__main__":
    unittest.main()
