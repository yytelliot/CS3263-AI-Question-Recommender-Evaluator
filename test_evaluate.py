"""Independent arithmetic, input-contract, baseline and integration checks."""
import contextlib
import csv
import io
import itertools
import json
from pathlib import Path
import statistics
import tempfile
import unittest

import evaluate as ev

BASE = Path(__file__).resolve().parent


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.settings = ev.Settings(bootstrap_samples=200)
        self.bank = {str(i): {"skill_id": str((i - 1) // 4 + 1),
                              "difficulty_beta_mean": (.2, .6, .8)[(i - 1) // 4]}
                     for i in range(1, 13)}
        self.history = {"u": [
            {"question_id": "1", "skill_id": "1", "order_id": 10, "correct": 0},
            {"question_id": "2", "skill_id": "1", "order_id": 20, "correct": 0},
            {"question_id": "5", "skill_id": "2", "order_id": 30, "correct": 1},
            {"question_id": "6", "skill_id": "2", "order_id": 40, "correct": 1},
        ]}
        self.states = ev.build_states(self.history, self.bank, {"u": 40}, self.settings)

    def recs(self):
        return [{"model_id": "m", "student_id": "u", "rank": i, "question_id": str(i)}
                for i in range(1, 11)]

    def test_hand_calculated_three_metrics(self):
        rows, details = ev.score({"m": {"u": self.recs()}}, self.states, self.bank, self.settings)
        self.assertEqual(self.states["u"]["weak"], {"1"})
        self.assertEqual(rows[0][ev.METRICS[0]], .4)
        self.assertAlmostEqual(rows[0][ev.METRICS[1]], .86)
        self.assertEqual(rows[0][ev.METRICS[2]], .4)
        self.assertEqual(rows[0]["low_support_recommendation_rate_at_10"], .2)
        self.assertEqual(details[-1]["student_skill_proficiency"], .5)
        self.assertNotIn("weak_topic_coverage_at_10", rows[0])

    def test_all_tied_minima_included_but_not_stronger_skill(self):
        history = {"u": [dict(r, correct=0) for r in self.history["u"]] + [
            {"question_id": "9", "skill_id": "3", "order_id": 50, "correct": 1},
            {"question_id": "10", "skill_id": "3", "order_id": 60, "correct": 1}]}
        state = ev.build_states(history, self.bank, {"u": 60}, self.settings)["u"]
        self.assertEqual(state["weak"], {"1", "2"})
        self.assertTrue(state["weakest_skill_tie"])

    def test_equal_fractions_with_different_support_tie(self):
        history = {"u": [
            {"question_id": "1", "skill_id": "1", "order_id": i, "correct": c}
            for i, c in enumerate([0, 1], 1)] + [
            {"question_id": "5", "skill_id": "2", "order_id": i, "correct": c}
            for i, c in enumerate([0, 1, 0, 1], 3)]}
        state = ev.build_states(history, self.bank, {"u": 6}, self.settings)["u"]
        self.assertEqual(state["weak"], {"1", "2"})

    def test_one_response_does_not_make_a_weak_skill(self):
        history = {"u": self.history["u"] + [
            {"question_id": "9", "skill_id": "3", "order_id": 50, "correct": 0}]}
        state = ev.build_states(history, self.bank, {"u": 50}, self.settings)["u"]
        self.assertNotIn("3", state["eligible"])
        self.assertEqual(state["weak"], {"1"})

    def test_future_history_excluded(self):
        history = {"u": self.history["u"] + [
            {"question_id": "3", "skill_id": "1", "order_id": 50, "correct": 1}]}
        self.assertEqual(ev.build_states(history, self.bank, {"u": 40}, self.settings), self.states)

    def test_no_eligible_skill_fails(self):
        with self.assertRaisesRegex(ev.ValidationError, "need at least one"):
            ev.build_states(self.history, self.bank, {"u": 10}, self.settings)

    def test_rank_order_does_not_change_metrics(self):
        a = self.recs()
        b = [dict(r, rank=11-r["rank"]) for r in a]
        rows, _ = ev.score({"a": {"u": a}, "b": {"u": b}}, self.states, self.bank, self.settings)
        for metric in ev.METRICS:
            self.assertEqual(rows[0][metric], rows[1][metric])

    def test_random_expectation_matches_exhaustive_lists(self):
        # All unordered lists are enough because these metrics do not use rank.
        measured = []
        for questions in itertools.combinations(self.bank, 10):
            recs = [{"question_id": q, "rank": i} for i, q in enumerate(questions, 1)]
            rows, _ = ev.score({"m": {"u": recs}}, self.states, self.bank, self.settings)
            measured.append(rows[0])
        expected = ev.random_expected_scores(self.states, self.bank, self.settings)[0]
        for metric in ev.METRICS:
            self.assertAlmostEqual(expected[metric], statistics.mean(r[metric] for r in measured))

    def test_bad_outputs_fail(self):
        variants = [self.recs()[:-1]]
        for field, value in [("question_id", "1"), ("question_id", "999"), ("rank", 1),
                             ("rank", "2.5"), ("student_id", "unknown"),
                             ("model_id", ev.RANDOM_BASELINE)]:
            rows = self.recs()
            rows[-1][field] = value
            variants.append(rows)
        rows = [dict(r, checkpoint_order_id=50) for r in self.recs()]
        variants.append(rows)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder) / "recs.csv"
            for rows in variants:
                with self.subTest(last=rows[-1]):
                    ev.write_csv(path, rows)
                    with self.assertRaises(ev.ValidationError):
                        ev.load_recommendations([path], self.states, self.bank)

    def test_missing_students_fail(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder) / "recs.csv"
            ev.write_csv(path, self.recs())
            with self.assertRaisesRegex(ev.ValidationError, "missing evaluation"):
                ev.load_recommendations([path], {**self.states, "v": self.states["u"]}, self.bank)

    def test_model_metadata_cannot_override_scoring(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder) / "recs.csv"
            ev.write_csv(path, self.recs())
            original = ev.load_recommendations([path], self.states, self.bank)
            ev.write_csv(path, [dict(r, skill_id="fake", difficulty_beta_mean=999,
                                     predicted_mastery=999) for r in self.recs()])
            changed = ev.load_recommendations([path], self.states, self.bank)
            self.assertEqual(original, changed)

    def test_training_provenance_and_overlap(self):
        bank = {"1": {"skill_id": "1", "difficulty_beta_mean": .5}}
        training = {"train": [{"question_id": "1", "correct": 0},
                              {"question_id": "1", "correct": 1}]}
        self.assertTrue(ev.validate_training(training, self.history, bank)[
            "training_only_difficulty_recomputation_matches"])
        with self.assertRaisesRegex(ev.ValidationError, "overlap"):
            ev.validate_training(training, training, bank)
        with self.assertRaisesRegex(ev.ValidationError, "difficulty"):
            ev.validate_training(training, self.history, {"1": {"difficulty_beta_mean": .9}})

    def test_paired_directions_and_win_counts(self):
        rows = [{"model_id": model, "student_id": sid, "history_group": "short",
                 **{m: value + offset for m in ev.METRICS}}
                for model, offset in [("a", .1), ("b", 0)]
                for sid, value in [("u", .2), ("v", .6)]]
        comparisons = ev.paired_comparisons(rows, self.settings)
        for row in comparisons:
            self.assertAlmostEqual(row["mean_difference_a_minus_b"], .1)
            self.assertAlmostEqual(row["ci95_low"], .1)
            self.assertAlmostEqual(row["ci95_high"], .1)
            self.assertTrue(row["interval_excludes_zero"])
            self.assertEqual(row["a_wins"], 0 if row["metric"] == ev.METRICS[2] else 2)
        differences = ev.student_differences(rows)
        self.assertTrue(all(r["winner"] == ("b" if r["metric"] == ev.METRICS[2] else "a")
                            for r in differences))

    def test_real_dataset_and_complete_cli(self):
        base = Path(__file__).resolve().parent
        bank = ev.load_bank(base / "data/question_bank.csv")
        history = ev.load_responses(base / "data/evaluation.csv", bank)
        states = ev.build_states(history, bank, ev.resolve_checkpoints(history), self.settings)
        self.assertEqual(len(states), 30)
        self.assertEqual(sum(s["weakest_skill_tie"] for s in states.values()), 3)
        self.assertEqual(sum(s["history_group"] == "short" for s in states.values()), 15)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            out = Path(folder) / "results"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ev.main(["--demo", "--bootstrap-samples", "100", "--out", str(out)]), 0)
            summary = ev.read_csv(out / "summary.csv", ["model_id"])
            self.assertEqual(len(summary), 9)  # Two demos + expected baseline, three groups.
            self.assertEqual(len(ev.read_csv(out / "per_recommendation.csv", [])), 600)
            self.assertEqual(len(ev.read_csv(out / "paired_comparison.csv", [])), 27)
            self.assertEqual(len(ev.read_csv(out / "per_student_differences.csv", [])), 270)
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertTrue(manifest["smoke_test_only"])
            self.assertTrue(manifest["dataset_checks"]["training_only_difficulty_recomputation_matches"])
            for row in summary:
                for metric in ev.METRICS:
                    self.assertTrue(0 <= float(row[metric]) <= 1)
            # Exercise the actual model-output route too, with the generated valid fixture.
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ev.main(["--recommendations", str(out / "smoke_test_recommendations.csv"),
                                          "--bootstrap-samples", "100", "--out", str(Path(folder) / "loaded")]), 0)


if __name__ == "__main__":
    unittest.main()
