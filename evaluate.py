"""CS3263 three-metric recommendation evaluator. Python 3.10+, no extra packages."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


METRICS = (
    "weakest_skill_targeting_at_10",
    "difficulty_appropriateness_at_10",
    "repeat_rate_at_10",
)
K = 10
RANDOM_BASELINE = "baseline_random_expected"
DIFFERENCE_FIELDS = ["model_a", "model_b", "student_id", "history_group", "metric",
                     "score_a", "score_b", "difference_a_minus_b", "benefit_a_minus_b", "winner"]
COMPARISON_FIELDS = ["model_a", "model_b", "history_group", "metric", "students",
                     "mean_difference_a_minus_b", "ci95_low", "ci95_high", "better_direction",
                     "a_wins", "ties", "b_wins", "practical_threshold", "mean_benefit_a_minus_b",
                     "interval_excludes_zero", "estimated_effect_meets_threshold",
                     "interval_entirely_beyond_threshold"]


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    min_skill_responses: int = 2
    short_history_max: int = 12
    bootstrap_samples: int = 5000
    seed: int = 3263
    targeting_threshold: float = .10
    repeat_threshold: float = .10


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def integer(value, label):
    try:
        # Reject fractional numbers and missing values, rather than truncating.
        result = int(str(value).strip())
    except (ValueError, TypeError):
        raise ValidationError(f"{label}: expected an integer, got {value!r}") from None
    return result


def identifier(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: missing ID")
    return value.strip()


def id_key(value):
    return (0, int(value)) if value.isdigit() else (1, value)


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), f"{path}: duplicate column names")
        missing = set(required) - set(fields)
        require(not missing, f"{path}: missing columns {sorted(missing)}")
        rows = []
        for line, row in enumerate(reader, 2):
            require(None not in row and all(v is not None for v in row.values()),
                    f"{path}, line {line}: malformed CSV row")
            rows.append(row)
    require(bool(rows), f"{path}: contains no data rows")
    return rows


def load_bank(path):
    bank = {}
    for row in read_csv(path, ["question_id", "skill_id", "difficulty_beta_mean"]):
        qid = identifier(row["question_id"], "question_id")
        skill = identifier(row["skill_id"], "skill_id")
        require(qid not in bank, f"Question bank: duplicate question_id {qid}")
        try:
            difficulty = float(row["difficulty_beta_mean"])
        except ValueError:
            raise ValidationError(f"Question {qid}: difficulty is not numeric") from None
        require(math.isfinite(difficulty) and 0 <= difficulty <= 1,
                f"Question {qid}: difficulty must be finite and between 0 and 1")
        bank[qid] = {**row, "question_id": qid, "skill_id": skill,
                     "difficulty_beta_mean": difficulty}
    return bank


def load_responses(path, bank):
    result = defaultdict(list)
    events = set()
    for row in read_csv(path, ["student_id", "question_id", "skill_id", "order_id", "correct"]):
        sid = identifier(row["student_id"], "student_id")
        qid = identifier(row["question_id"], "question_id")
        skill = identifier(row["skill_id"], "skill_id")
        order = integer(row["order_id"], "order_id")
        correct = integer(row["correct"], "correct")
        require(correct in (0, 1), f"Student {sid}, order {order}: correct must be 0 or 1")
        require(qid in bank, f"Response question {qid} is absent from the bank")
        require(skill == bank[qid]["skill_id"], f"Question {qid}: response/bank skill mismatch")
        require((sid, order) not in events, f"Student {sid}: duplicate response order_id {order}")
        events.add((sid, order))
        result[sid].append({"question_id": qid, "skill_id": skill,
                            "order_id": order, "correct": correct})
    return {sid: sorted(rows, key=lambda r: r["order_id"]) for sid, rows in result.items()}


def validate_training(training, evaluation, bank):
    overlap = set(training) & set(evaluation)
    require(not overlap, f"Training/evaluation student overlap: {sorted(overlap)[:10]}")
    questions = defaultdict(list)
    for sid, rows in training.items():
        for row in rows:
            questions[row["question_id"]].append((sid, row["correct"]))
    supports = []
    for qid, meta in bank.items():
        observations = questions[qid]
        require(bool(observations), f"Question {qid}: no training responses")
        n = len(observations)
        correct = sum(value for _, value in observations)
        students = len({sid for sid, _ in observations})
        expected = (n - correct + 1) / (n + 2)
        require(math.isclose(meta["difficulty_beta_mean"], expected, rel_tol=0, abs_tol=1e-10),
                f"Question {qid}: difficulty does not match the supplied training responses")
        for field, value in [("attempts", n), ("students", students),
                             ("correct_count", correct), ("incorrect_count", n - correct)]:
            if field in meta:
                require(integer(meta[field], f"{qid}.{field}") == value,
                        f"Question {qid}: {field} does not match training responses")
        supports.append(students)
    return {"training_students": len(training), "evaluation_students": len(evaluation),
            "training_evaluation_student_overlap": 0,
            "training_rows": sum(map(len, training.values())),
            "bank_questions": len(bank),
            "bank_skills": len({q["skill_id"] for q in bank.values()}),
            "minimum_training_students_per_question": min(supports),
            "training_only_difficulty_recomputation_matches": True}


def resolve_checkpoints(history, path=None):
    if path is None:
        return {sid: rows[-1]["order_id"] for sid, rows in history.items()}
    checkpoints = {}
    for row in read_csv(path, ["student_id", "checkpoint_order_id"]):
        sid = identifier(row["student_id"], "student_id")
        require(sid not in checkpoints, f"Checkpoints: duplicate student {sid}")
        checkpoints[sid] = integer(row["checkpoint_order_id"], "checkpoint_order_id")
    require(set(checkpoints) == set(history), "Checkpoints must contain exactly the evaluation students")
    for sid, cutoff in checkpoints.items():
        require(cutoff in {r["order_id"] for r in history[sid]},
                f"Student {sid}: checkpoint must be an order_id in that student's history")
    return checkpoints


def build_states(history, bank, checkpoints, settings):
    skills = {meta["skill_id"] for meta in bank.values()}
    states = {}
    for sid, all_rows in history.items():
        rows = [r for r in all_rows if r["order_id"] <= checkpoints[sid]]
        counts = Counter(r["skill_id"] for r in rows)
        correct = Counter()
        for row in rows:
            correct[row["skill_id"]] += row["correct"]
        proficiency = {s: (correct[s] + 1) / (counts[s] + 2) for s in skills}
        eligible = [s for s in skills if counts[s] >= settings.min_skill_responses]
        eligible.sort(key=lambda s: (proficiency[s], id_key(s)))
        require(bool(eligible),
                f"Student {sid}: only {len(eligible)} skills meet the minimum response count; "
                "need at least one. Choose another checkpoint or a different rubric.")
        # Compare exact fractions so equal rational values receive equal treatment.
        reference = {s: Fraction(correct[s] + 1, counts[s] + 2) for s in eligible}
        minimum = min(reference.values())
        weak = {s for s in eligible if reference[s] == minimum}
        seen = {r["question_id"] for r in rows}
        states[sid] = {
            "checkpoint_order_id": checkpoints[sid], "history_length": len(rows),
            "history_group": "short" if len(rows) <= settings.short_history_max else "long",
            "seen": seen, "counts": dict(counts), "proficiency": proficiency,
            "weak": weak, "eligible": eligible, "weakest_skill_tie": len(weak) > 1,
        }
    return states


def load_recommendations(paths, states, bank):
    models = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for row in read_csv(path, ["model_id", "student_id", "rank", "question_id"]):
            model = identifier(row["model_id"], "model_id")
            require(model != RANDOM_BASELINE, f"{RANDOM_BASELINE} is a reserved baseline identifier")
            sid = identifier(row["student_id"], "student_id")
            qid = identifier(row["question_id"], "question_id")
            rank = integer(row["rank"], "rank")
            require(sid in states, f"{model}: unknown evaluation student {sid}")
            require(qid in bank, f"{model}, student {sid}: unknown question {qid}")
            require(1 <= rank <= K, f"{model}, student {sid}: rank must be 1 through {K}")
            if "checkpoint_order_id" in row:
                cutoff = integer(row["checkpoint_order_id"], "checkpoint_order_id")
                require(cutoff == states[sid]["checkpoint_order_id"],
                        f"{model}, student {sid}: checkpoint mismatch")
            models[model][sid].append({"rank": rank, "question_id": qid})
    for model, student_lists in models.items():
        require(set(student_lists) == set(states), f"{model}: missing evaluation students "
                f"{sorted(set(states) - set(student_lists), key=id_key)}")
        for sid, recs in student_lists.items():
            label = f"{model}, student {sid}"
            require(len(recs) == K, f"{label}: expected {K} recommendations, found {len(recs)}")
            require({r["rank"] for r in recs} == set(range(1, K + 1)),
                    f"{label}: ranks must be 1 through {K} exactly once")
            require(len({r["question_id"] for r in recs}) == K,
                    f"{label}: duplicate recommended question IDs")
            recs.sort(key=lambda r: r["rank"])
    return dict(models)


def score(models, states, bank, settings):
    student_rows, question_rows = [], []
    for model in sorted(models):
        for sid in sorted(states, key=id_key):
            state = states[sid]
            details = []
            for rec in models[model][sid]:
                qid = rec["question_id"]
                skill = bank[qid]["skill_id"]
                p = state["proficiency"][skill]
                d = bank[qid]["difficulty_beta_mean"]
                details.append({
                    "model_id": model, "student_id": sid, **rec,
                    "skill_id": skill, "difficulty_beta_mean": d,
                    "student_skill_proficiency": p,
                    "prior_skill_responses": state["counts"].get(skill, 0),
                    "is_weak_skill": int(skill in state["weak"]),
                    "difficulty_appropriateness": 1 - abs(p - d),
                    "is_repeat": int(qid in state["seen"]),
                    "is_low_support_skill": int(state["counts"].get(skill, 0) < settings.min_skill_responses),
                })
            student_rows.append({
                "model_id": model, "student_id": sid,
                "checkpoint_order_id": state["checkpoint_order_id"],
                "history_length": state["history_length"], "history_group": state["history_group"],
                "unique_prior_questions": len(state["seen"]),
                "weak_skill_ids": ";".join(sorted(state["weak"], key=id_key)),
                "weakest_skill_count": len(state["weak"]),
                "eligible_skill_count": len(state["eligible"]),
                "weakest_skill_tie": int(state["weakest_skill_tie"]),
                "weakest_skill_targeting_at_10": sum(d["is_weak_skill"] for d in details) / K,
                "difficulty_appropriateness_at_10": statistics.mean(d["difficulty_appropriateness"] for d in details),
                "repeat_rate_at_10": sum(d["is_repeat"] for d in details) / K,
                "low_support_recommendation_rate_at_10": sum(d["is_low_support_skill"] for d in details) / K,
            })
            question_rows.extend(details)
    return student_rows, question_rows


def random_expected_scores(states, bank, settings):
    """Exact expectations for uniformly sampling 10 distinct questions from the bank.

    Each question has inclusion probability K/len(bank), so expected list means
    equal catalog means. No Monte Carlo variation or invented rankings are needed.
    """
    result = []
    for sid in sorted(states, key=id_key):
        state = states[sid]
        result.append({
            "model_id": RANDOM_BASELINE, "student_id": sid,
            "checkpoint_order_id": state["checkpoint_order_id"],
            "history_length": state["history_length"], "history_group": state["history_group"],
            "unique_prior_questions": len(state["seen"]),
            "weak_skill_ids": ";".join(sorted(state["weak"], key=id_key)),
            "weakest_skill_count": len(state["weak"]), "eligible_skill_count": len(state["eligible"]),
            "weakest_skill_tie": int(state["weakest_skill_tie"]),
            "weakest_skill_targeting_at_10": statistics.mean(
                int(q["skill_id"] in state["weak"]) for q in bank.values()),
            "difficulty_appropriateness_at_10": statistics.mean(
                1 - abs(state["proficiency"][q["skill_id"]] - q["difficulty_beta_mean"])
                for q in bank.values()),
            "repeat_rate_at_10": len(state["seen"]) / len(bank),
            "low_support_recommendation_rate_at_10": statistics.mean(
                int(state["counts"].get(q["skill_id"], 0) < settings.min_skill_responses)
                for q in bank.values()),
        })
    return result


def benefit_sign(metric):
    return -1 if metric == "repeat_rate_at_10" else 1


def student_differences(student_rows):
    by_model = {}
    for row in student_rows:
        by_model.setdefault(row["model_id"], {})[row["student_id"]] = row
    result = []
    for first, second in itertools.combinations(sorted(by_model), 2):
        for sid in sorted(by_model[first], key=id_key):
            for metric in METRICS:
                a, b = by_model[first][sid], by_model[second][sid]
                delta = a[metric] - b[metric]
                benefit = benefit_sign(metric) * delta
                result.append({"model_a": first, "model_b": second, "student_id": sid,
                               "history_group": a["history_group"], "metric": metric,
                               "score_a": a[metric], "score_b": b[metric],
                               "difference_a_minus_b": delta, "benefit_a_minus_b": benefit,
                               "winner": "tie" if abs(benefit) <= 1e-12 else first if benefit > 0 else second})
    return result


def aggregate(student_rows):
    result = []
    for model in sorted({r["model_id"] for r in student_rows}):
        for group in ("overall", "short", "long"):
            selected = [r for r in student_rows if r["model_id"] == model
                        and (group == "overall" or r["history_group"] == group)]
            if selected:
                result.append({"model_id": model, "history_group": group, "students": len(selected),
                               **{m: statistics.mean(r[m] for r in selected) for m in METRICS},
                               "low_support_recommendation_rate_at_10": statistics.mean(
                                   r["low_support_recommendation_rate_at_10"] for r in selected)})
    return result


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_comparisons(student_rows, settings):
    result = []
    models = sorted({r["model_id"] for r in student_rows})
    for first, second in itertools.combinations(models, 2):
        by_model = {model: {r["student_id"]: r for r in student_rows if r["model_id"] == model}
                    for model in (first, second)}
        for group in ("overall", "short", "long"):
            sids = sorted([sid for sid, row in by_model[first].items()
                           if group == "overall" or row["history_group"] == group], key=id_key)
            if not sids:
                continue
            # A draw resamples students and keeps each student's model pair intact.
            rng = random.Random(settings.seed)
            values = {m: [by_model[first][s][m] - by_model[second][s][m] for s in sids]
                      for m in METRICS}
            boot = {m: [] for m in METRICS}
            if len(sids) > 1:
                for _ in range(settings.bootstrap_samples):
                    draw = [rng.randrange(len(sids)) for _ in sids]
                    for metric in METRICS:
                        boot[metric].append(statistics.mean(values[metric][i] for i in draw))
            for metric in METRICS:
                samples = sorted(boot[metric])
                low = percentile(samples, .025) if samples else None
                high = percentile(samples, .975) if samples else None
                mean = statistics.mean(values[metric])
                sign = benefit_sign(metric)
                threshold = (settings.repeat_threshold if sign == -1 else
                             settings.targeting_threshold if metric == METRICS[0] else None)
                ties = sum(abs(v) <= 1e-12 for v in values[metric])
                wins = sum(sign * v > 1e-12 for v in values[metric])
                result.append({
                    "model_a": first, "model_b": second, "history_group": group,
                    "metric": metric, "students": len(sids),
                    "mean_difference_a_minus_b": mean,
                    "ci95_low": low, "ci95_high": high,
                    "better_direction": "lower" if metric == "repeat_rate_at_10" else "higher",
                    "a_wins": wins, "ties": ties, "b_wins": len(sids) - wins - ties,
                    "practical_threshold": threshold,
                    "mean_benefit_a_minus_b": sign * mean,
                    "interval_excludes_zero": (low > 0 or high < 0) if samples else None,
                    "estimated_effect_meets_threshold": (abs(mean) + 1e-12 >= threshold) if threshold is not None else None,
                    "interval_entirely_beyond_threshold": (low > threshold or high < -threshold)
                    if threshold is not None and samples else None,
                })
    return result


def write_csv(path, rows, fields=None):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def report_text(summary, checks, settings, demo, comparisons):
    lines = ["# Recommendation evaluation", ""]
    if demo:
        lines += ["SMOKE TEST ONLY: these randomly generated recommendations are not BKT or LLM results.", ""]
    lines += [f"Validated {checks['evaluation_students']} evaluation students; "
              f"{checks['bank_questions']} candidate questions; no training/evaluation student overlap.", "",
              "| Model | History group | Students | Weakest-skill targeting | Difficulty match | Repeat rate |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['model_id']} | {row['history_group']} | {row['students']} | "
                     + " | ".join(f"{row[m]:.4f}" for m in METRICS) + " |")
    lines += ["", "## Overall paired comparisons", "",
              "Raw difference = A minus B. For repeat rate, negative favours A; for other metrics, positive favours A.", "",
              "| A | B | Metric | Mean difference | 95% interval | A wins / ties / B wins |",
              "| --- | --- | --- | ---: | --- | --- |"]
    for row in comparisons:
        if row["history_group"] != "overall":
            continue
        interval = (f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
                    if row["ci95_low"] is not None else "Unavailable (one student)")
        lines.append(f"| {row['model_a']} | {row['model_b']} | {row['metric']} | "
                     f"{row['mean_difference_a_minus_b']:.4f} | {interval} | "
                     f"{row['a_wins']} / {row['ties']} / {row['b_wins']} |")
    lines += ["", "All scores range from 0 to 1. Lower repeat rate is better; higher is better for the other scores.",
              "Results are macro averages: each student contributes equally.", "",
              f"Short histories contain at most {settings.short_history_max} responses. "
              "The history-group comparison is descriptive; it does not establish a causal effect of history length.", "",
              "## Definitions and interpretation", "",
              "Weak skills: all skills tied for the lowest smoothed proficiency among skills with "
              f"at least {settings.min_skill_responses} prior responses. "
              "All exact ties are included and flagged in per_student.csv. "
              "These are relative weaknesses, even when the student's accuracy is high.", "",
              "Proficiency = (correct + 1) / (responses + 2), using only supplied pre-checkpoint history. "
              "Unobserved skills receive a neutral 0.5 prior. Recommendations to skills below the support "
              "minimum are counted in low_support_recommendation_rate_at_10.", "",
              "Question difficulty = (incorrect training responses + 1) / (training responses + 2). "
              "Every bank value was checked against the supplied training file.", "",
              "Weakest-skill targeting: fraction of 10 recommendations targeting the weak set. "
              "Difficulty match: mean of 1 - abs(student skill proficiency - question difficulty). "
              "Repeat rate: fraction previously attempted by the same student.", "",
              "Difficulty matching is a chosen policy heuristic; it is not a calibrated probability of "
              "answering a question correctly. Repeats can be useful revision, so repeat rate measures novelty "
              "rather than learning harm. Recommendation order does not affect these three metrics.", "",
              "Recommendations remain eligible even if previously seen. Missing or invalid model outputs fail "
              "validation instead of being silently removed. No weighted overall winner score is calculated.", "",
              "This subset contains histories filtered to the 50-question bank. Weakness and repetition "
              "therefore refer to that scope. Its selection used held-out exposure/coverage (not correctness), "
              "so findings apply to this purposive subset rather than an untouched random test sample.", "",
              "Paired differences and percentile bootstrap 95% intervals are in paired_comparison.csv. "
              "The resampling unit is the student, not an individual recommended question. "
              "Intervals describe student sampling uncertainty only; they do not measure LLM run-to-run variation. "
              "Small groups and rubric-based scores limit conclusions about learning effectiveness.", "",
              "baseline_random_expected is an exact mathematical expectation for uniformly choosing ten "
              "distinct questions from the full bank, including seen questions. It is not a model run, "
              "and has no ranked recommendation rows. Averaging over the full bank avoids random-seed noise.", "",
              "per_student_differences.csv lists paired scores and winners. paired_comparison.csv includes "
              "wins/ties/losses, raw A-minus-B differences, and benefit A-minus-B (positive always favours A). "
              "Model A and B are ordered alphabetically; read their names in each row.", "",
              f"Practical thresholds: targeting {settings.targeting_threshold:g}; repeat rate "
              f"{settings.repeat_threshold:g}. These are project choices, not established standards. "
              "A 0.10 difference is one question per ten on average. Difficulty has no automatic practical threshold. "
              "Freeze thresholds before model results are inspected. Intervals are unadjusted for multiple "
              "comparisons; subgroup and baseline comparisons are exploratory. Inconclusive intervals do "
              "not establish equivalence. Purposive selection limits population generalisation.", "",
              "Checksums and exact settings are recorded in run_manifest.json. Validation checks submitted "
              "files, but cannot prove what data a model actually accessed or whether it was tuned on evaluation results.", ""]
    return "\n".join(lines)


def main(argv=None):
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=base / "data" / "evaluation.csv")
    parser.add_argument("--bank", type=Path, default=base / "data" / "question_bank.csv")
    parser.add_argument("--training", type=Path, default=base / "data" / "training.csv")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--recommendations", nargs="+", type=Path)
    inputs.add_argument("--demo", action="store_true", help="Generate a random smoke test, NOT model results")
    parser.add_argument("--out", type=Path, required=True, help="New or empty result directory")
    parser.add_argument("--checkpoints", type=Path, help="Optional explicit cutoff CSV; default: each student's last response")
    parser.add_argument("--min-skill-responses", type=int, default=2)
    parser.add_argument("--short-history-max", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3263)
    parser.add_argument("--expected-students", type=int, default=30)
    parser.add_argument("--targeting-threshold", type=float, default=.10)
    parser.add_argument("--repeat-threshold", type=float, default=.10)
    args = parser.parse_args(argv)
    try:
        require(args.min_skill_responses >= 1, "Minimum responses must be positive")
        require(all(math.isfinite(v) and 0 < v <= 1 for v in
                    (args.targeting_threshold, args.repeat_threshold)),
                "Practical thresholds must be finite and in (0, 1]")
        require(args.short_history_max >= 1 and args.expected_students >= 1,
                "History threshold and expected student count must be positive")
        require(args.bootstrap_samples >= 100, "Use at least 100 bootstrap samples")
        require(not args.out.exists() or (args.out.is_dir() and not any(args.out.iterdir())),
                f"Output directory {args.out} must be new or empty; choose another path")
        settings = Settings(args.min_skill_responses, args.short_history_max,
                            args.bootstrap_samples, args.seed,
                            args.targeting_threshold, args.repeat_threshold)
        bank = load_bank(args.bank)
        require(len(bank) >= K, f"Question bank must contain at least {K} questions")
        history = load_responses(args.history, bank)
        training = load_responses(args.training, bank)
        checks = validate_training(training, history, bank)
        require(len(history) == args.expected_students,
                f"Expected {args.expected_students} evaluation students; found {len(history)}")
        checkpoints = resolve_checkpoints(history, args.checkpoints)
        states = build_states(history, bank, checkpoints, settings)
        checks.update({
            "prior_history_rows": sum(s["history_length"] for s in states.values()),
            "history_length_min": min(s["history_length"] for s in states.values()),
            "history_length_max": max(s["history_length"] for s in states.values()),
            "students_with_tied_weakest_skills": sum(s["weakest_skill_tie"] for s in states.values()),
            "short_history_students": sum(s["history_group"] == "short" for s in states.values()),
            "long_history_students": sum(s["history_group"] == "long" for s in states.values()),
        })
        if args.demo:
            rng = random.Random(args.seed)
            qids = sorted(bank, key=id_key)
            require(len(qids) >= K, f"Smoke test requires at least {K} bank questions")
            models = {model: {sid: [
                {"rank": i, "question_id": qid}
                for i, qid in enumerate(rng.sample(qids, K), 1)]
                for sid in sorted(states, key=id_key)} for model in ("demo_a", "demo_b")}
        else:
            models = load_recommendations(args.recommendations, states, bank)
        student_rows, question_rows = score(models, states, bank, settings)
        student_rows.extend(random_expected_scores(states, bank, settings))
        summary = aggregate(student_rows)
        comparisons = paired_comparisons(student_rows, settings)
        differences = student_differences(student_rows)
        skill_rows = []
        for sid in sorted(states, key=id_key):
            state = states[sid]
            for skill in sorted(state["proficiency"], key=id_key):
                skill_rows.append({"student_id": sid, "skill_id": skill,
                                   "prior_responses": state["counts"].get(skill, 0),
                                   "proficiency_beta_mean": state["proficiency"][skill],
                                   "eligible_for_weak_set": int(skill in state["eligible"]),
                                   "is_weak_skill": int(skill in state["weak"])})
        files = [args.history, args.bank, args.training, *(args.recommendations or [])]
        if args.checkpoints:
            files.append(args.checkpoints)
        manifest = {
            "evaluator_version": "2.0.0", "status": "PASS", "smoke_test_only": args.demo,
            "evaluator_sha256": fingerprint(__file__)["sha256"],
            "settings": {**settings.__dict__, "k": K,
                         "weak_definition": "lowest smoothed proficiency among eligible skills; include all exact ties",
                         "unobserved_skill_proficiency": 0.5,
                         "checkpoint_default": "last available response per student",
                         "difficulty_match": "1 - abs(proficiency - training difficulty)"},
            "dataset_checks": checks, "models": sorted(models),
            "baseline": {"id": RANDOM_BASELINE, "method": "exact expectation over full bank; uniform sampling without replacement"},
            "inputs": [fingerprint(path) for path in files],
        }
        args.out.mkdir(parents=True, exist_ok=True)
        write_csv(args.out / "per_student.csv", student_rows)
        write_csv(args.out / "per_recommendation.csv", question_rows)
        write_csv(args.out / "summary.csv", summary)
        write_csv(args.out / "student_skill_reference.csv", skill_rows)
        write_csv(args.out / "checkpoints.csv", [
            {"student_id": sid, "checkpoint_order_id": checkpoints[sid]} for sid in sorted(states, key=id_key)])
        write_csv(args.out / "paired_comparison.csv", comparisons, COMPARISON_FIELDS)
        write_csv(args.out / "per_student_differences.csv", differences, DIFFERENCE_FIELDS)
        if args.demo:
            write_csv(args.out / "smoke_test_recommendations.csv", [
                {"model_id": model, "student_id": sid, **rec}
                for model, lists in models.items() for sid, recs in lists.items() for rec in recs])
        (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (args.out / "report.md").write_text(report_text(summary, checks, settings, args.demo, comparisons), encoding="utf-8")
        print(f"PASS: {len(models)} model(s), {len(states)} students, {len(question_rows)} recommendations.")
        if args.demo:
            print("SMOKE TEST ONLY: randomly generated recommendations, not BKT/LLM performance.")
        print(f"Results: {args.out.resolve()}")
        return 0
    except (ValidationError, OSError, csv.Error) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
