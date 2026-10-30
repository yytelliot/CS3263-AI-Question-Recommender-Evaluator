# Recommendation evaluation

SMOKE TEST ONLY: these randomly generated recommendations are not BKT or LLM results.

Validated 30 evaluation students; 50 candidate questions; no training/evaluation student overlap.

| Model | History group | Students | Weakest-skill targeting | Difficulty match | Repeat rate |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_random_expected | overall | 30 | 0.2200 | 0.7483 | 0.2393 |
| baseline_random_expected | short | 15 | 0.2267 | 0.7490 | 0.1787 |
| baseline_random_expected | long | 15 | 0.2133 | 0.7476 | 0.3000 |
| demo_a | overall | 30 | 0.2000 | 0.7446 | 0.2433 |
| demo_a | short | 15 | 0.1733 | 0.7387 | 0.2067 |
| demo_a | long | 15 | 0.2267 | 0.7505 | 0.2800 |
| demo_b | overall | 30 | 0.2367 | 0.7551 | 0.2300 |
| demo_b | short | 15 | 0.2200 | 0.7523 | 0.1600 |
| demo_b | long | 15 | 0.2533 | 0.7579 | 0.3000 |

## Overall paired comparisons

Raw difference = A minus B. For repeat rate, negative favours A; for other metrics, positive favours A.

| A | B | Metric | Mean difference | 95% interval | A wins / ties / B wins |
| --- | --- | --- | ---: | --- | --- |
| baseline_random_expected | demo_a | weakest_skill_targeting_at_10 | 0.0200 | [-0.0167, 0.0567] | 12 / 10 / 8 |
| baseline_random_expected | demo_a | difficulty_appropriateness_at_10 | 0.0037 | [-0.0098, 0.0189] | 14 / 0 / 16 |
| baseline_random_expected | demo_a | repeat_rate_at_10 | -0.0040 | [-0.0493, 0.0420] | 16 / 0 / 14 |
| baseline_random_expected | demo_b | weakest_skill_targeting_at_10 | -0.0167 | [-0.0600, 0.0233] | 10 / 9 / 11 |
| baseline_random_expected | demo_b | difficulty_appropriateness_at_10 | -0.0068 | [-0.0178, 0.0044] | 12 / 0 / 18 |
| baseline_random_expected | demo_b | repeat_rate_at_10 | 0.0093 | [-0.0380, 0.0553] | 12 / 3 / 15 |
| demo_a | demo_b | weakest_skill_targeting_at_10 | -0.0367 | [-0.0933, 0.0167] | 8 / 10 / 12 |
| demo_a | demo_b | difficulty_appropriateness_at_10 | -0.0105 | [-0.0288, 0.0079] | 11 / 0 / 19 |
| demo_a | demo_b | repeat_rate_at_10 | 0.0133 | [-0.0467, 0.0767] | 11 / 5 / 14 |

All scores range from 0 to 1. Lower repeat rate is better; higher is better for the other scores.
Results are macro averages: each student contributes equally.

Short histories contain at most 12 responses. The history-group comparison is descriptive; it does not establish a causal effect of history length.

## Definitions and interpretation

Weak skills: all skills tied for the lowest smoothed proficiency among skills with at least 2 prior responses. All exact ties are included and flagged in per_student.csv. These are relative weaknesses, even when the student's accuracy is high.

Proficiency = (correct + 1) / (responses + 2), using only supplied pre-checkpoint history. Unobserved skills receive a neutral 0.5 prior. Recommendations to skills below the support minimum are counted in low_support_recommendation_rate_at_10.

Question difficulty = (incorrect training responses + 1) / (training responses + 2). Every bank value was checked against the supplied training file.

Weakest-skill targeting: fraction of 10 recommendations targeting the weak set. Difficulty match: mean of 1 - abs(student skill proficiency - question difficulty). Repeat rate: fraction previously attempted by the same student.

Difficulty matching is a chosen policy heuristic; it is not a calibrated probability of answering a question correctly. Repeats can be useful revision, so repeat rate measures novelty rather than learning harm. Recommendation order does not affect these three metrics.

Recommendations remain eligible even if previously seen. Missing or invalid model outputs fail validation instead of being silently removed. No weighted overall winner score is calculated.

This subset contains histories filtered to the 50-question bank. Weakness and repetition therefore refer to that scope. Its selection used held-out exposure/coverage (not correctness), so findings apply to this purposive subset rather than an untouched random test sample.

Paired differences and percentile bootstrap 95% intervals are in paired_comparison.csv. The resampling unit is the student, not an individual recommended question. Intervals describe student sampling uncertainty only; they do not measure LLM run-to-run variation. Small groups and rubric-based scores limit conclusions about learning effectiveness.

baseline_random_expected is an exact mathematical expectation for uniformly choosing ten distinct questions from the full bank, including seen questions. It is not a model run, and has no ranked recommendation rows. Averaging over the full bank avoids random-seed noise.

per_student_differences.csv lists paired scores and winners. paired_comparison.csv includes wins/ties/losses, raw A-minus-B differences, and benefit A-minus-B (positive always favours A). Model A and B are ordered alphabetically; read their names in each row.

Practical thresholds: targeting 0.1; repeat rate 0.1. These are project choices, not established standards. A 0.10 difference is one question per ten on average. Difficulty has no automatic practical threshold. Freeze thresholds before model results are inspected. Intervals are unadjusted for multiple comparisons; subgroup and baseline comparisons are exploratory. Inconclusive intervals do not establish equivalence. Purposive selection limits population generalisation.

Checksums and exact settings are recorded in run_manifest.json. Validation checks submitted files, but cannot prove what data a model actually accessed or whether it was tuned on evaluation results.
