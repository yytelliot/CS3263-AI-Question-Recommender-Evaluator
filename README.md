# CS3263 evaluator v2

This version implements the agreed static top-10 experiment with **three** main
metrics. It replaces the earlier bottom-two/40% threshold weak-skill definition. It runs with
Python 3.10 or newer and needs no additional packages. The original evaluator
is kept separately.

## Quick start

Extract the complete ZIP. Put the two model output files beside `evaluate.py`,
then open a terminal in that extracted folder and run:

```sh
python evaluate.py --recommendations bkt_recommendations.csv llm_recommendations.csv --out results
```

Use `py` instead of `python` if that is how Python is installed on Windows.
The output directory must be new or empty. File arguments are relative to your
terminal folder; default dataset paths are relative to `evaluate.py`.

Start with `results/report.md` for the overview. Open the CSVs in Excel or read
them in Python for further analysis. These outputs cannot contain BKT/LLM results
until you supply their actual recommendations.

To try it before receiving recommendations:

```sh
python evaluate.py --demo --out demo_results
```

This creates two random fixtures, `demo_a` and `demo_b`, strictly for testing the
pipeline. They are not real models. A completed demo is included in
`example_results/`. Never use its numbers as BKT/LLM performance.

## Recommendation File contract

Each recommendation file needs:

```csv
model_id,student_id,rank,question_id
```

Each model must include all 30 evaluation students, with exactly ten rows per
student (300 rows total). Ranks must be 1 through 10 exactly once, and each
student's ten question IDs must be distinct and in the official bank. Use
different model IDs, such as `bkt` and `llm`.

Both models receive the same complete history for each evaluation student and
the same full 50-question bank. Previously attempted questions remain candidates;
repeat rate evaluates their selection. Give both teams the same novelty preference.
The ten recommendations form one list without new responses between ranks.

An optional `checkpoint_order_id` must match the evaluator's cutoff. By default,
this is each student's maximum supplied `order_id`; it does not need to be returned.
Other fields, including model scores, reasoning, mastery, skill IDs or difficulty,
are ignored for scoring. The evaluator joins the official metadata by question ID.

## Definitions to freeze before comparing model outputs

For each evaluation student u and skill s:

```text
n[u,s] = number of prior response rows for that skill
c[u,s] = number of correct prior responses
proficiency[u,s] = (c[u,s] + 1) / (n[u,s] + 2)
```

A skill is eligible for the weakest set when it has at least **two** responses.
Select every eligible skill tied for the **lowest** smoothed proficiency. Exact
rational comparison includes equal values even when response counts differ.
These are relative weakest skills, not evidence of failing an absolute standard.
In the supplied data, all 30 students qualify; three have tied lowest skills.

If a student has no eligible skill, the run fails with an explanation instead of
silently dropping them. Unobserved skills have proficiency 0.5 under the smoothing
formula; one response gives 1/3 or 2/3. These low-support values may enter difficulty
matching but cannot make the skill eligible for weakest-skill selection.

Question difficulty is `difficulty_beta_mean` from the bank:

```text
difficulty[q] = (incorrect TRAINING responses + 1) / (TRAINING response rows + 2)
```

The evaluator independently recomputes it from `data/training.csv` and checks
the stored support counts. It counts interaction rows, not the sum of `attempt_count`.

| Main metric | Per-student calculation | Preferred direction |
| --- | --- | --- |
| Weakest-skill targeting@10 | Number targeting any skill in the weakest set / 10 | Higher |
| Difficulty appropriateness@10 | Mean of `1 - abs(proficiency[u, skill(q)] - difficulty[q])` | Higher |
| Repeat rate@10 | Number of recommended IDs in that student's prior history / 10 | Lower |

Scores are calculated per student, then averaged with equal weight across students.
All ten ranks have equal weight. Coverage is omitted. A low-support recommendation
rate is reported separately as a diagnostic, not a fourth performance metric.

Difficulty matching is the chosen policy of giving lower-proficiency students
easier questions. It is a heuristic; proficiency and item difficulty are not a
calibrated common ability scale or an estimate of correctness probability.
Repeating a previously incorrect question may be useful revision: repeat rate
measures novelty, not demonstrated learning harm.

## Random baseline

Every run includes `baseline_random_expected`: exact expected scores for choosing
ten distinct questions uniformly from the full bank, without filtering seen questions.
For these three list-average metrics, the expected score equals the mean over
all 50 candidate questions. This is the analytical average of random lists,
so multiple seeds are unnecessary. It has no invented ranked recommendations.

## Output guide

| File | Use |
| --- | --- |
| `report.md` | Overall and subgroup summaries plus overall paired comparisons |
| `summary.csv` | Model/baseline averages for overall, short and long histories |
| `per_student.csv` | Each student's three scores, history length, weakest set and flags |
| `per_student_differences.csv` | Model A/B scores and differences for each student and metric |
| `paired_comparison.csv` | Average differences, bootstrap 95% intervals, wins/ties/losses and practical-effect checks |
| `per_recommendation.csv` | Auditable scoring details for each submitted recommendation |
| `student_skill_reference.csv` | Central proficiency estimates, evidence counts and weakest-set membership |
| `checkpoints.csv` | Cutoff used for each student |
| `run_manifest.json` | Version, settings, input/script hashes and validation results |

TL;DR: the most important outputs for the comparison are summary.csv, paired_comparison.csv and per_student.csv

The expected random baseline appears in student scores and comparisons but not
`per_recommendation.csv`, because it represents an expectation rather than a list.
The short group has 8-12 responses and the long group 13-20 in this dataset,
with 15 students each. This is an observational subgroup comparison, not a causal
test of additional history.

## Interpreting paired comparisons

Read `model_a` and `model_b`: pairs are alphabetically ordered and include baseline
comparisons. Raw differences are A minus B. Positive favours A for targeting and
difficulty; negative favours A for repeat rate. The separate
`benefit_a_minus_b` / `mean_benefit_a_minus_b` fields reverse repeat rate so positive
always favours A. Wins and losses follow the preferred metric direction.
Floating differences within 1e-12 count as ties.

Bootstrap draws resample students with replacement while keeping each student's
model scores together. Default: 5,000 draws, seed 3263. The sample size is 30
students, not 300 independent recommendations. Intervals are percentile intervals,
unadjusted for multiple comparisons; they are exploratory for subgroups and
baseline comparisons. With one student, confidence intervals are left unavailable.
Intervals do not capture model run-to-run randomness. Use distinct model IDs per
run if preserving repeat runs, but do not treat those as extra independent students.

The default practical threshold is 0.10 for targeting and repeat rate (one question
per top-10 list on average). This is a configurable project choice, not a scientific
standard. Difficulty has no automatic practical threshold. Freeze these values
before looking at model results. For example:

```sh
python evaluate.py --recommendations bkt_recommendations.csv llm_recommendations.csv --targeting-threshold 0.10 --repeat-threshold 0.10 --out results
```

`estimated_effect_meets_threshold` checks the absolute mean difference.
`interval_excludes_zero` describes directional uncertainty.
`interval_entirely_beyond_threshold` is stronger evidence that the entire interval
exceeds the chosen positive or negative practical threshold. A blank threshold
field means no threshold is defined. An interval containing zero is inconclusive;
it is not proof of equivalence. Report trade-offs instead of a weighted overall winner.

## Validation and limitations

Malformed files, missing students, unknown question IDs, duplicate questions or
ranks, checkpoint mismatches, training/evaluation student overlap, duplicate
response events, invalid correctness and mismatched response/bank skills fail
before scoring outputs are written. Existing nonempty result folders are rejected.
Use the supplied unchanged datasets for comparisons. Model-reported metadata cannot
override the official question metadata.

The supplied histories are filtered to 50 questions rather than complete five-skill
histories. The subset was selected for exposure/coverage using held-out data, so
it is a purposive pilot rather than an untouched random test sample. Paired
intervals are descriptive uncertainty summaries and do not establish broad
population representativeness. Metrics evaluate adherence to your defined policy,
not causal learning benefits.

The software cannot prove what data the models actually accessed or whether prompts
were tuned on evaluation results. Develop policies using training-only development
splits, then freeze them for the final comparison.

For earlier-cutoff experiments, use `--checkpoints cutoffs.csv` with one
`student_id,checkpoint_order_id` row per student. The cutoff must be an order ID
in that student's history. Both models must already have received only history
up to that cutoff. Truncating evaluator history cannot undo future data seen by a model.

Other options: `--training`, `--history`, `--bank`, `--min-skill-responses`,
`--short-history-max`, `--bootstrap-samples`, `--seed`, `--expected-students`.

Run verification:

```sh
python -m unittest discover -s . -p test_evaluate.py -v
```

Tests cover independent score arithmetic, all minimum ties, insufficient evidence,
future history exclusion, malformed model outputs, metadata overrides, training
provenance, paired directions, an exhaustive check of the random expectation,
and both CLI paths using the actual 30-student dataset.
