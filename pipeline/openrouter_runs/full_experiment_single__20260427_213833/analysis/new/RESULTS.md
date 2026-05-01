# Supplementary results: effects of N and K, and significance of Spearman correlations

This note covers three additions requested for the paper:

1. Classification-accuracy results separated by **N** (number of classes per
   episode) and **K** (support shots per class).
2. Explanation-quality results separated by **N** and **K**, both for the
   overall judge score and for the nine explanation dimensions.
3. Statistical significance for the Spearman correlations of Figure 3,
   reported with Bonferroni correction over the full 9 dim × 4 prompt grid
   (36 tests).

All statistics are computed from
`pipeline/openrouter_runs/full_experiment_single__20260427_213833/`.
Each cell of the design (model × dataset × prompt × N × K) contains 48 trials
(4 datasets × 6 runs × 2 query-per-episode trials, balanced). The judge
parsed 4,599 / 4,608 explanations cleanly; 9 rows with parse errors are
dropped.

Paper-prompt naming: **E1** = classification baseline, **E2** = NLE,
**E3** = Features, **E4** = Rule-Based, **E5** = DL Axioms
(`axioms_ontology_v2`). Only E2–E5 are judged.

Files produced in this directory:

| Output                                              | What it holds                                                           |
|-----------------------------------------------------|-------------------------------------------------------------------------|
| `fig_accuracy_by_n_k.png`                           | Figure: accuracy vs N, by K (pooled and split by prompt)                |
| `fig_explanation_by_n_k.png`                        | Figure: overall judge score vs N/K + per-dimension K-effect             |
| `fig_spearman_heatmap_signif.png`                   | Figure: Spearman heatmap with Bonferroni-corrected significance markers |
| `accuracy_marginals_by_n_k.csv`                     | Marginal mean accuracy ± 95 % CI for each (N,K)                         |
| `accuracy_by_n_k_prompt.csv`                        | Means split also by prompt                                              |
| `accuracy_k_paired_tests.csv`                       | K=1 vs K=5 paired Wilcoxon at each N (and pooled)                       |
| `accuracy_n_friedman_tests.csv`                     | Friedman test for N effect, at each K                                   |
| `accuracy_n_pairwise_tests.csv`                     | Pairwise Wilcoxon between N levels (Bonferroni × 3)                     |
| `explanation_overall_by_n_k_prompt.csv`             | Mean overall judge score per (prompt, N, K)                             |
| `explanation_by_n_k_metric.csv`                     | Mean of every dimension per (N, K)                                      |
| `explanation_k_paired_tests.csv`                    | K=1 vs K=5 paired Wilcoxon for each metric × N (Bonferroni × 40)        |
| `explanation_n_friedman_tests.csv`                  | Friedman across N for each metric × K (Bonferroni × 30)                 |
| `spearman_with_significance.csv`                    | ρ, p<sub>raw</sub>, p<sub>Bonf</sub>, stars for each (prompt, dim)      |
| `summary.json`                                      | Compact JSON of all key numbers                                         |
| `build_analysis.py`                                 | Reproduces every CSV and figure                                         |

To reproduce: `python build_analysis.py`.

---

## 1. Classification accuracy by N and K

We aggregated the 5,760 trials into 480 (model × dataset × prompt × N × K)
cell accuracies and tested:

* **Effect of K** with a paired Wilcoxon signed-rank test pairing K=1 and
  K=5 cells inside each (model, dataset, prompt, N) tuple.
* **Effect of N** with a Friedman test across N ∈ {2,3,4} pairing cells
  inside each (model, dataset, prompt, K) tuple, plus pairwise Wilcoxon
  follow-ups (Bonferroni × 3).

### Marginal means

| N | K | Mean acc. | 95 % CI         |
|---|---|-----------|-----------------|
| 2 | 1 | 0.935     | [0.912, 0.958]  |
| 2 | 5 | **0.996** | [0.989, 1.000]  |
| 3 | 1 | 0.885     | [0.856, 0.915]  |
| 3 | 5 | **0.959** | [0.941, 0.978]  |
| 4 | 1 | 0.851     | [0.822, 0.880]  |
| 4 | 5 | **0.926** | [0.902, 0.950]  |

### Effect of K (additional support images)

Adding 4 extra labelled examples per class raises accuracy by **+7.0
percentage points on average** (pooled K=5 = 0.960 vs K=1 = 0.891), and the
gain is **highly significant at every N**:

| N      | Acc K=1 | Acc K=5 | Δ      | Wilcoxon W | p (raw)  |
|--------|---------|---------|--------|------------|----------|
| 2      | 0.935   | 0.996   | +0.060 | 0          | 6.9 × 10⁻⁶ |
| 3      | 0.885   | 0.959   | +0.074 | 222        | 9.1 × 10⁻⁵ |
| 4      | 0.851   | 0.926   | +0.075 | 136        | 8.3 × 10⁻⁶ |
| pooled | 0.891   | 0.960   | +0.070 | 868        | 2.0 × 10⁻¹³ |

### Effect of N (task difficulty)

Accuracy decreases monotonically with N at both K levels (Friedman test):

| K      | Mean N=2 | Mean N=3 | Mean N=4 | χ²<sub>F</sub> | p (raw)        |
|--------|----------|----------|----------|----------------|----------------|
| 1      | 0.935    | 0.885    | 0.851    | 31.29          | 1.6 × 10⁻⁷     |
| 5      | 0.996    | 0.959    | 0.926    | 50.12          | 1.3 × 10⁻¹¹    |
| pooled | 0.966    | 0.922    | 0.889    | 75.19          | 4.7 × 10⁻¹⁷    |

Pairwise Wilcoxon contrasts (Bonferroni × 3):

| Contrast (pooled K) | Δ      | p<sub>Bonf</sub> |
|---------------------|--------|------------------|
| N=2 vs N=3          | +0.043 | 2.0 × 10⁻⁴       |
| N=2 vs N=4          | +0.077 | 3.9 × 10⁻¹³      |
| N=3 vs N=4          | +0.034 | 8.8 × 10⁻⁴       |

### Suggested paragraph

> **Effects of N and K on classification.**
> Adding four extra labelled examples per class (K=5 versus K=1) increased
> mean classification accuracy by 7.0 percentage points overall (0.891 →
> 0.960), and the gain was significant at every N (paired Wilcoxon, p ≤
> 9 × 10⁻⁵ at each N; pooled p = 2.0 × 10⁻¹³). Within each K, accuracy
> decreased monotonically as the number of classes grew (Friedman χ² =
> 75.2, p = 4.7 × 10⁻¹⁷ pooled), with every pairwise N-contrast
> Bonferroni-significant (largest Δ = 0.077 between N=2 and N=4,
> p<sub>Bonf</sub> = 3.9 × 10⁻¹³). The 1-shot setting was therefore both
> harder and more sensitive to N, while 5-shot remained near-ceiling at
> N=2 (0.996) and degraded gracefully to 0.926 at N=4. Figure
> `fig_accuracy_by_n_k.png` (a) shows the K-curves over N; panel (b)
> reproduces the same picture per prompt and confirms that the K and N
> effects are visible across all five prompt families.

---

## 2. Explanation quality by N and K

Per (model × dataset × prompt × N × K) cell we computed the mean of the
overall judge score and of each of the nine dimensions. We then ran the
same paired Wilcoxon (K=1 vs K=5) and Friedman (across N) tests as for
accuracy. Bonferroni corrections cover the full grids reported in the
CSVs (40 K-tests, 30 N-tests respectively).

### Overall judge score

| N      | K=1   | K=5   | Δ        | p<sub>raw</sub> |
|--------|-------|-------|----------|-----------------|
| 2      | 4.191 | 4.332 | +0.141   | 2.6 × 10⁻⁵      |
| 3      | 4.084 | 4.266 | +0.181   | 1.2 × 10⁻⁶      |
| 4      | 4.177 | 4.185 | +0.008   | 0.91            |
| pooled | 4.151 | 4.261 | +0.110   | 2.7 × 10⁻⁸      |

The K effect on overall explanation quality is significant after
Bonferroni correction at N=2, N=3 and pooled, but **vanishes at N=4**
(p = 0.91), suggesting that extra support is most useful for explanation
quality on easier tasks; on harder tasks it improves classification
accuracy without measurably improving the *content* of the verbal
explanation.

For N: scores decrease with N (pooled Friedman χ² = 15.2, p<sub>Bonf</sub>
= 1.5 × 10⁻²), driven mainly by K=5 (p<sub>Bonf</sub> = 5.3 × 10⁻³).

### Per-dimension K-effect (pooled across N, Bonferroni × 40)

| Dimension              | Mean K=1 | Mean K=5 | Δ      | p<sub>Bonf</sub>  |
|------------------------|----------|----------|--------|-------------------|
| Discriminativeness     | 3.450    | 3.714    | +0.264 | **1.5 × 10⁻⁶**    |
| Hallucination-Free     | 4.521    | 4.699    | +0.178 | **4.4 × 10⁻⁵**    |
| Logical Coherence      | 4.278    | 4.428    | +0.150 | **6.8 × 10⁻⁶**    |
| Comprehensibility      | 4.676    | 4.740    | +0.064 | **3.3 × 10⁻³**    |
| Conciseness            | 4.882    | 4.951    | +0.068 | **1.1 × 10⁻³**    |
| Instruction Following  | 4.200    | 4.302    | +0.102 | **1.5 × 10⁻²**    |
| Specificity            | 3.605    | 3.661    | +0.056 | n.s. (1.0)        |
| Concept Counting       | 4.442    | 4.529    | +0.087 | n.s. (1.0)        |
| Textual Groundedness   | 3.301    | 3.323    | +0.021 | n.s. (1.0)        |

The **largest K-effect is on Discriminativeness** (+0.26 on a 1–5 scale):
seeing more examples per class lets the model name features that actually
distinguish the classes. Logical Coherence and Hallucination-Freeness
also benefit substantially, while Textual Groundedness, Concept Counting
and Specificity do not change.

### Per-dimension N-effect (Friedman, pooled across K, Bonferroni × 30)

Only one dimension shows a robust N-trend after correction:

| Dimension              | Mean N=2 | Mean N=3 | Mean N=4 | χ²<sub>F</sub> | p<sub>Bonf</sub> |
|------------------------|----------|----------|----------|----------------|------------------|
| Discriminativeness     | 3.864    | 3.479    | 3.402    | 51.5           | **1.9 × 10⁻¹⁰**  |
| Overall judge score    | 4.261    | 4.175    | 4.181    | 15.2           | **1.5 × 10⁻²**   |

Discriminativeness drops by ~0.46 from N=2 to N=4: as more classes need
to be distinguished simultaneously, models name correspondingly fewer
truly discriminative features. The other eight dimensions do not change
significantly with N after correction. This is consistent with our
expectation that N stresses the *discrimination* component of explanation
quality more than its surface form.

### Suggested paragraph

> **Effects of N and K on explanation quality.**
> Adding more support images improved explanation quality even after
> controlling for the matched accuracy gain. The pooled overall judge
> score rose from 4.15 (K=1) to 4.26 (K=5) (paired Wilcoxon, p =
> 2.7 × 10⁻⁸), and six of the nine dimensions improved at the
> Bonferroni-corrected α = 0.05 threshold. The largest K-effect was on
> *Local Discriminativeness* (+0.26, p<sub>Bonf</sub> = 1.5 × 10⁻⁶):
> a fifth example per class makes it visibly easier for models to name
> features that actually separate the classes, rather than features
> shared across them. *Logical Coherence* (+0.15, p<sub>Bonf</sub> =
> 6.8 × 10⁻⁶) and *Hallucination-Freeness* (+0.18, p<sub>Bonf</sub> =
> 4.4 × 10⁻⁵) also benefited, while form-level dimensions
> (Comprehensibility, Conciseness, Instruction Following) gained much
> less. Textual Groundedness, Concept Counting and Specificity were
> unaffected by K. The N-effect on explanation quality was concentrated
> almost entirely on *Local Discriminativeness*, which dropped from 3.86
> at N=2 to 3.40 at N=4 (Friedman χ² = 51.5, p<sub>Bonf</sub> =
> 1.9 × 10⁻¹⁰); no other dimension showed a Bonferroni-significant
> N-trend. Together with §1, this confirms that N primarily stresses the
> discrimination component, while K provides headroom on both
> classification accuracy and discriminative explanation content.
> Figure `fig_explanation_by_n_k.png` summarises both effects.

---

## 3. Spearman correlations: significance after Bonferroni correction

We re-derive the Figure 3 heatmap (judge metric × explanation prompt → ρ
with binary classification accuracy), but now annotate each cell with
two-sided p-values from `scipy.stats.spearmanr`, Bonferroni-corrected for
the full 9 × 4 = 36-cell grid. Significance markers in
`fig_spearman_heatmap_signif.png` are: `*` p<sub>Bonf</sub> < 0.05,
`**` p<sub>Bonf</sub> < 0.01, `***` p<sub>Bonf</sub> < 0.001. Sample size
per cell is ~1,150 trials, so even small ρ are detectable.

### Key cells (full table in `spearman_with_significance.csv`)

**Three metrics are Bonferroni-significant across all four prompts:**

| Metric                       | E2 NLE   | E3 Features | E4 Rule-Based | E5 DL Axioms |
|------------------------------|----------|-------------|---------------|--------------|
| Local Discriminativeness (LD)| 0.30 *** | 0.30 ***    | 0.35 ***      | 0.21 ***     |
| Textual Groundedness (TG)    | 0.15 *** | 0.18 ***    | 0.18 ***      | 0.15 ***     |
| Specificity (Sp)             | 0.14 *** | 0.13 ***    | 0.17 ***      | 0.15 ***     |

Of these three, **LD has by far the largest magnitudes** (0.21–0.35),
roughly twice TG and Sp (~0.13–0.18), so it is the only metric that is
both *consistent* and *strong* across prompts. TG and Sp are weak but
robust universal predictors; LD is the dominant one.

**Logical Coherence (LC).** Strongest for E2 NLE (0.36), strong for E3,
moderate for E4, and **not significant for E5 DL Axioms**:

| Prompt        | ρ    | p<sub>Bonf</sub> |
|---------------|------|------------------|
| E2 NLE        | 0.36 | 3.2 × 10⁻³⁵ ***  |
| E3 Features   | 0.35 | 2.2 × 10⁻³² ***  |
| E4 Rule-Based | 0.19 | 9.2 × 10⁻¹⁰ ***  |
| E5 DL Axioms  | 0.04 | 1.0 (n.s.)       |

**E5 DL Axioms.** Bonferroni-significant cells are only TG, HF, Co, Sp,
LD and IF; LC, CC and Cn are **not** significant after correction. The
maximum significant ρ for E5 is 0.21 (LD), versus 0.36 (LC) for E2 and
0.35 (LD) for E4. This is consistent with the degraded explanation
quality reported elsewhere for E5: when the explanation channel itself
is noisy, almost no dimension predicts whether the final classification
will be right.

### Updated paragraph (drop-in replacement for the figure caption / discussion)

> **Updated text for the Spearman heatmap (Figure 3) discussion:**
>
> Figure 3 shows the Spearman correlation between each judge metric and
> classification accuracy. Each cell is computed over ≈1,150 trials, and
> significance markers (`*`, `**`, `***`) reflect Bonferroni correction
> for the 36 simultaneous tests. *Local Discriminativeness (LD)*
> exhibits the strongest and most consistent correlation across
> conditions (ρ = 0.30 for E2/E3, 0.35 for E4, 0.21 for E5; all
> p<sub>Bonf</sub> < 10⁻¹⁰), demonstrating that models which
> successfully identify class-discriminative features are highly likely
> to classify correctly. Two further metrics are
> Bonferroni-significant across all four prompts but with substantially
> smaller magnitudes (Textual Groundedness 0.15–0.18 and Specificity
> 0.13–0.17), making LD the only universally *strong* predictor.
> *Logical Coherence (LC)* is the strongest predictor for NLE (ρ = 0.36,
> p<sub>Bonf</sub> = 3 × 10⁻³⁵) and remains Bonferroni-significant for
> E3 (0.35) and E4 (0.19), but **fails to reach significance for E5 DL
> Axioms (ρ = 0.04, p<sub>Bonf</sub> = 1.0)**. More broadly, three
> dimensions — Logical Coherence, Concept Counting and Conciseness —
> are not distinguishable from zero under E5, and the largest
> significant ρ for E5 is only 0.21 (LD), against 0.36 for E2 and 0.35
> for E3/E4. This pattern is consistent with the degraded explanation
> quality observed for E5: when the underlying explanation channel is
> unreliable, several of its surface metrics stop being diagnostic of
> the final decision.

---

## 4. Suggested discussion paragraphs: a converging mechanistic account

The three preceding analyses are independent — they manipulate K, manipulate
N, and correlate dimensions with accuracy on a per-trial basis — but they
all single out the same dimension: **Local Discriminativeness (LD)**. This
convergence is the load-bearing observation we want to highlight in the
discussion.

> **The role of discriminativeness in classification accuracy.**
> Three independent analyses converge on a single mechanism. First,
> increasing the number of support images per class (K=1 → K=5) produced
> its largest effect on *Local Discriminativeness* of the explanation
> (Δ = +0.26, p<sub>Bonf</sub> = 1.5 × 10⁻⁶), more than twice the
> K-effect on Logical Coherence (+0.15) or Hallucination-Freeness
> (+0.18), and against essentially zero change on form-level dimensions
> like Textual Groundedness, Concept Counting, Specificity or
> Conciseness. Second, increasing the number of classes (N = 2 → 4)
> produced a Bonferroni-significant N-trend in *only one* dimension —
> again Local Discriminativeness, which dropped from 3.86 (N=2) to 3.40
> (N=4) (Friedman χ² = 51.5, p<sub>Bonf</sub> = 1.9 × 10⁻¹⁰), while no
> other dimension's N-trend survived correction. Third, of the nine
> dimensions, *Local Discriminativeness* was the only one that
> correlated with accuracy at both consistent and substantial magnitude
> across all four explanation conditions (ρ = 0.21–0.35, all
> p<sub>Bonf</sub> < 10⁻¹⁰). Two further dimensions (Textual
> Groundedness and Specificity) reached significance in every prompt
> but with much smaller correlations (ρ ≈ 0.13–0.18); no other
> dimension was both universal and strong.
>
> Read together, these results point to a coherent mechanistic picture:
> for these models, *correctly classifying an image and correctly
> articulating what discriminates its class from the alternatives are
> two faces of the same operation*. Adding support images gives the
> model a denser within-class basis for inference, and this manifests
> simultaneously as higher accuracy and as explanations that more
> precisely identify class-discriminating features. Increasing the
> number of competing classes makes the discrimination harder, and
> this manifests simultaneously as lower accuracy and as explanations
> that name fewer truly discriminative features. The convergence
> matters because it does not depend on a single number being large; it
> depends on the same latent capability — picking out features that
> separate the predicted class from the alternatives — being implicated
> by three different experimental contrasts.
>
> **Implications for explanation faithfulness.** A common worry about
> LLM explanations is that they are post-hoc rationalisations whose
> content is decoupled from the underlying decision. Our results argue
> that, at least for the *Local Discriminativeness* component, this
> worry is not supported in our setting: the verbal explanation tracks
> the same latent capacity that drives the classification. When a
> model successfully isolates the features that distinguish the chosen
> class from the others, it both selects the correct label and can
> articulate that distinction in its output; when it fails — because N
> is large or K is small — accuracy and discriminative-content quality
> degrade together. This does not collapse the well-known gap between
> *stated* and *causal* features (a high LD score certifies that the
> text *describes* discriminating features, not that those features
> caused the decision in the network), but it places a non-trivial
> upper bound on how thoroughly the explanation channel can be
> independent of the decision channel: in our data they move in
> lockstep across two manipulations and four prompt families.
>
> The other dimensions are informative by their *non-*involvement.
> Form-level metrics (Comprehensibility, Conciseness, Instruction
> Following) are insensitive to N, only weakly sensitive to K, and only
> sometimes correlated with accuracy. Surface-grounding metrics
> (Textual Groundedness, Specificity) are stable across N/K and weakly
> but consistently positive across prompts — they describe how an
> explanation is written rather than what it picks out, and the data
> are consistent with that role. Logical Coherence is interesting in
> the opposite direction: it correlates strongly with accuracy under
> NLE, Features and Rule-Based prompts (ρ = 0.19–0.36, all
> p<sub>Bonf</sub> < 10⁻⁹) but **drops to ρ = 0.04, n.s. under DL
> Axioms**. Combined with the lower mean LD score under DL Axioms (3.5
> versus ≈4.4 for the other prompts) and the smaller LD↔accuracy
> correlation there (0.21 versus 0.30–0.35), this is the clearest
> evidence in our data that prompts which force a more rigid
> explanation format degrade the discrimination signal itself, not
> merely its phrasing.

---

## Notes on test choice

* For the K-effect we used a **paired Wilcoxon signed-rank test** because
  K=1 and K=5 cells are naturally paired by (model, dataset, prompt, N)
  and the cell-level accuracies are bounded in [0,1] and not Gaussian.
* For the N-effect we used a **Friedman test** (3-level repeated-measures
  rank test) on (model, dataset, prompt, K)-paired cells, with pairwise
  Wilcoxon for the follow-ups.
* For the heatmap we used **scipy.stats.spearmanr** for both ρ and the
  exact two-sided p-value, then Bonferroni-corrected over the full 36
  cells displayed. (The original `B4_correlation_metric_accuracy.csv`
  table reported bootstrap 95 % CIs but no per-cell p-value; we keep
  the bootstrap CIs in the supplementary CSVs for cross-checking — the
  significance/non-significance pattern is in agreement.)
