# Thesis Proposal v3 — Final

## Title

**FDR-Controlled Variable Selection under Parameter Instability: Comparing IPAD Knockoffs and OCMT with an Application to Equity Premium Prediction**

---

## Context and Motivation

Variable selection in high-dimensional regressions is a central problem in econometric forecasting. Two recent approaches address it from different angles:

1. **IPAD** (Fan, Lv, Sharifvaghefi, & Uematsu, 2020, JASA): uses Model-X knockoffs with a factor model for the covariate distribution to achieve FDR-controlled variable selection. Explicitly assumes stable parameters: "Our work has focused on the scenario of static models" (p. 1833).

2. **OCMT under instability** (Chudik, Pesaran, & Sharifvaghefi, 2024, J. Econometrics): studies the One Covariate at a Time Multiple Testing procedure under time-varying parameters. Shows OCMT selects variables with non-zero *average* effects and recommends full-sample selection with down-weighted forecasting. Provides no FDR control.

**Mahrad Sharifvaghefi is a co-author on both papers.** These are two lines of work by the same researcher attacking the same problem — high-dimensional variable selection for economic forecasting — from different frameworks. No paper compares them or asks: *under parameter instability, when should a forecaster prefer FDR-controlled selection (IPAD) over threshold-based selection (OCMT), and can we get the best of both?*

---

## Research Questions

1. **Diagnostic:** How does parameter instability affect the FDR control and power of IPAD knockoffs? How does it affect the false discovery behaviour of OCMT (which has no formal FDR guarantee)?

2. **Comparative:** Under structural breaks, which method provides better variable selection for forecasting — IPAD (FDR control, assumes stability) or OCMT (handles instability, no FDR control)?

3. **Constructive:** Can discounted IPAD — using exponentially weighted LCD statistics — bridge the gap, achieving both OCMT's robustness to breaks and IPAD's FDR control?

---

## Core Conceptual Contribution: The Dual-Null Separation

Under a structural break at $t^*$ with coefficients $\beta^{(1)}$ (pre-break) and $\beta^{(2)}$ (post-break), the relevant null hypothesis is ambiguous:

| Null | Definition | Targeted by | Relevant for |
|------|-----------|-------------|--------------|
| **Local null** | $H_0^{(2)} = \{j : \beta_j^{(2)} = 0\}$ | Discounted IPAD, Oracle | Forecasting at $T+1$ |
| **Average null** | $H_0^{\text{avg}} = \{j : T^{-1} \sum_t \beta_{t,j} = 0\}$ | OCMT (implicitly) | Detecting "ever-active" variables |
| **Full-sample null** | $H_0^{\text{full}} = \{j : \text{plim}\ \hat\beta_j^{\text{OLS}} = 0\}$ | Standard IPAD | Misaligned under breaks |

**This is not just a technicality.** The same method can appear FDR-controlled or inflated depending on which null is evaluated. A variable active in regime 1 but null in regime 2 is:
- A true discovery under $H_0^{\text{avg}}$ (OCMT perspective)
- A false discovery under $H_0^{(2)}$ (forecasting perspective)

This separation explains *why* IPAD and OCMT behave differently under breaks — they answer different inferential questions. Both the local and average nulls are evaluated and reported throughout.

---

## Theoretical Foundation

### Model-X Knockoffs (Candès et al., 2018)
Given i.i.d. observations $(X_i, Y_i)$, construct knockoff variables $\tilde{X}$ satisfying:
- Pairwise exchangeability: $(X, \tilde{X})_{\text{swap}(S)} \stackrel{d}{=} (X, \tilde{X})$ for all $S \subseteq \{1,\ldots,p\}$
- Conditional independence: $\tilde{X} \perp\!\!\!\perp Y \mid X$

Feature importance statistics $W_j$ satisfy the sign-flip property: $W_j(Y, [X, \tilde{X}]_{\text{swap}(S)}) = -W_j(Y, [X, \tilde{X}])$ for $j \in S$. The knockoff+ filter at target level $q$ provides finite-sample FDR control: $\text{FDR} \leq q$.

**Key point for this thesis:** The knockoff construction depends only on $P_X$, not on $P_{Y|X}$. Therefore, if $P_X$ is stable but $\beta_t$ changes, knockoffs remain valid — only the *signal detection* step is affected.

### IPAD (Fan et al., 2020)
Models covariates via a latent factor model $X_t = B F_t + U_t$. Estimates $\Sigma_X = BB' + \Sigma_U$ via PCA (with Ahn-Horenstein eigenvalue ratio for $\hat{r}$), enabling approximate knockoff construction. Uses LCD (Lasso Coefficient Difference) as the feature statistic:
$$W_j = |\hat\beta_j| - |\hat\beta_{j+p}|$$
where $(\hat\beta_1, \ldots, \hat\beta_{2p})$ is the Lasso estimate on the augmented design $[X, \tilde{X}]$. Provides asymptotic FDR control. **Assumes stable $\beta$.**

### OCMT (Chudik, Pesaran, & Sharifvaghefi, 2018; 2024)
Tests each covariate individually: regress $Y$ on $X_j$, obtain $t$-statistic, threshold at critical value $c_{p,T} = \Phi^{-1}(1 - p/(2\delta_T))$ where $p/\delta_T \to 0$. Under stability, OCMT consistently selects all signals and no noise. Under instability (2024), OCMT selects variables with non-zero *average* marginal effects. Recommends full-sample selection + down-weighted forecasting. **No FDR control is provided or claimed.**

---

## The Gap (Source-Grounded)

| Source | What it says | What remains open |
|--------|-------------|-------------------|
| Fan et al. (2020), p. 1833 | "Our work has focused on the scenario of static models" | What happens to IPAD FDR when $\beta$ changes? |
| Chudik et al. (2024), §5 | OCMT selects variables with non-zero average effects; recommends down-weighting at forecasting stage only | Can selection itself be localised? What is OCMT's empirical FDR? |
| Chi et al. (2025) — TSKI | Handles serial dependence; assumes stable $\beta$ | Orthogonal to this thesis (dependence ≠ instability) |
| Liu, Sun & Ke (2024) — GKnockoff | Uses knockoffs to *detect* change points | Does not do variable selection *despite* breaks |
| Barber, Candès & Samworth (2020) | Robust knockoffs: FDR ≤ $q \cdot e^\epsilon$ when $P_X$ is misspecified | Does not address $\beta$ instability (different source of misspecification) |

**No paper compares IPAD and OCMT under instability.** No paper studies discounted knockoff statistics. No paper formalises the dual-null issue.

---

## Methods

### Method 1: Standard IPAD (Baseline)
- PCA-based factor estimation with Ahn-Horenstein (2013) eigenvalue ratio for $\hat{r}$
- Second-order approximate knockoff construction from $\hat\Sigma_X$
- LCD statistics via Lasso (cross-validated $\lambda$) on augmented design $[X, \tilde{X}]$
- Knockoff+ filter at $q = 0.20$

### Method 2: OCMT (Baseline)
- Univariate OLS: $Y_t = \alpha_j + \gamma_j X_{t,j} + u_{t,j}$ for each $j = 1,\ldots,p$
- Threshold: select $j$ if $|t_j| > c_{p,T}$ with $c_{p,T} = \Phi^{-1}(1 - p/(2T^{1.1}))$
- Report empirical FDR (informative even without formal guarantee)

### Method 3: Discounted IPAD (Proposed Modification)

**Idea:** Replace the standard Lasso with an exponentially weighted Lasso:
$$\hat\beta^{(\lambda,\delta)} = \underset{b}{\arg\min}\ \sum_{t=1}^T \delta^{T-t}(Y_t - [X_t, \tilde{X}_t]' b)^2 + \lambda \|b\|_1$$

**Implementation:** Multiply observation $t$ by weight $\sqrt{\delta^{T-1-t}}$, then fit standard Lasso on the reweighted data. Cross-validate $\lambda$ on the reweighted sample.

**Sign-flip property (proof):**

> **Proposition.** If $w_t > 0$ are deterministic weights constructed independently of $\tilde{X}$, then the weighted LCD statistic $W_j^{(w)} = |\hat\beta_j^{(w)}| - |\hat\beta_{j+p}^{(w)}|$ satisfies the sign-flip property.
>
> *Proof.* Define the reweighted data $(Y^w, Z^w)$ where $Y_t^w = w_t Y_t$ and $Z_t^w = w_t [X_t, \tilde{X}_t]$. Since $w_t$ is deterministic and does not depend on $\tilde{X}$:
> 1. Conditional independence: $\tilde{X} \perp\!\!\!\perp Y \mid X$ implies $Z^w_{\cdot,j+p} \perp\!\!\!\perp Y^w \mid Z^w_{\cdot,1:p}$ (weighting preserves conditional independence).
> 2. Column exchangeability: $(X, \tilde{X})_{\text{swap}(S)} \stackrel{d}{=} (X, \tilde{X})$ implies $(Z^w_{\cdot,1:p}, Z^w_{\cdot,p+1:2p})_{\text{swap}(S)} \stackrel{d}{=} (Z^w_{\cdot,1:p}, Z^w_{\cdot,p+1:2p})$ since the same row-scaling is applied to both original and knockoff columns.
> 3. Therefore $W_j^{(w)}(Y^w, [Z^w]_{\text{swap}(S)}) = -W_j^{(w)}(Y^w, Z^w)$ for $j \in S$, by the standard sign-flip argument applied to the reweighted sample.  ∎

**Effective sample size:** The weighted Lasso behaves as if operating on $n_{\text{eff}}(\delta) = (1 - \delta^{2T})/(1 - \delta^2)$ effective observations. For $\delta = 0.97$, $T = 500$: $n_{\text{eff}} \approx 17$. This analytically predicts the power-localisation tradeoff.

**Discount factor selection:** Use a grid $\delta \in \{0.95, 0.97, 0.98, 0.99, 1.00\}$. In the empirical application, select $\delta$ by minimising out-of-sample forecast MSFE over a validation window (treating $\delta$ as a tuning parameter analogous to $\lambda$).

### Method 4: Rolling-Window IPAD
- Knockoff construction from full sample (valid: $P_X$ is stable)
- LCD statistics computed within a trailing window of size $h$
- Same knockoff+ filter applied to windowed LCD values
- Window sizes: $h \in \{0.3T, 0.5T, 0.7T\}$

### Method 5: Down-Weighted OCMT (Chudik et al., 2024 recommendation)
- Full-sample OCMT for variable selection (unweighted)
- Exponentially weighted OLS for post-selection forecasting
- This is the approach recommended by the source paper

### Method 6: Oracle Benchmark
- Knows the break date $t^*$
- Uses full sample for knockoff construction ($P_X$ stable)
- Restricts LCD statistics to post-break observations only
- Provides upper bound on what localised methods can achieve

---

## Data Generating Processes

### Covariate model (stable throughout):
$$X_t = B F_t + U_t, \quad F_t \sim N(0, I_r),\ U_t \sim N(0, \text{diag}(\sigma_1^2, \ldots, \sigma_p^2))$$

With $r = 3$ factors, loadings $B_{jk} \sim U(-1, 1)$, idiosyncratic variances $\sigma_j^2 \sim U(0.5, 1.5)$.

### Response model:
$$Y_t = X_t' \beta_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, 1)$$

### Scenarios (prioritised):

| Priority | Scenario | Mechanism | What it tests |
|----------|----------|-----------|---------------|
| **Primary** | Signal entry/exit | $S^{(1)} \neq S^{(2)}$: some variables active only pre-break, others only post-break | FDR inflation under local null; dual-null divergence |
| **Primary** | Stable (baseline) | $\beta$ constant | Sanity check: verify both methods work as advertised |
| **Secondary** | Magnitude shift | $\beta^{(2)} = c \cdot \beta^{(1)}$, same support | Power under signal dilution |
| **Secondary** | Sign reversal | $\beta^{(2)} = -\beta^{(1)}$ | Average effect = 0; OCMT should fail |

### Signal entry/exit specification (primary scenario):
- $|S^{(1)}| = |S^{(2)}| = s$ (sparsity level)
- Overlap: $|S^{(1)} \cap S^{(2)}| = s/2$ (half the signals persist, half switch)
- Non-zero coefficients: $\beta_j = A / \sqrt{s}$ where $A$ calibrated to target SNR

---

## Simulation Design

### Dimensionality configurations:

| Config | $(T, p, s)$ | Role |
|--------|-------------|------|
| Low-dim | $(500, 50, 5)$ | Sanity check; fast iteration |
| **Baseline** | **(500, 200, 20)** | **Primary results** |
| High-dim | $(300, 300, 15)$ | $p = T$ boundary |

### Primary grid (reported in main text):
- **Break location:** $\tau \in \{0.5, 0.7\}$ (midpoint; late break most relevant for forecasting)
- **Break magnitude:** controlled via overlap fraction $\in \{1.0, 0.75, 0.5, 0.25, 0.0\}$ (1.0 = no break, 0.0 = complete support switch)
- **SNR:** $\in \{2, 4\}$ (low, moderate)
- **Discount factor:** $\delta \in \{0.95, 0.97, 0.99, 1.00\}$
- **Replications:** $M = 500$

**Total primary configurations:** 2 × 5 × 2 × 4 = 80 cells × 6 methods × 500 reps. At ~0.5s per rep (Lasso on $n=500, p=400$), this is ~67 hours. Parallelisable to <8 hours on 8 cores.

### Secondary grid (reported in appendix):
- Window sizes for rolling IPAD: $h \in \{150, 250, 350\}$
- High-dim configuration
- Sign reversal scenario
- $M = 200$

### Evaluation metrics:

**Primary (reported for all methods):**
1. Empirical FDR against local null $H_0^{(2)}$ (the forecasting-relevant null)
2. Empirical FDR against average null $H_0^{\text{avg}}$
3. Power (TPR) against local null $H_0^{(2)}$
4. Out-of-sample $R^2_{\text{OOS}}$ (one-step-ahead forecast using post-selection OLS)

**Secondary:**
5. Selection stability: Jaccard similarity $J(\hat{S}_i, \hat{S}_j)$ averaged over replication pairs
6. FDR conditional on discovery: $\text{FDR} \mid |\hat{S}| > 0$ (avoids the 0/0 issue)

### Key analyses (section structure for results chapter):

1. **§4.1 Baseline replication.** Under stability ($\tau = 1.0$): verify IPAD controls FDR at $q = 0.20$, verify OCMT's empirical FDR. This is the sanity check.

2. **§4.2 Breakdown analysis.** Trace the FDR of standard IPAD as break severity increases. Identify the threshold where FDR first exceeds $q$. Plot: FDR vs. overlap fraction, stratified by SNR. *Expected finding:* FDR inflates when regime-1 signals contaminate post-break estimation.

3. **§4.3 Head-to-head comparison.** IPAD vs. OCMT: FDR–power frontiers under each scenario. *Expected finding:* IPAD dominates under stability; OCMT dominates under severe breaks; neither dominates uniformly.

4. **§4.4 Discounted IPAD.** Show that $\delta < 1$ restores FDR control at the cost of power (via $n_{\text{eff}}$ reduction). Plot the breakdown threshold curve in $(\text{overlap}, \delta)$ space. Compare empirical power to the analytical $n_{\text{eff}}$ prediction.

5. **§4.5 The dual null in action.** Show that the same run of discounted IPAD has different empirical FDR depending on whether evaluated against $H_0^{(2)}$ or $H_0^{\text{avg}}$. This is the conceptual contribution visualised.

6. **§4.6 Forecasting comparison.** $R^2_{\text{OOS}}$ across methods. Include DM test for pairwise forecast comparisons.

---

## Empirical Application: Equity Premium Prediction

**Committed scope** (not "if time permits" — this is planned for week 6):

### Data:
- Monthly equity premium: $r_{t+1}^e = r_{t+1}^{\text{mkt}} - r_f$
- Predictors: 14 standard Welch-Goyal (2008) variables + 3 lags of each = $p = 56$
- Sample: 1950:01–2023:12
- OOS evaluation: 1966:01–2023:12 (expanding window, minimum 15 years in-sample)

### Design:
- At each forecast origin $t$, apply: (i) IPAD, (ii) OCMT, (iii) Discounted IPAD ($\delta$ chosen by validation MSFE over prior 60 months)
- Post-selection forecast: OLS on selected variables (or weighted OLS for discounted methods)
- Record: selected variables, OOS forecast, realised premium

### Outputs:
1. **Selection heatmap:** Variables (rows) × time (columns), colored by selection frequency across 100 knockoff repetitions. Shows which predictors enter/exit.
2. **OOS $R^2$ comparison:** Cumulative $R^2_{\text{OOS}}$ for each method vs. historical mean benchmark.
3. **Break narrative:** Do variable selections shift around known instability episodes (1973 oil shock, 2008 GFC, 2020 COVID)?

**Note:** The student already has `ols_expanding_window.py` for this dataset. Marginal implementation cost: ~1 day for IPAD/OCMT wrappers.

---

## Maintained Assumptions

1. **Stable $P_X$:** The covariate distribution does not change at the break. Only $\beta_t$ changes. This isolates the coefficient instability channel and ensures knockoff validity.
2. **Single break:** The simplest form of instability. Discussion covers extensions to multiple breaks and gradual drift.
3. **Cross-sectional independence across $t$:** Observations are i.i.d. draws from the factor model within each period. Serial dependence is handled by TSKI (orthogonal contribution — explicitly out of scope).
4. **Known target FDR level:** $q = 0.20$ throughout (standard in the knockoff literature).

---

## Expected Contributions

| # | Contribution | Type | Strength |
|---|-------------|------|----------|
| 1 | First comparison of IPAD and OCMT under parameter instability | Empirical (MC) | Main result: who wins when? |
| 2 | Dual-null separation: IPAD and OCMT target different nulls | Conceptual | Framework-level insight; claimed regardless of MC outcomes |
| 3 | Discounted LCD statistics with formal sign-flip verification | Methodological | Simple but novel; extends knockoff toolkit |
| 4 | Breakdown threshold characterisation: when does IPAD's FDR guarantee fail? | Diagnostic | Practical guidance for applied researchers |
| 5 | Equity premium application with time-varying selections | Empirical | Connects to macro-finance literature |

---

## Related Work

### Must-cite (directly engaged):
- Candès, Fan, Janson, & Lv (2018) — Model-X knockoffs. *JRSSB* 80(3).
- Fan, Lv, Sharifvaghefi, & Uematsu (2020) — IPAD. *JASA* 115(532).
- Chudik, Pesaran, & Sharifvaghefi (2018) — OCMT original. *J. Econometrics* 203(2).
- Chudik, Pesaran, & Sharifvaghefi (2024) — OCMT under instability. *J. Econometrics*.
- Barber, Candès, & Samworth (2020) — Robust knockoffs. *Ann. Statist.* 48(3).
- Liu, Sun, & Ke (2024) — GKnockoff for structural change. *J. Econometrics* 239(2).

### Context (positioning):
- Chi, Fan, Ing, & Lv (2025) — TSKI. arXiv:2112.09851.
- Wei, Ke, & Zhang (2025) — FAR-Knockoff.
- Ren & Barber (2023) — Derandomised knockoffs. *JASA* 118(542).
- Nguyen, Chevalier, Thirion, & Arlot (2020) — AKO. *ICML*.
- Welch & Goyal (2008) — Equity premium prediction. *RFS* 21(4).
- Ahn & Horenstein (2013) — Eigenvalue ratio test. *Econometrica* 81(3).
- Rossi (2021) — Forecasting in the presence of instabilities. *JEL* 59(4).
- Ke, Liu, & Ma (2024) — Power of knockoff. *JMLR* 25(3).

### Aware-of (footnote or discussion):
- Barber & Candès (2015) — Fixed-X knockoffs. *Ann. Statist.*
- Romano, Sesia, & Candès (2020) — Deep knockoffs.
- Stock & Watson (2002) — Forecasting with many predictors.
- Pesaran, Pick, & Timmermann (2011) — Variable selection under model instability.

---

## Timeline (8 weeks)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Finalise proposal. Verify IPAD baseline (already done). Implement OCMT. Run stable-DGP sanity check. | Working comparison pipeline; Table 1 (baseline FDR/power). |
| 2 | Implement break DGPs. Run IPAD under instability. Document FDR inflation. | Figure: FDR vs. overlap fraction. §4.1–4.2 draft results. |
| 3 | Implement discounted IPAD + rolling-window IPAD. Run against oracle. | §4.3–4.4 results. Breakdown threshold figure. |
| 4 | Full simulation grid (primary). Generate all tables/figures. | Complete simulation results. |
| 5 | Secondary grid (high-dim, sign reversal). Begin empirical application. | Appendix tables. Application data pipeline working. |
| 6 | Finish empirical application. Write §4 (Results) and §5 (Application). | Draft results + application chapters. |
| 7 | Write §1 (Intro), §2 (Literature), §3 (Methods), §6 (Conclusion). | Full draft. |
| 8 | Revisions, polish, buffer for supervisor feedback. | Final submission. |

---

## Scope Management

### Hard scope (must deliver):
- Simulation comparison: IPAD vs. OCMT vs. Discounted IPAD vs. Oracle
- Primary grid: baseline + signal entry/exit scenario, $(500, 200, 20)$
- Sign-flip proof for discounted LCD
- Dual-null evaluation
- Empirical application (streamlined: 1 expanding window, 3 methods)

### Soft scope (deliver if time allows):
- High-dimensional configuration
- Rolling-window IPAD variant
- Magnitude shift and sign reversal scenarios
- Derandomisation (run each method 50× with different knockoff seeds; report stability)
- DM tests for forecast comparison

### Explicitly out of scope:
- Formal asymptotic theory for discounted IPAD FDR
- Serial dependence in $X_t$ (handled by TSKI — different paper)
- Joint $P_X$ + $\beta$ instability
- Deep generative knockoff construction
- Multiple simultaneous breaks (discussed qualitatively only)

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Discounted IPAD doesn't control FDR | Medium | The *comparison* is the main contribution, not the modification. Negative result → "discounting helps power but doesn't fully restore FDR" is still informative. Report oracle as upper bound. |
| Computation too slow | Low | Primary grid is ~67 CPU-hours. Parallelise on 8 cores (~8 hrs). Cut reps to $M=200$ if needed. |
| OCMT's FDR is accidentally good | Low | This would be an interesting finding. Report it honestly; attribute to Bonferroni-like conservatism in the critical value. |
| Empirical application shows no instability | Medium | The simulation study stands alone. Application becomes "methods agree → stability era" which is itself a finding. |
| Supervisor wants formal theory | Medium | Offer the $n_{\text{eff}}$ analytical prediction + sign-flip proof as the theoretical component. Acknowledge asymptotic FDR theory as future work. Frame thesis as "Monte Carlo evidence motivating future theoretical development." |

---

## What Makes This a Strong Thesis

1. **Dual contribution structure.** Even if discounted IPAD "fails," the comparison and dual-null insight remain. The thesis has a floor.

2. **Explicit connection between two published papers by the same co-author.** Examiners immediately see why this matters.

3. **Clean separation of concerns.** $P_X$ instability (→ TSKI) vs. $\beta$ instability (→ this thesis). The student can explain exactly what is and isn't in scope.

4. **Analytical prediction meets simulation.** The $n_{\text{eff}}(\delta)$ formula gives a testable quantitative prediction — if the simulation matches, it validates the mechanism; if it diverges, it reveals break contamination.

5. **Real data grounds the findings.** The equity premium application isn't an afterthought — it's where the instability story is most natural (everyone knows macro-finance relationships are unstable).

6. **Reproducibility.** Python code, fixed seeds, specified grid. Any examiner can replicate.
