Regime-Adaptive Machine Learning Does Not Beat Rule-Based Strategies in NIFTY Options Trading
P. Rameswara Anand¹ and O. Sri Nagesh²

¹ Research Scholar, Department of Computer Science and Engineering, Anurag University, Hyderabad, India
dr.pr.anand25@gmail.com | https://orcid.org/0009-0000-8057-7034

² Associate Professor, Department of Computer Science and Engineering, Anurag University, Hyderabad, India
nagesh.osri@gmail.com | https://orcid.org/0000-0003-3344-9413

Abstract
Previous work in our research program established that static machine learning (ML) fails to produce tradeable edges in NIFTY 50 options after transaction costs [1]–[4], and that adding ML to an already-profitable rule-based strategy degrades its performance [5]. A natural hypothesis is that ML might succeed if it were made regime-aware — that is, if a classifier first identified the current market regime and then routed the decision to a sub-model trained specifically for that regime. This paper tests that hypothesis rigorously. We build a rule-based regime labeler, train a LightGBM regime classifier (99.8% out-of-sample accuracy), train five regime-specific sub-models for each of two trading structures (bull put credit spread for selling, and 0.30-delta directional option buying), and backtest the full ensemble on NIFTY daily data from 2019 to 2026 with full transaction costs. The adaptive architecture fails on both sides of the market. For selling, adaptive ML produces a Sharpe of 1.047 versus 3.488 for the rule-based baseline; average per-trade P&L is ₹2,199, less than two-thirds of the rule-based baseline's ₹3,835. For buying, adaptive ML produces a Sharpe of −0.102 with total P&L of −₹89,579 — worse than random. Sub-model AUCs average 0.51 across all regimes in the buying case, statistically indistinguishable from chance. We conclude that the failure of ML in NIFTY options is not caused by regime blindness. The problem is structural: trade P&L is fundamentally unpredictable from price features, regardless of regime conditioning. Regime labels describe market state, not future direction. We recommend that future work in this area focus on cost-aware rule-based strategies rather than increasingly complex ML architectures.

Keywords: regime-adaptive machine learning; NIFTY options; negative results; reinforcement learning; rule-based trading; transaction costs; regime classification

1. Introduction
Our research program has produced a consistent set of findings over the last several years. Papers 1–4 established that machine learning models trained on price-derived features fail to produce tradeable edges on NIFTY 50 intraday options after accounting for realistic transaction costs [1]–[4]. The failure is not specific to any single model class, feature set, or training window; it appears across gradient-boosted trees, deep networks, and stacked ensembles alike. Paper 5 showed that the opposite approach — a rule-based bull put credit spread strategy gated by three regime filters — achieves a Sharpe ratio of 2.35 over seven years with a maximum drawdown of −15.0% [5]. Paper 6 demonstrated that adding ML to this rule-based strategy degrades its performance in nine of ten ablation configurations [6]. Paper 7 showed that a same-expiry tail hedge improves the rule-based strategy further, raising Sharpe to 2.84 and reducing maximum drawdown to −4.9% [7].

These findings raise an obvious follow-up question. Was the failure of ML caused by the models' inability to recognize the current market regime? A common practitioner intuition is that ML fails because it treats all market days as exchangeable when they are not — a trending market behaves differently from a choppy market, and a choppy market behaves differently from a crash. If a regime classifier were placed upstream of the ML model, the argument goes, each ML sub-model could specialize on a narrow regime and produce a genuine edge.

This paper tests that hypothesis directly. We build a regime-adaptive ML architecture and compare it against both the static ML baselines of Papers 1–4 and the rule-based strategy of Paper 5. We test the architecture on two trading structures:

Option selling: the bull put credit spread of Paper 5.

Option buying: directional 0.30-delta calls and puts.

The results are unambiguous. Adaptive ML fails on both sides. The failure is not caused by regime blindness. The problem is deeper.

The paper makes three contributions. First, it provides the first rigorous test of regime-adaptive ML on NIFTY options. Second, it demonstrates that the failure mode is structural — trade P&L is not predictable from price features within any regime. Third, it offers a clear recommendation for future work: focus on rule-based strategies rather than ML architectures.

2. Related Work
2.1 The Volatility Risk Premium and Rule-Based Option Selling
The volatility risk premium — the persistent tendency of implied volatility to exceed realized volatility — is one of the most robust empirical regularities in equity index options markets [8, 9]. Rule-based credit spread strategies that harvest this premium are well-documented and have been studied extensively in the academic literature [10, 11]. The defining feature of these strategies is that they do not attempt to predict direction; they exploit the structural bias in option prices.

Our Paper 5 [5] applied this framework to NIFTY 50 weekly options, showing that a bull put credit spread gated by three regime filters (Bollinger Bandwidth percentile, Madrid Trend Spotter, and Ichimoku Cloud) produces a Sharpe of 2.35 over seven years. The edge does not come from prediction but from systematically selling overpriced volatility in favorable regimes.

2.2 Regime-Aware Machine Learning
The idea that market regimes affect the predictive power of ML models is well-established [12, 13]. Several papers have proposed regime-switching models that explicitly condition the prediction on an observable regime variable [14]–[16]. The typical architecture has two components: a regime classifier and a set of regime-specific sub-models.

The theoretical appeal is clear. If the market has multiple distinct modes of behavior and each mode has different predictive structure, then a model that switches between modes should outperform a single monolithic model. This is the intuition we test.

2.3 Machine Learning in Option Markets
Several recent papers have applied ML to options trading, with mixed results. Some report improvements in Sharpe ratios [17, 18]; others find that the improvements do not survive transaction costs [19, 20]. The literature on the Indian options market is sparse; Papers 1–4 of our program are, to our knowledge, among the most extensive ML studies in that specific market [1]–[4].

2.4 Negative Results in ML for Finance
The case for publishing negative results in ML for finance has been made recently by several authors [21, 22]. Our program has contributed to this literature with Papers 1–4, Paper 6, and now this paper [23].

2.5 Research Gap
No prior work has tested whether regime-adaptive ML can succeed where static ML has failed on NIFTY options. This paper fills that gap.

3. Methodology
3.1 Data
We use daily NIFTY 50 closing prices and India VIX closing values from 2019-01-03 to 2026-07-01 (1,829 trading days over 7.05 years). NIFTY data is sourced from NSE archives. India VIX is computed by NSE using the CBOE methodology [24]. Transaction costs follow the model from Papers 5–7: ₹130 per round trip for 2-leg structures, ₹195 for 3-leg structures.

3.2 Regime Definition
We define five regimes using a rule-based labeling function applied to NIFTY + VIX features at the close of each day:

CRASH: VIX percentile over trailing 252 days > 0.80

TREND_DOWN: VIX percentile in [0.60, 0.80] and NIFTY below the Ichimoku Cloud

TREND_UP: VIX percentile < 0.40, NIFTY above the Cloud, and Madrid Trend Spotter is Green

CHOP_LOW_VOL: VIX percentile < 0.40 and none of the above

CHOP_HIGH_VOL: everything else

Table 1 shows the distribution of regimes over the 1,829-day sample.

**Table 1: Regime distribution**

| Regime | Days | Share |
| --- | --- | --- |
| CHOP_LOW_VOL | 458 | 25.0% |
| CHOP_HIGH_VOL | 421 | 23.0% |
| CRASH | 296 | 16.2% |
| TREND_UP | 250 | 13.7% |
| TREND_DOWN | 153 | 8.4% |
| UNKNOWN | 251 | 13.7% |

3.3 Regime Classifier
We train a LightGBM multiclass classifier to predict the regime label from 13 features: Bollinger Bandwidth percentile, ATR ratio, RSI(14), EMA5–EMA13 spread, ROC over 5 and 20 days, distance from the Ichimoku Cloud midpoint, VIX level, VIX percentile rank, 5-day VIX change, above-cloud binary indicator, MTS Green binary indicator, and MTS Red binary indicator.

The classifier is trained on an expanding walk-forward basis with annual folds: train on 2019–2021, validate on 2022; train on 2019–2022, validate on 2023; and so on through 2026.

3.4 Sub-Models
For each regime, we train a LightGBM binary classifier that predicts whether a trade entered on a given day would be profitable. Two trade structures are tested:

Option Selling: bull put credit spread (short put at 0.30 delta, long put 500 points below, 30 DTE, 20-day hold, equity-curve stop from Paper 5).

Option Buying: directional 0.30-delta option purchase. The option type (call vs put) is chosen by the regime. Hold period 5 trading days with a 50% premium stop-loss.

Table 2 shows the feature sets used per regime for each trade structure.

**Table 2: Feature sets per regime**

| Regime | Selling Features | Buying Features |
| --- | --- | --- |
| TREND_UP | bb_w_pct, atr_ratio, ema_spread, roc_5, roc_20, rsi_14, vix_level | ema_spread, roc_5, roc_20, rsi_14, vix_level, vix_change_5 |
| TREND_DOWN | atr_ratio, roc_5, roc_20, vix_level, vix_change_5, dist_cloud | ema_spread, roc_5, roc_20, rsi_14, vix_level, vix_change_5, dist_cloud |
| CHOP_LOW_VOL | bb_w_pct, rsi_14, dist_cloud, vix_rank, atr_ratio | bb_w_pct, rsi_14, ema_spread, dist_cloud, vix_rank |
| CHOP_HIGH_VOL | atr_ratio, vix_level, vix_rank, vix_change_5, roc_5 | atr_ratio, rsi_14, roc_5, vix_level, vix_change_5 |
| CRASH | vix_level, vix_change_5, atr_ratio, dist_cloud, roc_5 | vix_level, vix_change_5, atr_ratio, dist_cloud, roc_5 |


3.5 Adaptive Trading Simulation
For each validation day:

The regime classifier predicts the current regime.

The corresponding sub-model predicts the probability of a profitable trade.

If the probability exceeds a threshold (0.65 for selling, 0.60 for buying), the trade is entered.

Trades are simulated using the exit rules described in §3.4.

![**Figure 1** — Regime-adaptive ML architecture. Each validation day is routed by the regime classifier to one of five regime-specific sub-models (selling and buying sides shown in parallel). A trade is entered only if the sub-model's predicted probability exceeds the side-specific threshold (0.65 selling, 0.60 buying).](figures/figure1.png)

3.6 Performance Metrics
For each configuration we compute: number of trades, win rate, average P&L per trade, total P&L, Sharpe ratio, Calmar ratio, and maximum drawdown. All metrics are computed over the concatenated out-of-sample periods 2022–2026.

4. Results
4.1 Regime Classifier Performance
Table 3 shows the regime classifier's out-of-sample accuracy.

**Table 3: Regime classifier accuracy (label reproduction)**

| Fold | Train N | Val N | Accuracy | F1 Macro |
| --- | --- | --- | --- | --- |
| 1 (2022) | 476 | 247 | 0.996 | 0.997 |
| 2 (2023) | 723 | 245 | 1.0 | 1.0 |
| 3 (2024) | 968 | 244 | 0.992 | 0.992 |
| 4 (2025) | 1212 | 247 | 1.0 | 1.0 |
| 5 (2026) | 1459 | 119 | 1.0 | 1.0 |
| **Mean** | — | — | **99.8%** | **0.998** |
 Regime identification is essentially solved.

4.2 Sub-Model Performance — Option Selling
Table 4 shows the mean AUC of the selling sub-models.

**Table 4: Selling sub-model AUC by regime**

| Regime | Mean AUC | Folds |
| --- | --- | --- |
| TREND_DOWN | 0.781 | 3 |
| CRASH | 0.661 | 4 |
| CHOP_HIGH_VOL | 0.536 | 5 |
| TREND_UP | 0.487 | 4 |
| CHOP_LOW_VOL | 0.407 | 4 |
| **Average** | **0.574** | — |

Only TREND_DOWN has an AUC meaningfully above 0.70. Two regimes have AUCs below 0.50. Average across all regimes: 0.574.

4.3 Sub-Model Performance — Option Buying
Table 5 shows the same for the buying sub-models.

**Table 5: Buying sub-model AUC by regime**

| Regime | Mean AUC | Folds |
| --- | --- | --- |
| CRASH | 0.548 | 4 |
| TREND_DOWN | 0.541 | 3 |
| CHOP_HIGH_VOL | 0.503 | 5 |
| TREND_UP | 0.484 | 4 |
| CHOP_LOW_VOL | 0.454 | 4 |
| **Average** | **0.506** | — |

Every regime has an AUC within 0.05 of 0.50. Average: 0.506. No predictive power.

4.4 Main Comparison — Option Selling
Table 6 shows the out-of-sample performance of the four selling configurations.

**Table 6: Selling performance comparison**

| Config | Trades | Win % | Avg P&L | Total P&L | Sharpe | Max DD |
| --- | --- | --- | --- | --- | --- | --- |
| A — Rule-based | 486 | 83.2% | ₹3,835 | ₹18,63,714 | 3.488 | −25.7% |
| B — Static ML | 221 | 71.9% | ₹2,143 | ₹4,73,568 | 0.849 | −45.2% |
| C — Adaptive ML | 226 | 64.6% | ₹2,199 | ₹4,96,936 | 1.047 | −42.9% |
| D — Adaptive + Rule | 226 | 64.6% | ₹2,199 | ₹4,96,936 | 1.047 | −42.9% |

The rule-based baseline dominates every ML configuration. Adaptive ML produces a Sharpe of 1.047 (a 70% drop) and per-trade P&L of ₹2,199 versus ₹3,835 for the rule-based baseline.

![**Figure 2** — Out-of-sample Sharpe ratios for option selling (left) and option buying (right). The rule-based baseline (Sharpe 3.488) dominates every ML variant on the selling side; on the buying side all configurations are negative.](figures/figure2_sharpe.png)


4.5 Main Comparison — Option Buying
Table 7 shows the corresponding results for buying.

**Table 7: Buying performance comparison**

| Config | Trades | Win % | Avg P&L | Total P&L | Sharpe | Max DD |
| --- | --- | --- | --- | --- | --- | --- |
| A — Static Buying | 996 | 38.1% | −₹414 | −₹4,12,458 | -0.263 | −166.6% |
| B — Adaptive Buying | 170 | 40.2% | −₹527 | −₹89,579 | -0.102 | −59.8% |

Both configurations lose money. Adaptive buying improves the Sharpe slightly but remains negative.

4.6 Cross-Regime Analysis of Buying
Table 8 shows the raw trade-level win rates and average P&L by regime, before any ML filtering.

**Table 8: Raw buying performance by regime**

| Regime | N | Win Rate | Avg P&L |
| --- | --- | --- | --- |
| TREND_UP | 250 | 48.8% | +₹451 |
| TREND_DOWN | 153 | 42.5% | +₹1,417 |
| CHOP_LOW_VOL | 458 | 40.2% | +₹8 |
| CHOP_HIGH_VOL | 246 | 36.6% | −₹519 |
| CRASH | 296 | 28.4% | −₹965 |
| **Total** | **1,403** | **38.8%** | **−₹57** |

Two regimes (TREND_UP and TREND_DOWN) produce positive average P&L. Two regimes (CHOP_HIGH_VOL and CRASH) produce large losses. The regime classification correctly identifies which regimes are tradeable — but the ML sub-models cannot improve on this observation.


4.7 AGPro 1-2-3 Pattern Test
As a supplementary test, we evaluated whether the AGPro 1-2-3 reversal pattern could serve as an alternative regime filter. We reimplemented AGPro's detection logic in Python and identified 164 patterns over the seven-year sample. These were applied to bull put and bear call credit spreads.

**Table 9: AGPro pattern performance**

| Config | N | Win % | Sharpe | Max DD |
| --- | --- | --- | --- | --- |
| A — AGPro Bull (no filter) | 48 | 64.6% | -0.093 | −45.3% |
| B — Bull + Above Cloud | 43 | 65.1% | -0.259 | −55.2% |
| C — AGPro Bear | 30 | 63.3% | -0.477 | −72.4% |
| D — Bear + Below Cloud | 19 | 57.9% | -0.424 | −43.4% |
| E — Both filtered | 61 | 62.3% | -0.468 | −88.9% |

Score correlation with P&L was 0.203 (bull) and 0.121 (bear) — weak but positive. The subset of trades with score >= 85 produced a Sharpe of 1.47, but with only 9 trades, this is not statistically reliable. AGPro fails as a tradeable strategy.

### 4.8 Statistical Significance

To verify that the reported differences are not artefacts of sampling variability, we computed per-trade significance tests on the 486 rule-based trades, 221 static-ML trades, and 226 adaptive-ML trades in the selling case, and on the 996 static-buying trades and 170 adaptive-buying trades in the buying case.

**Option selling.** The rule-based baseline produces a mean per-trade P&L of ₹3,835 with a t-statistic of 19.33 against zero (p < 10⁻¹⁵) and a 95% bootstrap confidence interval for the annualized Sharpe ratio of [6.94, 10.86]. Static ML produces a mean of ₹2,143 (t = 4.92, p = 8.7 × 10⁻⁷), and adaptive ML produces a mean of ₹2,199 (t = 5.11, p = 3.2 × 10⁻⁷). A bootstrap test of Sharpe ratio differences between rule-based and adaptive ML yields an observed difference of +6.36 with a p-value below 10⁻¹⁵. A Diebold–Mariano test on per-trade P&L produces a statistic of +3.18 (p = 1.5 × 10⁻³). The rule-based baseline is statistically distinguishable from every ML variant.

**Option buying.** Static buying produces a mean per-trade P&L of −₹414 (t = −1.56, p = 0.12); adaptive buying produces −₹527 (t = −0.79, p = 0.43). Neither is statistically distinguishable from zero at the 5% level, but both bootstrap confidence intervals for the Sharpe ratio include zero (static: [−1.68, +0.17]; adaptive: [−1.41, +0.50]) and lie entirely below 1.0. A bootstrap test of Sharpe differences between the two configurations yields an observed difference of −0.35 with a p-value of 0.59; a Diebold–Mariano test gives a statistic of +1.17 (p = 0.24). Neither test rejects the null that the two configurations are statistically equivalent on the tested data.

**Interpretation.** The selling-side result is unambiguous: adaptive ML underperforms the rule-based baseline by a statistically significant margin. The buying-side result is also unambiguous in a different way: neither configuration produces a positive Sharpe, and adaptive ML does not improve on static ML by a statistically detectable amount. Both results are consistent with the paper's central claim that regime-adaptive ML does not beat rule-based strategies in NIFTY options trading.

5. Discussion
5.1 The Central Finding
Regime-adaptive ML does not beat rule-based strategies in NIFTY options trading after transaction costs, on either the selling or the buying side. The failure cannot be attributed to regime misclassification (99.8% accuracy) or insufficient data (150–460 trades per regime).

Three observations support this conclusion. First, sub-model AUCs hover near 0.50 across nearly all regimes in the buying case. Second, in the selling case, even where AUCs are above 0.50, the improvement does not translate into higher Sharpe. Third, the trade-level P&L distribution by regime shows that regime labels capture a real statistical regularity — but the ML sub-models cannot improve on this basic structural observation.

5.2 Why Regime Awareness Does Not Help
Within any regime, the P&L of a trade is essentially unpredictable from price features. The regime label captures everything that is statistically learnable. What remains is noise. This is consistent with the efficient market hypothesis [25, 26].

5.3 The Wrong Objective Problem Revisited
In the selling case, the adaptive classifier cut trade count from 486 to 226 while reducing average P&L from ₹3,835 to ₹2,199. The trades the classifier retained were not just fewer — they were worse on average. This echoes the finding in Paper 6.

5.4 Implications for Practitioners
Regime-adaptive ML should not be deployed by retail NIFTY options traders. The rule-based bull put credit spread from Paper 5, hedged with a same-expiry long put as in Paper 7, remains the recommended approach.

5.5 Limitations
Four limitations: single underlying; Black-Scholes pricing; sample excludes prolonged bear market; rule-based regime definitions rather than data-driven.

5.6 Future Work
Three directions: other indices (Bank NIFTY, SENSEX); actual option chain data; direct RL policies for regime-conditional decisions.

6. Conclusion
This paper has tested whether regime-adaptive machine learning can succeed where static ML fails in NIFTY options trading. We built a full pipeline: rule-based regime labeling, a LightGBM regime classifier (99.8% accuracy), and five regime-specific sub-models per trade structure. We tested two trade structures: bull put credit spreads (selling) and 0.30-delta directional options (buying).

The architecture fails on both sides. For selling, Sharpe drops from 3.488 to 1.047. For buying, Sharpe is −0.102 with total P&L of −₹89,579.

The failure is not caused by regime misclassification. It is caused by the fundamental unpredictability of trade P&L from price features within any regime.

We recommend that future work focus on rule-based strategies rather than increasingly complex ML architectures.

Data and Code Availability
All simulation code and derived data are available at https://github.com/dr-pr-anand25/nifty-bull-put-credit-spread. The regime-adaptive scripts are in the /paper8/ and /paper9/ directories.

References
[1] Anand, P. R., & Nagesh, O. S. (2025). Single-regime overfitting in NIFTY gamma blast studies [Working paper]. GitHub.

[2] Anand, P. R., & Nagesh, O. S. (2025). Regime-dependent feature importance in NIFTY options [Working paper]. GitHub.

[3] Anand, P. R., & Nagesh, O. S. (2025). Microstructure failure analysis of NIFTY signals [Working paper]. GitHub.

[4] Anand, P. R., & Nagesh, O. S. (2025). Reproducibility gaps in synthetic ML studies of NIFTY options [Working paper]. GitHub.

[5] Anand, P. R., & Nagesh, O. S. (2025). A cost-aware empirical investigation of bull put credit spreads on NIFTY 50 [Working paper]. GitHub.

[6] Anand, P. R., & Nagesh, O. S. (2025). When ML hurts: A rigorous ablation of ML enhancements to a profitable NIFTY options strategy [Working paper]. GitHub.

[7] Anand, P. R., & Nagesh, O. S. (2026). Tail-hedging improves risk-adjusted returns in NIFTY bull put credit spreads [Working paper]. GitHub.

[8] Bakshi, G., & Kapadia, N. (2003). Delta-hedged gains and the negative market volatility risk premium. The Review of Financial Studies, *16*(2), 527–566.

[9] Coval, J. D., & Shumway, T. (2001). Expected option returns. The Journal of Finance, *56*(3), 983–1009.

[10] Israelov, R., & Nielsen, L. N. (2015). Covered calls and tail risk. Financial Analysts Journal, *71*(4), 53–68.

[11] Frazzini, A., & Pedersen, L. H. (2012). Embedded leverage. NBER Working Paper 18558.

[12] Ang, A., & Timmermann, A. (2012). Regime changes and financial markets. Annual Review of Financial Economics, *4*, 313–337.

[13] Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, *57*(2), 357–384.

[14] Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. Journal of Economic Dynamics and Control, *31*(11), 3503–3544.

[15] Chen, S.-S., & Hsu, K.-W. (2010). Reverse globalization and value premium. Journal of Empirical Finance, *17*(5), 913–931.

[16] Tu, J. (2010). Is regime switching in stock returns important in portfolio decisions? Management Science, *56*(7), 1198–1215.

[17] Rasekhschaffe, K. C., & Jones, R. C. (2019). Machine learning for stock selection. Financial Analysts Journal, *75*(3), 70–88.

[18] Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, *33*(5), 2223–2273.

[19] Harvey, C. R., & Liu, Y. (2015). Backtesting. The Journal of Portfolio Management, *42*(1), 13–28.

[20] Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. Journal of Empirical Finance, *15*(5), 850–859.

[21] Kelly, B. T., & Xiu, D. (2023). Financial machine learning. Foundations and Trends in Finance, *13*(3–4), 205–363.

[22] López de Prado, M. (2018). Advances in financial machine learning. Wiley.

[23] Anand, P. R., & Nagesh, O. S. (2026). Regime-adaptive machine learning does not beat rule-based strategies in NIFTY options trading [Working paper]. GitHub.

[24] NSE. (2025). India VIX: Computation methodology. National Stock Exchange of India.

[25] Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. The Journal of Finance, *25*(2), 383–417.

[26] Malkiel, B. G. (2003). The efficient market hypothesis and its critics. Journal of Economic Perspectives, *17*(1), 59–82.

[27] Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. Quantitative Finance, *1*(2), 223–236.
