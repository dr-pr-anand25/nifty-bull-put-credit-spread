# A Cost-Aware Empirical Investigation of Bull Put Credit Spreads on NIFTY 50: Robust Positive Expected Value and the Role of Equity Curve Stops

**P. Rameswara Anand¹ and O. Sri Nagesh²**

¹ Research Scholar, Department of Computer Science and Engineering, Anurag University, Hyderabad, India
dr.pr.anand25@gmail.com | https://orcid.org/0009-0000-8057-7034

² Associate Professor, Department of Computer Science and Engineering, Anurag University, Hyderabad, India
nagesh.osri@gmail.com | https://orcid.org/0000-0003-3344-9413

---

## Abstract

Four earlier papers from our group documented a consistent result: directional option buying on NIFTY 50 using price-derived features does not clear the transaction-cost barrier. This paper asks the question that naturally follows. If buying does not work, does selling?

We test the bull put credit spread on NIFTY 50 weekly options over the 2019–2026 period. The strategy uses India VIX as the implied-volatility input to Black-Scholes, a Bollinger Bandwidth filter to detect volatility regimes, the Madrid Trend Spotter for direction, and the Ichimoku Cloud for trend confirmation. Over seven years, the strategy produces a Sharpe ratio of 1.37 and a maximum drawdown of −25.5%. Adding an equity-curve stop that pauses trading after three consecutive losses lifts the Sharpe ratio to 2.35 and cuts the drawdown to −15.0%. Total return over the period is +560.9% against +109.2% for NIFTY buy-and-hold. The strategy is robust across 87% of 192 parameter combinations tested [1]–[3]. We also document a structural weakness — sudden market reversals after extended trends — that the equity-curve stop mitigates but does not prevent.

**Keywords:** bull put credit spread; NIFTY 50; volatility risk premium; equity-curve stop; transaction costs; option selling

---

## 1. Introduction

Our earlier research reached a conclusion that surprised us. Four independent studies — a single-regime evaluation paper, a regime-dependent feature-importance paper, a microstructure failure analysis, and a reproducibility study — all pointed the same way. Price-derived machine learning models cannot produce a durable edge in NIFTY intraday option buying [1]–[4]. The evidence was consistent across model classes, time windows, feature sets, and evaluation methodologies.

If that is true, a natural question follows. Does the opposite side of the market work? Retail traders lose money buying options. Do they lose money selling them? The academic literature gives us some reason to hope they do not.

The Volatility Risk Premium (VRP) is one of the most robust empirical regularities in options markets [5]. Implied volatility is, on average, higher than realized volatility. Sellers of options collect this premium as a form of insurance payment for bearing tail risk. The premium exists across markets, across time periods, and across asset classes [6].

But a premium existing on paper does not mean a retail trader can capture it. Retail transaction costs in the NIFTY options market are high. The round-trip cost for a single lot can reach ₹130 after brokerage, STT, exchange fees, GST, SEBI fees, stamp duty, and slippage [7]. Any strategy that sells premium must collect enough to clear that bar.

That is the question this paper addresses. We test three structures — the bull put credit spread, the bear call credit spread, and the iron condor — over the 2019–2026 period with real India VIX as the implied-volatility input. Our answer is precise. The bull put credit spread produces positive expected P&L. The other two structures do not. The reason is directional alignment: NIFTY 50 has a long-term upward drift [8], and selling puts collects premium that is directionally favored by that drift.

We then test whether an equity-curve stop — pausing the strategy after three consecutive losses — improves the results. It does, substantially. The Sharpe ratio rises from 1.37 to 2.35, the maximum drawdown falls from −25.5% to −15.0%, and the total return rises by 37% [9].

The paper is organized as follows. Section 2 reviews related work. Section 3 describes the methodology. Section 4 presents the empirical results. Section 5 discusses implications and limitations. Section 6 concludes.

---

## 2. Related Work

### 2.1 The Variance Risk Premium

The empirical basis for options selling is the VRP. Carr and Wu formalized this as a return to bearing variance risk, showing that the payoff to a short variance position is positive on average across a wide range of underlying assets [5]. Bollerslev, Tauchen, and Zhou tied the premium to macroeconomic uncertainty, showing that it is highest when economic conditions are least predictable [6].

In the Indian market, the same pattern is visible. India VIX has averaged about 16% over the past decade, while realized volatility has averaged 12% to 14%. The gap represents the VRP. Whether it is large enough to cover retail transaction costs is the empirical question this paper addresses. Our earlier work did not test this directly [1]–[4].

### 2.2 Directional Bias in Equity Indices

Equity indices drift upward over long horizons. Fama's original work on market efficiency documented the persistent positive return to equity ownership [10]. Over the past two decades, NIFTY 50 has returned roughly 11% per year on an annualized basis. This drift has a direct consequence for options strategies. A strategy selling puts profits when the market stays flat or rises — that is the favored direction. A strategy selling calls profits when the market stays flat or falls — that is not [11].

Our earlier microstructure paper documented this asymmetry qualitatively [3]. The current paper confirms it quantitatively with a larger sample.

### 2.3 Options Selling in Retail Practice

SEBI's FY26 study found that the only category of retail traders with positive median returns were those selling options [7]. The median return was only 1%, but the direction is consistent with the VRP literature. The finding suggests that the structural edge of options selling is real, even if the median retail trader captures only a small fraction of it.

Academic work on retail options traders has mostly focused on the buying side, where losses are concentrated [12]. Selling strategies have received less attention, partly because they require more capital, more discipline, and more sophisticated risk management. This paper contributes to filling that gap.

### 2.4 Transaction Costs in NIFTY Options

Round-trip costs for a single NIFTY options lot (65 units) are substantial. Brokerage is typically ₹20 per side. STT applies at 0.125% on the sell-side premium. Exchange transaction fees are around 0.035% of premium per side. GST is 18% on brokerage and exchange charges. SEBI turnover fees and stamp duty add smaller amounts [13]. The largest component is slippage — ATM NIFTY options on expiry week trade with a bid-ask spread of 1 to 2 percentage points of premium.

The total round-trip cost per lot is roughly ₹90 to ₹120 under normal conditions and higher during volatile periods [14]. This cost is the primary constraint on profitability for any strategy selling short-dated premium. Our cost model includes every component and assumes 0.5% extra slippage per leg as a conservative estimate.

### 2.5 Equity Curve Stops in Systematic Trading

Equity-curve stops — pausing a strategy after a run of losses — have a long history in managed futures and systematic trading [15]. The idea is simple. If the strategy has started to lose money, the market regime may have changed. Pausing for a fixed period gives the regime time to shift again before trading resumes.

The mechanism is a form of regime detection through performance feedback. Instead of trying to predict the regime, the trader observes the strategy's P&L and reacts [16]. We test this mechanism specifically in the context of bull put credit spreads.

### 2.6 Comparison with Machine Learning Approaches

Recent work has applied machine learning to options trading with mixed results. Gu, Kelly, and Xiu surveyed ML applications in asset pricing and found that most reported high performance fails out-of-sample [17]. Lopez de Prado formalized the problem of backtest overfitting and its correction [18]. Our earlier papers documented the same pattern for NIFTY options specifically [1]–[4].

This paper takes a different approach. Rather than trying to predict direction, we sell a structured premium product in conditions where directional alignment favors the seller. The result is a strategy with a coherent economic rationale, not a black-box classifier.

---

## 3. Methodology

### 3.1 Data

We use daily NIFTY 50 closing prices from 2019-01-01 to 2026-07-01, totaling 1,829 trading days over 7.05 years. India VIX daily data is used as the implied-volatility input. Both datasets are publicly available from NSE archives [19].

### 3.2 Option Pricing

For each entry day, we use Black-Scholes to price the short and long put legs of the spread [20]. The implied-volatility input is India VIX divided by 100. The risk-free rate is set to 6.5%, approximately the India 10-year yield over the sample. The time to expiry is set to 30 days at entry.

The short strike is chosen to have delta 0.30 at entry. The long strike is 500 points below the short strike. This produces a spread with approximately ₹1,000 to ₹2,000 of collected credit per lot depending on the market.

### 3.3 Entry Conditions

A bull put credit spread is initiated on any day satisfying all three of the following conditions.

First, the Bollinger Bandwidth percentile over a 60-day lookback must be between 20% and 70% [21]. This excludes volatility extremes — both the squeeze that precedes a breakout and the wide regime that signals danger.

Second, the Madrid Trend Spotter must be Green. The 5-period EMA must be above the 13-period EMA, with the 3-bar change in the spread positive [22].

Third, the NIFTY close must be above both Senkou Span A and Senkou Span B of the Ichimoku Cloud [23].

These three conditions together capture the two things that matter for a bull put credit spread — a market that is not in a volatility extreme and a market with positive directional momentum.

### 3.4 Exit Conditions

The position is closed when either of the following occurs.

First, the stop-loss triggers. The spread value reaching 3.0 times the initial credit collected closes the trade. We tested 1.5×, 2.0×, 3.0×, and no stop. The 3.0× stop gave the best risk-adjusted result, because tighter stops exit the trade before it can recover.

Second, the time exit triggers. The position is held for 20 trading days, then closed at market.

### 3.5 Equity Curve Stop

In addition to the per-trade stop-loss, we test an equity-curve stop that pauses the strategy after a specified number of consecutive losing trades. After the pause, trading resumes. We test four configurations — pause after 3, 4, or 5 losses, and pause duration of 20, 30, or 40 trading days. The best configuration is reported in Section 4.

### 3.6 Cost Model

Round-trip costs per lot are modeled as follows.

| Component | Value |
|-----------|-------|
| Brokerage | ₹20 per side × 2 = ₹40 |
| STT (0.125% on sell premium) | Approximately ₹8 |
| Exchange transaction fee (0.035% per side) | Approximately ₹5 |
| GST (18% on brokerage + exchange) | Approximately ₹8 |
| SEBI turnover fee (0.0001% per side) | Approximately ₹0.5 |
| Stamp duty (0.003% on buy) | Approximately ₹1 |
| Slippage (0.5% per leg) | Approximately ₹50 to ₹100 |
| **Total round-trip cost per lot** | **₹130** |

The cost model is applied to every trade [24].

### 3.7 Performance Metrics

We compute seven metrics for each configuration. Number of trades and win rate. Average P&L per trade and total P&L. Sharpe ratio, annualized based on per-trade returns. Calmar ratio, computed as annualized return divided by maximum drawdown. Maximum drawdown, based on sequential equity curve. Best and worst single trade. Year-by-year breakdown.

---

## 4. Empirical Results

### 4.1 Baseline Strategy (No Equity Curve Stop)

Over the 7.05-year period, 212 bull put credit spread trades met the entry conditions. The results:

| Metric | Value |
|--------|-------|
| Trades | 212 |
| Trades per year | 30.1 |
| Win rate | 79.7% |
| Avg P&L per trade | ₹1,938 |
| Total P&L | ₹4,10,841 |
| Total return | +410.8% |
| Annual return | +24.0% |
| Max drawdown | −25.5% |
| Sharpe ratio | 1.372 |
| Calmar ratio | 2.305 |
| Worst trade | −₹26,746 |

The strategy produces a Sharpe ratio of 1.37 and a Calmar ratio of 2.31. Total return is +410.8% over the period, approximately 24% annualized.

### 4.2 Equity Curve Stop

Adding the equity-curve stop improves the results substantially.

| Config | Trades | Win rate | Avg P&L | Total P&L | Sharpe | Calmar | Max DD |
|--------|--------|----------|---------|-----------|--------|--------|--------|
| Baseline | 212 | 79.7% | ₹1,938 | ₹4,10,841 | 1.372 | 2.305 | −25.5% |
| Pause 3, 20 days | 201 | 83.1% | ₹2,725 | ₹5,47,736 | 2.169 | 5.211 | −15.0% |
| **Pause 3, 40 days** | **196** | **83.7%** | **₹2,862** | **₹5,60,906** | **2.348** | **5.336** | **−15.0%** |
| Pause 4, 20 days | 207 | 81.6% | ₹2,524 | ₹5,22,495 | 2.003 | 4.584 | −16.3% |
| Pause 5, 30 days | 207 | 81.2% | ₹2,358 | ₹4,88,023 | 1.787 | 3.471 | −20.1% |

The best configuration is pause after three losses, resume after 40 days. It produces a Sharpe ratio of 2.348 (up from 1.372), a Calmar ratio of 5.336 (up from 2.305), and a maximum drawdown of −15.0% (down from −25.5%). Total return rises from +410.8% to +560.9%.

The equity-curve stop eliminates the worst-case drawdown events without reducing the average return per trade. It is the single most effective risk-management improvement in the strategy.

### 4.3 Year-by-Year Breakdown

The yearly performance under the optimal equity-curve stop configuration:

| Year | Trades | Total P&L | Win rate |
|------|--------|-----------|----------|
| 2019 | 16 | +₹66,978 | 93.8% |
| 2020 | 27 | +₹1,00,292 | 88.9% |
| 2021 | 49 | +₹2,16,156 | 91.8% |
| 2022 | 26 | +₹1,02,861 | 88.5% |
| 2023 | 20 | +₹75,515 | 80.0% |
| 2024 | 25 | −₹6,771 | 68.0% |
| 2025 | 27 | +₹93,386 | 88.9% |
| 2026 | 6 | −₹87,512 | 0.0% |

Six of eight years are profitable. The two losing years — 2024 and 2026 — are associated with specific market events that the entry filter did not detect.

### 4.4 Parameter Robustness

We tested 192 parameter combinations varying delta (0.20, 0.25, 0.30), spread width (200 to 500 points), hold period (10 to 30 days), and stop-loss multiplier (1.5×, 2.0×, 3.0×, and no stop). The results:

- 167 of 192 combinations — 87% — produce positive expected P&L
- Median average P&L is +₹299
- Median Sharpe ratio is 0.117
- Best Sharpe ratio is 2.563
- Worst Sharpe ratio is −0.594

The parameter sensitivity shows clear patterns.

| Parameter | Optimal Value | Reasoning |
|-----------|---------------|-----------|
| Stop-loss | 3.0× premium | Tighter stops exit recovering trades |
| VIX filter | None | Filtering by VIX rank hurts performance |
| Delta | 0.30 | Higher delta collects more premium |
| Hold period | 20 days | Balanced between theta decay and price drift |
| Spread width | 400 to 500 points | Wider spreads reduce loss frequency |

The strategy does not depend on a single lucky parameter choice [25].

### 4.5 Benchmark Comparison

Over the same 7.05-year period, NIFTY 50 buy-and-hold produced:

| Metric | Value |
|--------|-------|
| Total return | +109.2% |
| Annual return | +13.11% |
| Annual volatility | 17.82% |
| Sharpe ratio | 0.736 |
| Max drawdown | −38.4% |
| Calmar ratio | 0.412 |

The bull put credit spread strategy with the equity-curve stop outperforms NIFTY buy-and-hold on every risk-adjusted metric.

### 4.6 Known Weaknesses

The strategy has a structural vulnerability that is worth documenting clearly. When the market reverses suddenly after an extended trend, the entry filter does not detect the regime change until three consecutive max-loss trades have been executed. In September 2024 and February 2026, this pattern produced consecutive losses of ₹23,000, ₹24,000, and ₹26,000 per trade.

The equity-curve stop triggers after the third loss, preventing further damage. But by then, the strategy has lost roughly ₹75,000 per reversal event. Future work should address this through more sophisticated regime detection [26].

---

## 5. Discussion

### 5.1 Why the Bull Put Credit Spread Works

The strategy works for three reasons.

First, it captures the Volatility Risk Premium [5]. Selling options on average collects a premium that exceeds the realized volatility of the underlying. This is a systematic feature of options markets.

Second, it aligns with the long-term upward drift of NIFTY 50 [8]. Selling puts collects premium that is directionally favored by the market's tendency to rise. Selling calls, which goes against the drift, does not work.

Third, it uses defined-risk structures. The 500-point spread limits the maximum loss to about ₹32,500 per lot, minus the premium collected. That makes the strategy survivable through adverse events.

### 5.2 Why the Equity Curve Stop Works

The equity-curve stop is a form of regime detection through performance feedback [15]. Instead of trying to predict when the market regime has changed, it waits for the strategy to accumulate losses and then pauses.

This is more robust than predictive regime detection because it does not depend on the accuracy of the regime filter. It uses the actual P&L of the strategy as the signal. When the strategy is working, no pause occurs. When the strategy starts losing, the pause activates.

The cost of the equity-curve stop is that it occasionally pauses during temporary drawdowns that would have recovered. In our backtest, the stop pauses the strategy 5 to 8 times over 7 years. The benefit — avoiding the worst-case drawdowns — clearly outweighs the cost.

### 5.3 Comparison with Our Earlier Findings

This paper represents a turning point in our research program. Papers 1 through 4 documented that directional option buying does not work [1]–[4]. The Cost-Edge paper formalized that result. This paper identifies a structure that does work on the tested data.

The distinction matters. Our earlier papers established that price-derived features and machine learning models cannot profitably time directional entries. This paper shows that structured premium selling can be profitable when the structure matches the market's directional bias, when entry conditions are disciplined, and when equity-curve risk management is used.

The two findings are compatible. One says you cannot predict direction. The other says you do not need to predict direction if you sell premium in the right structure.

### 5.4 Implications for Retail Traders

The practical implications are significant. Retail traders who have been losing money on directional option buying may find that a bull put credit spread strategy suits them better. The mindset required is different — patience, discipline, and acceptance of capped profits. The strategy does not produce spectacular wins. It collects small premiums repeatedly, punctuated by occasional stop-outs. The frequency of winning trades is high, between 80% and 84%, but the magnitude of the worst loss is meaningfully larger than the average win.

This is a business, not a game. It rewards consistency over brilliance.

### 5.5 Comparison with Academic Literature

Our findings are consistent with the academic literature on the VRP [5], [6] and with SEBI's observation that option sellers are the only profitable category of retail traders in Indian equity derivatives [7]. They extend the literature by documenting a specific strategy that works with parameter robustness and equity-curve risk management.

They do not establish that bull put credit spreads are universally profitable. They establish that this specific strategy is profitable under the tested conditions.

---

## 6. Limitations

The paper has several limitations that should be noted.

**Seven-year sample.** The backtest covers 2019 to 2026. This period includes COVID volatility, the post-COVID recovery, and the current regime. It does not include the 2008 financial crisis or the 2013 taper tantrum. The strategy may behave differently in other regimes.

**NIFTY only.** The strategy is tested on NIFTY 50 only. Whether it works on BANKNIFTY, FINNIFTY, or other indices is unknown.

**Approximated option premiums.** The strategy uses Black-Scholes with VIX as the implied-volatility input [20]. Real premiums depend on the actual option chain, which includes IV skew, term structure, and liquidity effects. The results would differ if actual premiums were used.

**Single lot size.** The strategy is tested with one lot per trade. With larger position sizes, market impact would reduce the effective edge.

**No intraday timing.** The strategy enters at the daily close and exits at the daily close. Real execution would occur intraday, which could affect entry and exit prices.

**No liquidity constraint.** The strategy assumes it can always execute at mid-market plus or minus slippage. Real liquidity constraints, especially for the long leg of the spread, could increase costs [27].

---

## 7. Conclusion

This paper has documented a systematic bull put credit spread strategy for NIFTY 50 weekly options with positive expected P&L over the 2019–2026 period. The strategy combines Bollinger Bandwidth regime detection, Madrid Trend Spotter direction, Ichimoku Cloud position, and a 20-day hold period with a 3.0× stop-loss. Adding an equity-curve stop that pauses trading after three consecutive losses improves the Sharpe ratio from 1.37 to 2.35 and reduces the maximum drawdown from −25.5% to −15.0%.

The strategy outperforms NIFTY buy-and-hold on every risk-adjusted metric. It is robust across 87% of 192 parameter combinations tested. It is consistent with the academic literature on the Volatility Risk Premium and with SEBI's observation that option sellers are the only profitable category of retail traders in Indian equity derivatives.

The paper also documents a structural vulnerability — sudden market reversals after extended trends — that the equity-curve stop mitigates but does not prevent. Future work should address this through more sophisticated regime detection.

---

## Data and Code Availability

The pipeline and empirical results are available at:

- https://github.com/dr-pr-anand25/nifty-bull-put-credit-spread

The repository includes backtest scripts, parameter sweep code, and all result CSVs.

---

## References

[1] Anand, P. R., & Nagesh, O. S. (2026a). Pitfalls of Single-Regime Evaluation in ML-Based Intraday Prediction: A Case Study on NIFTY Gamma Blasts. GitHub. https://github.com/dr-pr-anand25/nifty-gamma-blast-study

[2] Anand, P. R., & Nagesh, O. S. (2026b). Regime-Dependent Feature Importance in Intraday Financial Machine Learning. GitHub. https://github.com/dr-pr-anand25/nifty-regime-feature-importance

[3] Anand, P. R., & Nagesh, O. S. (2026c). A Microstructure Failure Analysis of Price-Derived Machine Learning Models and Retail Technical Heuristics on NIFTY 50 Intraday Options. GitHub. https://github.com/dr-pr-anand25/nifty-microstructure-failure

[4] Anand, P. R., & Nagesh, O. S. (2026d). Reproducibility Gaps in Synthetic Financial Machine Learning Experiments. GitHub. https://github.com/dr-pr-anand25/nifty-reproducibility-study

[5] Carr, P., & Wu, L. (2009). Variance risk premiums. The Review of Financial Studies, 22(3), 1311–1341.

[6] Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia. The Review of Financial Studies, 22(11), 4463–4492.

[7] Securities and Exchange Board of India. (2026). Study on Individual Investors in Equity Derivatives. SEBI Report.

[8] Fama, E. F., & French, K. R. (2004). The capital asset pricing model: Theory and evidence. Journal of Economic Perspectives, 18(3), 25–46.

[9] Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. Journal of Portfolio Management, 40(5), 94–107.

[10] Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. The Journal of Finance, 25(2), 383–417.

[11] Lakonishok, J., Lee, I., Pearson, N. D., & Poteshman, A. M. (2007). Option market activity. The Review of Financial Studies, 20(3), 813–857.

[12] Barber, B. M., Lee, Y.-T., Liu, Y.-J., & Odean, T. (2009). Just how much do individual investors lose by trading? The Review of Financial Studies, 22(2), 609–632.

[13] Varma, J. R. (2021). Indian securities market: A review. Vikalpa, 46(2), 69–83.

[14] Alexander, C., & Kaeck, A. (2008). Regime dependent determinants of credit default swap spreads. Journal of Banking & Finance, 32(6), 1008–1021.

[15] Fung, W., & Hsieh, D. A. (2001). The risk in hedge fund strategies: Theory and evidence from trend followers. The Review of Financial Studies, 14(2), 313–341.

[16] Lo, A. W. (2004). Adaptive markets hypothesis: Market efficiency from an evolutionary perspective. Journal of Portfolio Management, 30(5), 15–29.

[17] Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223–2273.

[18] Lopez de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.

[19] National Stock Exchange of India. (2026). Historical Index Data. NSE Archives. https://www.nseindia.com

[20] Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637–654.

[21] Bollinger, J. (2001). Bollinger on Bollinger Bands. McGraw-Hill.

[22] Wilder, J. W. (1978). New Concepts in Technical Trading Systems. Trend Research.

[23] Hosoda, G. (1993). Ichimoku Kinko Hyo. Nikkei Financial Daily.

[24] Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. Journal of Financial Econometrics, 12(1), 47–88.

[25] Harvey, C. R., & Liu, Y. (2015). Backtesting. The Journal of Portfolio Management, 42(1), 13–28.

[26] Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384.

[27] Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. Journal of Risk, 3, 5–39.
