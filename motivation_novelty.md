# Non-Stationarity as a Structural Problem: A New Paradigm for Stock Prediction

## 1. The Challenge: The "Static Architecture" Fallacy in Dynamic Systems

The fundamental tension in financial time-series forecasting lies between **Deep Learning's hunger for stationarity** and the **Market's inherent non-stationarity**.

Standard Deep Learning thrives on I.I.D. data, assuming that the underlying data generating process (the "physics" of the world) is constant. Financial markets, however, are **regime-switching systems**. The rules that govern price movements in a low-volatility "Bull Market" (where fundamentals and subtle alpha factors dominate) are fundamentally distinct from those in a "Liquidity Crisis" (where panic, momentum, and mean-reversion dominate).

**Current limitation:** Most state-of-the-art models (Transformers, GNNs, MLPs) treat this problem as *noise* or *covariate shift*. They build a single, monolithic architecture—a "Jack of All Trades"—expecting one set of weights to average out these conflicting realities.

*   *Paradox:* A model trained to be optimal on average is optimal in no specific regime. It is "blurred," dampening its conviction during critical regime shifts (e.g., 2020 crash, 2022 inflation).

---

## 2. Evolution of Solutions: A Historical Perspective

We trace the lineage of stock prediction through three generations of handling market dynamics, specifically analyzing why recent SOTA methods still fall short.

### Generation 1: The Static Specialist (RNNs/LSTMs, e.g., ALSTM)
*   *Philosophy:* "Learn the average historical rule."
*   *Mechanism:* Train a fixed model on 10 years of data.
*   *Failure Mode:* Catastrophic failure during distribution shifts (Concept Drift). The model learns the "average" market of the last decade, which may not exist today.

### Generation 2: Input & Parameter Adaptation (SOTA: DoubleAdapt, MASTER)
This generation acknowledges non-stationarity but treats it as a "calibration" problem.

*   **Type A: Weight Adaptation (e.g., DoubleAdapt, KDD'23)**
    *   *Mechanism:* Uses meta-learning to update model weights online based on recent data.
    *   *Critique:* **Lag & Cost.** It requires constant retraining/gradient steps, making it computationally heavy. More critically, it is *reactive*—it shifts the weights *after* the regime has already changed (learning from the error), rather than *proactively* switching logic based on the environment.
    *   *Analogy:* Tuning a race car's engine slightly while driving, rather than switching from a race car to an off-road truck when the terrain changes.

*   **Type B: Feature Gating (e.g., MASTER, AAAI'24)**
    *   *Mechanism:* Uses a "Market-Guided" attention mechanism to re-weight input features (gating volume vs. price) based on a market embedding.
    *   *Critique:* **Soft vs. Hard Change.** It changes *what* the model sees (attention weights), but not *how* it thinks (inductive bias). It still forces a single architectural pathway (e.g., a unified Transformer) to process both crisis and stability. It assumes that if we just "focus" on the right pixels, the same reasoning engine works. We argue this is false: the *physics* of a crash are different from the physics of a boom.

### Generation 3 (Our Proposal): Structural Fluidity via Mixture-of-Experts
*   *Philosophy:* "Change the reasoning engine, not just the inputs or parameters."
*   *Insight:* Different regimes require fundamentally different **Inductive Biases**.
    *   **Stable Markets** require **Cross-Sectional Reasoning** (Ranking, Factor Models, "Who is the best stock?").
    *   **Volatile Markets** require **Temporal Reasoning** (Trend following, Mean Reversion, "Where is the market going?").
*   *Proposal:* We do not force one network to do both. We build **Specialist Experts** (a Time-Expert and a Factor-Expert) and build a **Router** that effectively swaps the model's brain on the fly based on the macro state.

---

## 3. Our Method: RST-MoE (Regime-Separated Temporal Mixture-of-Experts)

We propose a shift from **Parameter Adaptation** to **Structural Fluidity**.

### Key Concept: The "Physical-Switch" Hypothesis
We hypothesize that the optimal trading function $f(x)$ is essentially disjoint across regimes.
$$ f(x) \approx \alpha \cdot f_{time}(x) + \beta \cdot f_{factor}(x) $$
where $\alpha, \beta$ are not learned constants, but dynamic variables controlled by a **Macro-State Router**.

### Inductive Bias Separation
Instead of a black-box MLP, we explicitly engineer experts with distinct views:
1.  **The "Time" Expert**: Architecturally biased towards sequential patterns (e.g., Causal convolutions / LSTM cells). It asks: *What happened yesterday to this specific asset?*
2.  **The "Factor" Expert**: Architecturally biased towards cross-sectional attention. It asks: *How does this asset compare to its peers today?*

This disentanglement allows the model to be **pure** in its reasoning. In a crash, the router can silence the "Factor Expert" (since P/E ratios don't matter in a panic) and amplify the "Time Expert" (since volatility clustering is dominant), without the gradients of one interfering with the other.

---

## 4. Novelty Summary

| Dimension | Previous Loop (e.g., MASTER, DoubleAdapt) | Our Approach (RST-MoE) |
| :--- | :--- | :--- |
| **Adaptation Type** | **Soft / Parametric** (Weight updates or Attention gating) | **Hard / Structural** (Routing to different architectural blocks) |
| **Response Speed** | **Reactive** (DoubleAdapt relies on gradient updates from past errors) | **Instantaneous** (Router switches immediately based on Macro State) |
| **Regime Awareness** | **Latent / Implicit** (Learned hidden vector in MASTER) | **Explicit / Interpretable** (Macro-State Encoder with physical priors) |
| **Handling Shifts** | "Retrain quickly" or "Attend differently" | **"Switch Logic"** (Dynamic inductive bias) |
| **Core Philosophy** | One model fits all regimes. | A mixture of specialists, orchestrated by the macro environment. |

We present **RST-MoE** not merely as an accuracy improvement, but as a more principled way to model the **schizophrenic nature** of financial markets—unifying the "Time" and "Space" views of finance into a single, fluid architecture.
