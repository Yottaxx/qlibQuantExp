# State-of-the-Art (SOTA) Stock Prediction Research (2022-2025)

This document curates influential and state-of-the-art research papers in stock prediction and financial time series forecasting from top-tier AI conferences (KDD, AAAI, IJCAI) between 2022 and 2025. It serves as a reference for selecting baselines and understanding current research trends.

## 2025 (Latest Advances)

### [KDD 2025] Learning Universal Multi-level Market Irrationality Factors to Improve Stock Return Forecasting
*   **Context**: Efficient Market Hypothesis (EMH) often assumes rationality, but real markets are driven by irrational behaviors.
*   **Contribution**: Proposes a framework to explicitly model "market irrationality" at three levels: individual stock, sector, and market-wide.
*   **Key Innovation**: Integrating behavioral finance theories (irrationality factors) into deep learning models to capture anomalies that rational models miss.

### [KDD 2025] Pre-training Time Series Models with Stock Data Customization
*   **Context**: General Time-Series Foundation Models (like Chronos, TimesFM) are rising, but stock data has unique low signal-to-noise ratios.
*   **Contribution**: Investigates domain-specific pre-training strategies. Likely introduces objectives tailored to financial "stylized facts" (volatility clustering, heavy tails) rather than generic MSE minimization.

### [IJCAI 2025] Deep Learning for Event-Driven Stock Prediction
*   **Context**: News and events drive sudden price changes.
*   **Contribution**: Utilizes **Neural Tensor Networks (NTN)** for event representation learning combined with CNNs.
*   **Performance**: Reported ~6% improvement on S&P 500 prediction over previous event-driven baselines (like HAN, generic BERT-based models).

### [IJCAI 2025] COGRASP
*   **Contribution**: A graph-based framework achieving SOTA on real-world stock market datasets. Focuses on capturing complex, dynamic inter-stock dependencies that evolve over time.

---

## 2024 (Foundation & Efficient Architectures)

### [AAAI 2024] StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price Forecasting
*   **Authors**: Jinyong Fan, Yanyan Shen
*   **Type**: MLP-based (Efficient)
*   **Core Logic**: Challenges the necessity of complex Transformers/GNNs.
    *   **Indicator Mixing**: Learning channel interactions.
    *   **Time Mixing**: Multi-scale time patch exchange.
    *   **Stock Mixing**: Global market context integration.
*   **Significance**: Proves that properly designed MLPs can outperform heavy Transformers in stock forecasting while being faster and memory-efficient.

### [AAAI 2024] MASTER: Market-Guided Stock Transformer for Stock Price Forecasting
*   **Type**: Transformer-based
*   **Core Logic**: Addresses non-stationarity in stock correlations.
*   **Mechanism**: "Market-Guided" attention that aggregates intra-stock and inter-stock information dynamically. It uses market context as a query to select relevant features, adapting to changing market regimes.

### [Industry/ArXiv 2024] FinCast & Chronos-Bolt (Amazon)
*   **Trend**: Shift towards **Foundation Models**.
*   **Chronos-Bolt**: A probabilistic model treating time series values as tokens (using LLM architectures). It offers high-speed inference and zero-shot capabilities, effective for multivariate financial series with exogenous variables.

---

## 2023 (Adaptation & Meta-Learning)

### [KDD 2023] DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting
*   **Problem**: **Concept Drift** / Distribution Shift. Models trained on past data fail as market regimes change.
*   **Solution**: End-to-end incremental learning framework.
    *   **Data Adaptation**: Transformation network to stabilize input distribution.
    *   **Model Adaptation**: Meta-learning operational to update weights efficiently with new daily data.
*   **Status**: A strong baseline for handling non-stationarity.

### [IJCAI 2023] StockFormer: Learning Hybrid Trading Machines with Predictive Coding
*   **Method**: Hybrid architecture combining Transformers with **Predictive Coding**.
*   **Idea**: Predictive coding acts as a regularization term, forcing the model to learn robust latent states rather than overfitting to high-frequency noise.

### [AAAI 2023] StockEmotions
*   **Focus**: **Sentiment Analysis** / Multi-modality.
*   **Contribution**: Fine-grained investor emotion detection from text, integrated with quantitative price features.

---

## 2022 (Robust Baselines)

### [KDD 2022] TRA: Temporal Routing Adaptor
*   **Significance**: Highly cited baseline for concept drift.
*   **Method**: Dynamically routes samples to different predictors based on temporal similarity to the current inference time.

### [AAAI 2022] FactorVAE
*   **Significance**: Bridge between Deep Learning and Financial Factor Models (e.g., Fama-French).
*   **Method**: Variational Autoencoder to generate dynamic factors and posteriors, providing better interpretability and robustness than "black box" RNNs.

---

## Standard Comparison Benchmarks
When evaluating a new stock prediction model, it is standard to compare against these categories of baselines:

1.  **RNN/LSTM Variants**:
    *   `ALSTM` (Attentive LSTM) - *Standard baseline*
    *   `Adv-ALSTM` (Adversarial ALSTM)

2.  **Transformer Variants**:
    *   `Transformer` (Vanilla)
    *   `iTransformer` (Inverted Transformer for Time Series)
    *   `MASTER` (AAAI 2024) - *SOTA Transformer*

3.  **Graph Neural Networks (GNNs)**:
    *   `AGCRN` (Adaptive Graph Convolutional Recurrent Network)
    *   `HyperSR` (Hypergraph Stock Ranking)
    *   `RSR` (Relational Stock Ranking)

4.  **MLP/Efficient Models**:
    *   `StockMixer` (AAAI 2024) - *SOTA Efficiency*
    *   `TSMixer` (Google)

5.  **Concept Drift / Adaptation**:
    *   `DoubleAdapt` (KDD 2023) - *SOTA Adaptation*
    *   `TRA` (KDD 2022)
