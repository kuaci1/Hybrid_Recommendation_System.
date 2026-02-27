## 🚀 Project Overview

This project presents a **state-of-the-art Hybrid Recommendation System** designed to overcome the limitations of traditional recommendation approaches. It intelligently fuses three complementary information sources to predict user ratings:

1.  **Collaborative Filtering (CF)**: Captures user-item interaction patterns.
2.  **Content-Based Filtering (NLP)**: Leverages semantic understanding of review texts using BERT.
3.  **Neural Fusion Layer**: A deep neural network that learns complex non-linear interactions between these diverse signals.

Our system aims to answer: *"Given a user, an item, and their associated review, what rating would the user give?"*

## ⚙️ Architecture at a Glance

The model's architecture is a multi-modal neural network that concatenates representations from different sources:

*   **User Embeddings (64-dim)**: Learned via `nn.Embedding`.
*   **Item Embeddings (64-dim)**: Learned via `nn.Embedding`.
*   **Review Text Embeddings**: Pre-extracted BERT `[CLS]` token embeddings (768-dim), projected down to 128-dim through a `Linear -> LayerNorm -> ReLU -> Dropout` layer.

These three components are then concatenated and fed into a **Fusion MLP** (`256 -> 128 -> 64`) with `BatchNorm`, `ReLU`, and `Dropout` layers, culminating in a final `Linear` layer for rating prediction.
## ✨ Key Contributions & Features

*   **Multi-Signal Fusion**: Jointly learns from collaborative signals and semantic review content, capturing both *who likes what* and *why they like it*.
*   **Review-Aware Embeddings**: Leverages pretrained BERT to extract contextual sentence-level representations from user reviews.
*   **Scalable Architecture**: BERT embeddings are pre-extracted and cached to avoid repeated computations during training, ensuring memory efficiency.
*   **Chronological Splitting**: Employs a robust chronological train/test split to prevent temporal data leakage and simulate real-world deployment.
*   **Production-Grade Training**: Implements Adam optimizer, `ReduceLROnPlateau` scheduler, early stopping, and gradient clipping.
*   **Explainability Layer**: Provides interpretable recommendations by surfacing feature contributions and semantically similar review excerpts as natural language explanations.
*   **Synthetic Item ID Generation**: Handles datasets without explicit item IDs by clustering review topics using title hashing, enabling collaborative filtering.

## 💾 Dataset

We utilize a subset of the **Amazon Product Reviews** dataset (`/content/Amazon_Reviews.csv`). This dataset contains detailed review information, including `Profile Link` (used as `user_id`), `Review Title`, `Review Text`, `Rating`, and `Review Date`.

*   **Total Reviews (processed)**: 21,055
*   **Unique Users**: 21,055
*   **Unique Items (synthetic)**: 18,697

## 📊 Evaluation & Results

The model is evaluated on a held-out test set (15% of the data) across both **rating prediction accuracy** and **ranking quality** metrics.

**Rating Prediction Metrics:**
*   **RMSE**: 0.7295
*   **MAE**: 0.3640
*   **R² Score**: 0.6735

**Ranking Metrics (K=10, Threshold=4.0):**
*   **Precision@10**: 1.0000
*   **Recall@10**: 0.0254
*   **NDCG@10**: 0.9590

**Baseline Comparison:**
Our Hybrid Model significantly outperforms simple baselines (Global Mean, User Mean, Random) with a **+52.2% RMSE reduction** over the best baseline. This highlights the value of fusing collaborative and content-based signals.

## 💬 Explainability

Understanding *why* a recommendation is made is crucial. Our explainer module provides:

*   **Feature Contributions**: Gradient-based attribution to show which input (user, item, text) contributed most to the prediction.
*   **Nearest Reviews**: Finds the most semantically similar reviews from the training set to provide natural language explanations.
*   **Embedding Visualization**: Uses PCA/t-SNE to visualize learned user and item embedding spaces, offering insights into their clustering based on properties like average rating.

## 🚧 Limitations & Future Work

*   **Frozen BERT**: BERT weights are currently frozen. Fine-tuning the last few BERT layers could further boost performance.
*   **Synthetic Item IDs**: While effective, using more advanced topic modeling (e.g., LDA, BERTopic) instead of simple title hashing could create more nuanced item clusters.
*   **Simple Fusion**: The current fusion uses concatenation. Exploring attention mechanisms could allow for adaptive weighting of signals.
*   **Cold-Start Inference**: Developing specific strategies for new users/items at inference time (e.g., content-only fallbacks) is an area for improvement.

## 🛠️ Technology Stack

*   **Deep Learning**: PyTorch
*   **NLP**: HuggingFace Transformers
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn
*   **Evaluation**: scikit-learn
