# 📚 Data Science Master

A personal, searchable **concept dictionary** for revising data science — from math
foundations through machine learning, deep learning, LLMs, and agentic AI.

Every topic has two files:
- **`README.md`** — the detailed note (intuition → math → code → interview questions)
- **`cheatsheet.md`** — a quick-recall summary for last-minute revision

> **139 topics** across **12 sections**.

## How to use this repo

- **Search:** press <kbd>/</kbd> or <kbd>t</kbd> on GitHub to fuzzy-find any topic file, or
  use the search bar with keywords (e.g. `regularization`, `attention`).
- **Browse:** click any folder — GitHub renders its `README.md` automatically.
- **Revise:** open a topic's `cheatsheet.md` for a 30-second refresh; dive into
  `README.md` when you need depth.
- **Add a topic:** copy `_templates/topic-note.md` and `_templates/topic-cheatsheet.md`
  into a new `slug/` folder, then add a link below.

## Note format

Each detailed note follows the same sections so scanning is fast:
`Intuition → Formal definition → Math → How it works → When to use → Pitfalls → Code →
Interview questions → References`.

Legend: 🟡 stub (needs writing) · 🟢 done · ✅ examples are fully written.

---

## 🗂️ Index

### Foundations


**Mathematics**

- [Linear Algebra](01-foundations/mathematics/linear-algebra/) · [cheatsheet](01-foundations/mathematics/linear-algebra/cheatsheet.md)
- [Calculus](01-foundations/mathematics/calculus/) · [cheatsheet](01-foundations/mathematics/calculus/cheatsheet.md)
- [Probability](01-foundations/mathematics/probability/) · [cheatsheet](01-foundations/mathematics/probability/cheatsheet.md)
- [Statistics](01-foundations/mathematics/statistics/) · [cheatsheet](01-foundations/mathematics/statistics/cheatsheet.md)
- [Optimization](01-foundations/mathematics/optimization/) · [cheatsheet](01-foundations/mathematics/optimization/cheatsheet.md)
- [Information Theory](01-foundations/mathematics/information-theory/) · [cheatsheet](01-foundations/mathematics/information-theory/cheatsheet.md)

**Programming**

- [Python](01-foundations/programming/python/) · [cheatsheet](01-foundations/programming/python/cheatsheet.md)
- [SQL](01-foundations/programming/sql/) · [cheatsheet](01-foundations/programming/sql/cheatsheet.md)
- [R](01-foundations/programming/r/) · [cheatsheet](01-foundations/programming/r/cheatsheet.md)
- [Git Version Control](01-foundations/programming/git-version-control/) · [cheatsheet](01-foundations/programming/git-version-control/cheatsheet.md)
- [Environments and Packaging](01-foundations/programming/environments-and-packaging/) · [cheatsheet](01-foundations/programming/environments-and-packaging/cheatsheet.md)

**CS Fundamentals**

- [Data Structures](01-foundations/cs-fundamentals/data-structures/) · [cheatsheet](01-foundations/cs-fundamentals/data-structures/cheatsheet.md)
- [Algorithms](01-foundations/cs-fundamentals/algorithms/) · [cheatsheet](01-foundations/cs-fundamentals/algorithms/cheatsheet.md)
- [Time Space Complexity](01-foundations/cs-fundamentals/time-space-complexity/) · [cheatsheet](01-foundations/cs-fundamentals/time-space-complexity/cheatsheet.md)

### Data Engineering

- [ETL ELT](02-data-engineering/etl-elt/) · [cheatsheet](02-data-engineering/etl-elt/cheatsheet.md)
- [Data Modeling](02-data-engineering/data-modeling/) · [cheatsheet](02-data-engineering/data-modeling/cheatsheet.md)
- [Relational Databases](02-data-engineering/relational-databases/) · [cheatsheet](02-data-engineering/relational-databases/cheatsheet.md)
- [NOSQL Databases](02-data-engineering/nosql-databases/) · [cheatsheet](02-data-engineering/nosql-databases/cheatsheet.md)
- [Data Warehousing](02-data-engineering/data-warehousing/) · [cheatsheet](02-data-engineering/data-warehousing/cheatsheet.md)
- [Data Lakes Lakehouse](02-data-engineering/data-lakes-lakehouse/) · [cheatsheet](02-data-engineering/data-lakes-lakehouse/cheatsheet.md)
- [Batch Processing Spark](02-data-engineering/batch-processing-spark/) · [cheatsheet](02-data-engineering/batch-processing-spark/cheatsheet.md)
- [Stream Processing Kafka](02-data-engineering/stream-processing-kafka/) · [cheatsheet](02-data-engineering/stream-processing-kafka/cheatsheet.md)
- [Workflow Orchestration Airflow](02-data-engineering/workflow-orchestration-airflow/) · [cheatsheet](02-data-engineering/workflow-orchestration-airflow/cheatsheet.md)
- [Cloud Fundamentals](02-data-engineering/cloud-fundamentals/) · [cheatsheet](02-data-engineering/cloud-fundamentals/cheatsheet.md)
- [Data Quality](02-data-engineering/data-quality/) · [cheatsheet](02-data-engineering/data-quality/cheatsheet.md)

### Data Analysis & Statistics

- [Exploratory Data Analysis](03-data-analysis/exploratory-data-analysis/) · [cheatsheet](03-data-analysis/exploratory-data-analysis/cheatsheet.md)
- [Data Cleaning](03-data-analysis/data-cleaning/) · [cheatsheet](03-data-analysis/data-cleaning/cheatsheet.md)
- [Data Visualization](03-data-analysis/data-visualization/) · [cheatsheet](03-data-analysis/data-visualization/cheatsheet.md)
- [Descriptive Statistics](03-data-analysis/descriptive-statistics/) · [cheatsheet](03-data-analysis/descriptive-statistics/cheatsheet.md)
- [Inferential Statistics](03-data-analysis/inferential-statistics/) · [cheatsheet](03-data-analysis/inferential-statistics/cheatsheet.md)
- [Hypothesis Testing](03-data-analysis/hypothesis-testing/) · [cheatsheet](03-data-analysis/hypothesis-testing/cheatsheet.md)
- [AB Testing](03-data-analysis/ab-testing/) · [cheatsheet](03-data-analysis/ab-testing/cheatsheet.md)
- [Feature Engineering](03-data-analysis/feature-engineering/) · [cheatsheet](03-data-analysis/feature-engineering/cheatsheet.md)
- [Correlation vs Causation](03-data-analysis/correlation-vs-causation/) · [cheatsheet](03-data-analysis/correlation-vs-causation/cheatsheet.md)

### Machine Learning


**Fundamentals**

- [ML Workflow](04-machine-learning/fundamentals/ml-workflow/) · [cheatsheet](04-machine-learning/fundamentals/ml-workflow/cheatsheet.md)
- [Bias Variance Tradeoff](04-machine-learning/fundamentals/bias-variance-tradeoff/) · [cheatsheet](04-machine-learning/fundamentals/bias-variance-tradeoff/cheatsheet.md)
- [Train Test Validation Split](04-machine-learning/fundamentals/train-test-validation-split/) · [cheatsheet](04-machine-learning/fundamentals/train-test-validation-split/cheatsheet.md)
- [Cross Validation](04-machine-learning/fundamentals/cross-validation/) · [cheatsheet](04-machine-learning/fundamentals/cross-validation/cheatsheet.md)
- [Overfitting Underfitting](04-machine-learning/fundamentals/overfitting-underfitting/) · [cheatsheet](04-machine-learning/fundamentals/overfitting-underfitting/cheatsheet.md)
- [Model Evaluation Metrics](04-machine-learning/fundamentals/model-evaluation-metrics/) · [cheatsheet](04-machine-learning/fundamentals/model-evaluation-metrics/cheatsheet.md)

**Supervised**

- [Linear Regression](04-machine-learning/supervised/linear-regression/) · [cheatsheet](04-machine-learning/supervised/linear-regression/cheatsheet.md)
- [Logistic Regression](04-machine-learning/supervised/logistic-regression/) · [cheatsheet](04-machine-learning/supervised/logistic-regression/cheatsheet.md) ✅
- [Decision Trees](04-machine-learning/supervised/decision-trees/) · [cheatsheet](04-machine-learning/supervised/decision-trees/cheatsheet.md)
- [Random Forest](04-machine-learning/supervised/random-forest/) · [cheatsheet](04-machine-learning/supervised/random-forest/cheatsheet.md)
- [Gradient Boosting](04-machine-learning/supervised/gradient-boosting/) · [cheatsheet](04-machine-learning/supervised/gradient-boosting/cheatsheet.md)
- [Xgboost](04-machine-learning/supervised/xgboost/) · [cheatsheet](04-machine-learning/supervised/xgboost/cheatsheet.md)
- [SVM](04-machine-learning/supervised/svm/) · [cheatsheet](04-machine-learning/supervised/svm/cheatsheet.md)
- [KNN](04-machine-learning/supervised/knn/) · [cheatsheet](04-machine-learning/supervised/knn/cheatsheet.md)
- [Naive Bayes](04-machine-learning/supervised/naive-bayes/) · [cheatsheet](04-machine-learning/supervised/naive-bayes/cheatsheet.md)

**Unsupervised**

- [K Means](04-machine-learning/unsupervised/k-means/) · [cheatsheet](04-machine-learning/unsupervised/k-means/cheatsheet.md)
- [Hierarchical Clustering](04-machine-learning/unsupervised/hierarchical-clustering/) · [cheatsheet](04-machine-learning/unsupervised/hierarchical-clustering/cheatsheet.md)
- [Dbscan](04-machine-learning/unsupervised/dbscan/) · [cheatsheet](04-machine-learning/unsupervised/dbscan/cheatsheet.md)
- [PCA](04-machine-learning/unsupervised/pca/) · [cheatsheet](04-machine-learning/unsupervised/pca/cheatsheet.md)
- [TSNE and UMAP](04-machine-learning/unsupervised/tsne-and-umap/) · [cheatsheet](04-machine-learning/unsupervised/tsne-and-umap/cheatsheet.md)

**Advanced**

- [Ensemble Methods](04-machine-learning/advanced/ensemble-methods/) · [cheatsheet](04-machine-learning/advanced/ensemble-methods/cheatsheet.md)
- [Feature Selection](04-machine-learning/advanced/feature-selection/) · [cheatsheet](04-machine-learning/advanced/feature-selection/cheatsheet.md)
- [Hyperparameter Tuning](04-machine-learning/advanced/hyperparameter-tuning/) · [cheatsheet](04-machine-learning/advanced/hyperparameter-tuning/cheatsheet.md)
- [Imbalanced Data](04-machine-learning/advanced/imbalanced-data/) · [cheatsheet](04-machine-learning/advanced/imbalanced-data/cheatsheet.md)
- [Regularization](04-machine-learning/advanced/regularization/) · [cheatsheet](04-machine-learning/advanced/regularization/cheatsheet.md)
- [Model Interpretability](04-machine-learning/advanced/model-interpretability/) · [cheatsheet](04-machine-learning/advanced/model-interpretability/cheatsheet.md)

### Deep Learning

- [Neural Network Basics](05-deep-learning/neural-network-basics/) · [cheatsheet](05-deep-learning/neural-network-basics/cheatsheet.md)
- [Backpropagation](05-deep-learning/backpropagation/) · [cheatsheet](05-deep-learning/backpropagation/cheatsheet.md)
- [Activation Functions](05-deep-learning/activation-functions/) · [cheatsheet](05-deep-learning/activation-functions/cheatsheet.md)
- [Loss Functions](05-deep-learning/loss-functions/) · [cheatsheet](05-deep-learning/loss-functions/cheatsheet.md)
- [Optimizers](05-deep-learning/optimizers/) · [cheatsheet](05-deep-learning/optimizers/cheatsheet.md)
- [CNN](05-deep-learning/cnn/) · [cheatsheet](05-deep-learning/cnn/cheatsheet.md)
- [RNN LSTM GRU](05-deep-learning/rnn-lstm-gru/) · [cheatsheet](05-deep-learning/rnn-lstm-gru/cheatsheet.md)
- [Transformers](05-deep-learning/transformers/) · [cheatsheet](05-deep-learning/transformers/cheatsheet.md)
- [Autoencoders](05-deep-learning/autoencoders/) · [cheatsheet](05-deep-learning/autoencoders/cheatsheet.md)
- [GANS](05-deep-learning/gans/) · [cheatsheet](05-deep-learning/gans/cheatsheet.md)
- [Regularization in DL](05-deep-learning/regularization-in-dl/) · [cheatsheet](05-deep-learning/regularization-in-dl/cheatsheet.md)
- [Transfer Learning](05-deep-learning/transfer-learning/) · [cheatsheet](05-deep-learning/transfer-learning/cheatsheet.md)
- [Pytorch Basics](05-deep-learning/pytorch-basics/) · [cheatsheet](05-deep-learning/pytorch-basics/cheatsheet.md)
- [Tensorflow Basics](05-deep-learning/tensorflow-basics/) · [cheatsheet](05-deep-learning/tensorflow-basics/cheatsheet.md)

### Natural Language Processing

- [Text Preprocessing](06-nlp/text-preprocessing/) · [cheatsheet](06-nlp/text-preprocessing/cheatsheet.md)
- [Tokenization](06-nlp/tokenization/) · [cheatsheet](06-nlp/tokenization/cheatsheet.md)
- [Bag of Words and TFIDF](06-nlp/bag-of-words-and-tfidf/) · [cheatsheet](06-nlp/bag-of-words-and-tfidf/cheatsheet.md)
- [Word Embeddings](06-nlp/word-embeddings/) · [cheatsheet](06-nlp/word-embeddings/cheatsheet.md)
- [Sequence Models](06-nlp/sequence-models/) · [cheatsheet](06-nlp/sequence-models/cheatsheet.md)
- [Attention Mechanism](06-nlp/attention-mechanism/) · [cheatsheet](06-nlp/attention-mechanism/cheatsheet.md)
- [Named Entity Recognition](06-nlp/named-entity-recognition/) · [cheatsheet](06-nlp/named-entity-recognition/cheatsheet.md)
- [Sentiment Analysis](06-nlp/sentiment-analysis/) · [cheatsheet](06-nlp/sentiment-analysis/cheatsheet.md)

### LLMs & Generative AI

- [Transformer Architecture](07-llms-genai/transformer-architecture/) · [cheatsheet](07-llms-genai/transformer-architecture/cheatsheet.md)
- [Pretraining](07-llms-genai/pretraining/) · [cheatsheet](07-llms-genai/pretraining/cheatsheet.md)
- [Fine Tuning](07-llms-genai/fine-tuning/) · [cheatsheet](07-llms-genai/fine-tuning/cheatsheet.md)
- [LORA and PEFT](07-llms-genai/lora-and-peft/) · [cheatsheet](07-llms-genai/lora-and-peft/cheatsheet.md)
- [Prompt Engineering](07-llms-genai/prompt-engineering/) · [cheatsheet](07-llms-genai/prompt-engineering/cheatsheet.md)
- [RAG](07-llms-genai/rag/) · [cheatsheet](07-llms-genai/rag/cheatsheet.md) ✅
- [Vector Databases](07-llms-genai/vector-databases/) · [cheatsheet](07-llms-genai/vector-databases/cheatsheet.md)
- [Embeddings and Similarity Search](07-llms-genai/embeddings-and-similarity-search/) · [cheatsheet](07-llms-genai/embeddings-and-similarity-search/cheatsheet.md)
- [Quantization](07-llms-genai/quantization/) · [cheatsheet](07-llms-genai/quantization/cheatsheet.md)
- [LLM Evaluation](07-llms-genai/llm-evaluation/) · [cheatsheet](07-llms-genai/llm-evaluation/cheatsheet.md)
- [Hallucination and Grounding](07-llms-genai/hallucination-and-grounding/) · [cheatsheet](07-llms-genai/hallucination-and-grounding/cheatsheet.md)
- [Decoding Strategies](07-llms-genai/decoding-strategies/) · [cheatsheet](07-llms-genai/decoding-strategies/cheatsheet.md)

### Agentic AI

- [Agent Fundamentals](08-agentic-ai/agent-fundamentals/) · [cheatsheet](08-agentic-ai/agent-fundamentals/cheatsheet.md)
- [Tool Use Function Calling](08-agentic-ai/tool-use-function-calling/) · [cheatsheet](08-agentic-ai/tool-use-function-calling/cheatsheet.md)
- [REACT Pattern](08-agentic-ai/react-pattern/) · [cheatsheet](08-agentic-ai/react-pattern/cheatsheet.md)
- [Planning and Reasoning](08-agentic-ai/planning-and-reasoning/) · [cheatsheet](08-agentic-ai/planning-and-reasoning/cheatsheet.md)
- [Memory Systems](08-agentic-ai/memory-systems/) · [cheatsheet](08-agentic-ai/memory-systems/cheatsheet.md)
- [Multi Agent Systems](08-agentic-ai/multi-agent-systems/) · [cheatsheet](08-agentic-ai/multi-agent-systems/cheatsheet.md)
- [Model Context Protocol](08-agentic-ai/model-context-protocol/) · [cheatsheet](08-agentic-ai/model-context-protocol/cheatsheet.md)
- [Agent Frameworks](08-agentic-ai/agent-frameworks/) · [cheatsheet](08-agentic-ai/agent-frameworks/cheatsheet.md)
- [Agent Evaluation](08-agentic-ai/agent-evaluation/) · [cheatsheet](08-agentic-ai/agent-evaluation/cheatsheet.md)

### Specialized Domains


**Time Series**

- [Time Series Basics](09-specialized/time-series/time-series-basics/) · [cheatsheet](09-specialized/time-series/time-series-basics/cheatsheet.md)
- [ARIMA](09-specialized/time-series/arima/) · [cheatsheet](09-specialized/time-series/arima/cheatsheet.md)
- [Forecasting With ML](09-specialized/time-series/forecasting-with-ml/) · [cheatsheet](09-specialized/time-series/forecasting-with-ml/cheatsheet.md)
- [Seasonality and Trends](09-specialized/time-series/seasonality-and-trends/) · [cheatsheet](09-specialized/time-series/seasonality-and-trends/cheatsheet.md)

**Recommender Systems**

- [Collaborative Filtering](09-specialized/recommender-systems/collaborative-filtering/) · [cheatsheet](09-specialized/recommender-systems/collaborative-filtering/cheatsheet.md)
- [Content Based Filtering](09-specialized/recommender-systems/content-based-filtering/) · [cheatsheet](09-specialized/recommender-systems/content-based-filtering/cheatsheet.md)
- [Matrix Factorization](09-specialized/recommender-systems/matrix-factorization/) · [cheatsheet](09-specialized/recommender-systems/matrix-factorization/cheatsheet.md)

**Computer Vision**

- [Image Basics](09-specialized/computer-vision/image-basics/) · [cheatsheet](09-specialized/computer-vision/image-basics/cheatsheet.md)
- [Object Detection](09-specialized/computer-vision/object-detection/) · [cheatsheet](09-specialized/computer-vision/object-detection/cheatsheet.md)
- [Image Segmentation](09-specialized/computer-vision/image-segmentation/) · [cheatsheet](09-specialized/computer-vision/image-segmentation/cheatsheet.md)

**Reinforcement Learning**

- [Markov Decision Process](09-specialized/reinforcement-learning/markov-decision-process/) · [cheatsheet](09-specialized/reinforcement-learning/markov-decision-process/cheatsheet.md)
- [Q Learning](09-specialized/reinforcement-learning/q-learning/) · [cheatsheet](09-specialized/reinforcement-learning/q-learning/cheatsheet.md)
- [Policy Gradients](09-specialized/reinforcement-learning/policy-gradients/) · [cheatsheet](09-specialized/reinforcement-learning/policy-gradients/cheatsheet.md)
- [Deep Rl](09-specialized/reinforcement-learning/deep-rl/) · [cheatsheet](09-specialized/reinforcement-learning/deep-rl/cheatsheet.md)

**Graph ML**

- [Graph Basics](09-specialized/graph-ml/graph-basics/) · [cheatsheet](09-specialized/graph-ml/graph-basics/cheatsheet.md)
- [Graph Neural Networks](09-specialized/graph-ml/graph-neural-networks/) · [cheatsheet](09-specialized/graph-ml/graph-neural-networks/cheatsheet.md)

**Genomics ML**

- [Sequence Data Basics](09-specialized/genomics-ml/sequence-data-basics/) · [cheatsheet](09-specialized/genomics-ml/sequence-data-basics/cheatsheet.md)
- [Splicing and Variants](09-specialized/genomics-ml/splicing-and-variants/) · [cheatsheet](09-specialized/genomics-ml/splicing-and-variants/cheatsheet.md)
- [ML for Genomics](09-specialized/genomics-ml/ml-for-genomics/) · [cheatsheet](09-specialized/genomics-ml/ml-for-genomics/cheatsheet.md)

### MLOps

- [Model Deployment](10-mlops/model-deployment/) · [cheatsheet](10-mlops/model-deployment/cheatsheet.md)
- [Model Serving](10-mlops/model-serving/) · [cheatsheet](10-mlops/model-serving/cheatsheet.md)
- [Experiment Tracking](10-mlops/experiment-tracking/) · [cheatsheet](10-mlops/experiment-tracking/cheatsheet.md)
- [Model Monitoring](10-mlops/model-monitoring/) · [cheatsheet](10-mlops/model-monitoring/cheatsheet.md)
- [CI CD for ML](10-mlops/ci-cd-for-ml/) · [cheatsheet](10-mlops/ci-cd-for-ml/cheatsheet.md)
- [Model Registry](10-mlops/model-registry/) · [cheatsheet](10-mlops/model-registry/cheatsheet.md)
- [Feature Stores](10-mlops/feature-stores/) · [cheatsheet](10-mlops/feature-stores/cheatsheet.md)
- [Reproducibility](10-mlops/reproducibility/) · [cheatsheet](10-mlops/reproducibility/cheatsheet.md)

### Responsible AI

- [Bias and Fairness](11-responsible-ai/bias-and-fairness/) · [cheatsheet](11-responsible-ai/bias-and-fairness/cheatsheet.md)
- [Explainability](11-responsible-ai/explainability/) · [cheatsheet](11-responsible-ai/explainability/cheatsheet.md)
- [Privacy and Security](11-responsible-ai/privacy-and-security/) · [cheatsheet](11-responsible-ai/privacy-and-security/cheatsheet.md)
- [Ethics](11-responsible-ai/ethics/) · [cheatsheet](11-responsible-ai/ethics/cheatsheet.md)
- [Data Governance](11-responsible-ai/data-governance/) · [cheatsheet](11-responsible-ai/data-governance/cheatsheet.md)

### Meta (Prep & Resources)

- [Interview Prep](12-meta/interview-prep/) · [cheatsheet](12-meta/interview-prep/cheatsheet.md)
- [Viva Prep](12-meta/viva-prep/) · [cheatsheet](12-meta/viva-prep/cheatsheet.md)
- [Paper Reading Notes](12-meta/paper-reading-notes/) · [cheatsheet](12-meta/paper-reading-notes/cheatsheet.md)
- [Resources and Books](12-meta/resources-and-books/) · [cheatsheet](12-meta/resources-and-books/cheatsheet.md)

---

## Suggested revision workflow
1. Skim the cheatsheet the night before.
2. Read the full note once, then close it and re-explain the topic aloud.
3. Answer the interview/viva questions without looking.
4. Mark the note 🟢 when you can teach it from memory.

## Contributing to your own notes
- Keep math in `$...$` / `$$...$$`.
- One concept per note; link related topics instead of repeating.
- Prefer your own words over pasted text (better for recall and originality).
