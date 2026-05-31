# The AI Universe Coding — A Complete Guide
### *From Neural Networks to Generative AI*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)

---

<p align="center">
  <img src="ai_universe_rev2.png" alt="The AI Universe Coding" width="600"/>
</p>

<p align="center">
  <em>The AI Universe — five concentric rings spanning every major sub-field of Artificial Intelligence</em>
</p>

---

## Overview

This notebook is a **complete, AI Coding Guide** built around *The AI Universe* — a concentric-ring taxonomy of artificial intelligence. Every labeled topic in the graphic has a dedicated markdown explanation, working code demo, and visual output.

The guide is organized into **six progressive modules** that mirror the five rings of the diagram, moving from the broadest concepts (Artificial Intelligence) inward to the most specialized (Generative AI).

```
Artificial Intelligence
  └── Machine Learning
        └── Neural Networks
              └── Deep Learning
                    └── Generative AI
```

> **Runtime Recommendation:** GPU (T4 or better)
> *Runtime → Change runtime type → T4 GPU*

---

## Table of Contents

- [Prerequisites](#-prerequisites)
- [Setup](#-module-0--setup--environment)
- [Module 1 — Artificial Intelligence](#-module-1--artificial-intelligence-outer-ring)
- [Module 2 — Machine Learning](#-module-2--machine-learning)
- [Module 3 — Neural Networks](#-module-3--neural-networks)
- [Module 4 — Deep Learning](#-module-4--deep-learning)
- [Module 5 — Generative AI](#-module-5--generative-ai-innermost-ring)
- [Module 6 — Summary](#-module-6--course-summary)
- [Complete Topic Coverage](#-complete-topic-coverage-29-demos)
- [Technologies Used](#-technologies-used)
- [Next Steps](#-next-steps)

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Platform | Google Colab (free tier works; GPU recommended) |
| Google Account | Required for Drive mounting |
| Python | 3.10+ (pre-installed in Colab) |
| Prior Knowledge | Basic Python helpful but not required |

All libraries are installed automatically at runtime. No local setup needed.

---

## Module 0 — Setup & Environment

### 0.1 Mount Google Drive

**Text Cell:** Explains why Drive is mounted — all charts and model outputs are saved persistently so students can review them after the session ends.

**Code Cell:**
```python
from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/AI_Universe_Course'
```

**Output:** Confirms mount path. Creates `AI_Universe_Course/` folder in your Drive. All 29 charts auto-save here as `.png` files.

---

### 0.2 Install & Import All Libraries

**Text Cell:** Lists every library used across all six modules with a brief note on its role.

**Code Cell:** Installs and imports in one block:

| Library | Purpose |
|---------|---------|
| `tensorflow / keras` | Deep learning (MLP, CNN, LSTM, GAN, Transformer) |
| `scikit-learn` | Classical ML (SVM, Random Forest, K-Means, PCA) |
| `transformers` | HuggingFace pretrained models |
| `minisom` | Self-Organizing Maps |
| `scikit-fuzzy` | Fuzzy Logic inference systems |
| `gymnasium` | Reinforcement Learning environments |
| `seaborn / matplotlib` | All visualizations |
| `numpy / pandas` | Data manipulation |

**Output:** TensorFlow version confirmation, GPU availability status.

---

### 0.3 AI Universe  — Concentric Rings Diagram

**Text Cell:** Introduces the course roadmap — the five concentric rings and what each represents.

**Code Cell:** Renders a full dark-theme concentric diagram using `matplotlib` with all topic labels positioned radially inside each ring. No external image required — entirely code-generated.

**Output:**
- Dark-background concentric circle diagram with all 5 rings labeled and color-coded
- All topic names positioned within their respective rings
- Saved as `00_ai_universe_overview.png`

---

## Module 1 — Artificial Intelligence (Outer Ring)

**Text Cell:** Defines Artificial Intelligence as the broadest category. Introduces each sub-field with a one-line description covering: Planning & Scheduling, Knowledge Representation, NLP, Computer Vision, Expert Systems, Speech Recognition, Robotics, Fuzzy Logic, AI Ethics, Cognitive Computing, Automated Reasoning.

---

### 1.1 Planning & Scheduling — BFS and A\* Search

**Text Cell:** Explains search algorithms as the foundation of AI planning. Describes BFS (guaranteed shortest path, no heuristic) vs A\* (heuristic-guided, faster in practice). Introduces Manhattan distance as the heuristic.

**Code Cell:**
- Implements BFS from scratch using `collections.deque`
- Implements A\* from scratch using `heapq`
- Runs both on the same 7×7 grid with walls
- Compares path length and approach side-by-side

**Output:**
- Two-panel grid visualization: BFS path (blue dots) vs A\* path (green dots)
- Start marked with blue square, goal with gold star
- Printed comparison table:

```
 Algorithm  Path Length  Optimal  Uses Heuristic
       BFS           13     True           False
        A*           13     True            True
```

Saved as `01_planning_search.png`

---

### 1.2 Knowledge Representation — Rule-Based Expert System

**Text Cell:** Explains forward-chaining inference — the engine behind early AI expert systems like MYCIN. Describes how facts are added to a working memory and matched against condition-action rules.

**Code Cell:**
- Builds a medical triage expert system with 6 rules (Flu, Measles, Cardiac Event, Strep, TB, Migraine)
- Runs 4 patient symptom sets through the system
- Renders results as a styled matplotlib table

**Output:**
```
               Symptoms                        Diagnosis
   fever, cough, fatigue                     Possible Flu
chest_pain, shortness_of_breath  Possible Cardiac Event - Seek ER
      fever, rash, headache                Possible Measles
headache, nausea, light_sensitivity       Possible Migraine
```

Saved as `01_expert_system.png`

---

### 1.x Fuzzy Logic — Fuzzy Inference System (AI Risk Assessment)

**Text Cell:** Explains fuzzy sets and membership functions. Unlike Boolean logic where a value is either "high" or "low", fuzzy logic assigns partial membership to multiple categories simultaneously. Describes the three-step process: fuzzification → rule evaluation → defuzzification.

**Code Cell:**
- Installs `scikit-fuzzy`
- Defines two input variables: *Model Accuracy* and *Dataset Size* (each with low/medium/high membership functions)
- Defines one output: *Deployment Risk* (low/medium/high)
- Encodes all 9 input-combination rules explicitly (3×3 matrix, fully covered)
- Runs 5 real-world AI deployment scenarios

**Output:**

```
              Scenario  Accuracy  Data Size  Risk Score  Risk Level
   High Acc, Large Data        92         90        12.3         LOW
   High Acc, Small Data        88         20        49.7      MEDIUM
Medium Acc, Medium Data        65         55        50.0      MEDIUM
    Low Acc, Large Data        38         85        49.7      MEDIUM
    Low Acc, Small Data        25         15        87.8        HIGH
```

Three-panel membership function plot (Accuracy, Data Size, Risk). Saved as `01x_fuzzy_logic.png`

---

## Module 2 — Machine Learning

**Text Cell:** Defines Machine Learning and its four paradigms — Supervised, Unsupervised, Semi-Supervised, and Reinforcement — with goal and canonical examples for each.

---

### Feature Engineering — Importance, Selection & Transformation

**Text Cell:** Explains why raw features are often insufficient. Covers feature importance ranking, correlation analysis, and SelectKBest filtering. Uses the Breast Cancer dataset (30 features, binary classification).

**Code Cell:**
- Trains a `RandomForestClassifier` and extracts `feature_importances_`
- Builds a correlation matrix on the top 10 features
- Runs `SelectKBest` with `f_classif` for k = 1..30 and measures accuracy at each level
- Identifies optimal k

**Output:**
- Horizontal bar chart of top 10 feature importances
- Correlation heatmap (10×10)
- Accuracy vs feature count line chart with optimal k marked
- Printed table of top features and importance scores

Saved as `02x_feature_engineering.png`

---

### 2.1 Supervised Learning — Classification Comparison

**Text Cell:** Defines supervised learning. Explains the train/test split, cross-validation, and why comparing multiple classifiers on the same dataset reveals their relative strengths.

**Code Cell:**
- Loads the Iris dataset (150 samples, 4 features, 3 classes)
- Trains 6 classifiers: Decision Tree, SVM (RBF), Random Forest, Gradient Boosting, KNN, Logistic Regression
- Reports train accuracy, test accuracy, 5-fold CV mean and std for each
- Plots a horizontal bar chart of test accuracies
- Renders the Decision Tree structure (depth=3) using `plot_tree`

**Output:**
```
           Model  Train Acc  Test Acc  CV Mean   CV Std
   Decision Tree      1.000     0.978    0.947   ±0.027
        SVM (RBF)     0.990     0.978    0.960   ±0.027
   Random Forest      1.000     0.978    0.960   ±0.033
Gradient Boosting      1.000     0.978    0.953   ±0.033
             KNN      0.990     0.978    0.953   ±0.040
   Logistic Reg       0.983     0.978    0.947   ±0.040
```

Saved as `02_ml_classifiers.png`

---

### 2.x Semi-Supervised Learning — Label Spreading on Digits

**Text Cell:** Explains the semi-supervised paradigm — a small set of labeled examples plus a large pool of unlabeled data. Describes graph-based Label Spreading: a similarity graph is built over all samples; labels propagate from labeled nodes to their neighbors.

**Code Cell:**
- Loads the Digits dataset (1,797 samples, 64 features, 10 classes)
- Tests 5 labeling fractions: 5%, 10%, 20%, 50%, 100%
- At each fraction: trains `LabelSpreading` (SSL) vs a supervised `RandomForestClassifier` using only the labeled subset
- Reports test accuracy for both and the SSL gain

**Output:**
```
 Labeled %  N Labeled  SSL Acc  Supervised Acc  SSL Gain
        5%         63    0.872           0.754    +0.118
       10%        126    0.906           0.839    +0.067
       20%        252    0.932           0.906    +0.026
       50%        630    0.955           0.955    +0.000
      100%       1257    0.960           0.960    +0.000
```

Grouped bar chart (SSL vs Supervised at each label fraction). Saved as `02x_semi_supervised.png`

---

### 2.2 Regression — Linear & Polynomial

**Text Cell:** Distinguishes regression (continuous output) from classification (discrete). Explains underfitting (degree 1 — high bias) vs overfitting (degree 9 — high variance) using polynomial regression as a visual demonstration.

**Code Cell:**
- Generates a noisy sine wave (80 samples)
- Fits polynomial regression pipelines at degrees 1, 3, and 9
- Plots fit curve and MSE for each degree

**Output:**
Three-panel chart showing the progression from underfit (straight line) to good fit (smooth curve) to overfit (wild oscillations). MSE reported per panel. Saved as `02_regression.png`

---

### 2.3 Unsupervised Learning — K-Means Clustering & PCA

**Text Cell:** Defines unsupervised learning — finding structure without labels. Explains K-Means (centroid-based partitioning) and PCA (variance-maximizing linear projection). Connects the two: PCA for visualization, K-Means for assignment.

**Code Cell:**
- Applies the Elbow Method (k = 1..10) to identify optimal k
- Runs K-Means (k=3) on Iris
- Projects to 2D with PCA for visualization
- Shows K-Means clusters alongside true labels for comparison

**Output:**
- Elbow curve with k=3 marked
- PCA scatter: K-Means colored clusters with centroid stars
- PCA scatter: True label colors for comparison
- Explained variance printout: `PC1=72.8%  PC2=23.0%  Total=95.8%`

Saved as `02_kmeans_pca.png`

---

### 2.4 Dimensionality Reduction — PCA vs t-SNE

**Text Cell:** Explains the curse of dimensionality and why high-dimensional data must be reduced for visualization. Compares PCA (linear, global structure) vs t-SNE (non-linear, local cluster structure). Shows how to choose the right number of components using cumulative explained variance.

**Code Cell:**
- Loads the Digits dataset (64 features)
- Applies PCA (2D) and t-SNE (2D, 500 iterations) 
- Plots both scatter plots colored by digit class
- Plots the cumulative explained variance curve for all 64 components
- Reports how many components are needed for 95% variance

**Output:**
- PCA scatter (all 1,797 samples, 10 colors)
- t-SNE scatter (500 samples, visibly tighter clusters)
- Cumulative variance curve: `Components to explain 95% variance: 29 (out of 64)`

Saved as `02_pca_tsne.png`

---

### 2.5 Ensemble Learning — Bagging, Boosting & Voting

**Text Cell:** Explains why combining weak learners outperforms any single model. Distinguishes Bagging (parallel, variance reduction — Random Forest), Boosting (sequential, bias reduction — AdaBoost, Gradient Boosting), and Voting (combine different model types).

**Code Cell:**
- Generates a synthetic classification problem (1,000 samples, 20 features, 10 informative)
- Trains 5 models: Single Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Voting Ensemble
- Compares test accuracy for all five

**Output:**
```
                  Model  Test Accuracy
    Single Decision Tree          0.813
Random Forest (Bagging)           0.910
   AdaBoost (Boosting)            0.907
      Gradient Boosting           0.917
        Voting Ensemble           0.910
```

Horizontal bar chart. Saved as `02_ensemble.png`

---

## Module 3 — Neural Networks

**Text Cell:** Defines neural networks and their biological inspiration. Introduces the full topic map for this module: Perceptrons → MLP → Backpropagation → Activation Functions → CNN → RNN/LSTM → SOM.

---

### 3.1 Activation Functions — Visualization & Comparison

**Text Cell:** Explains why activation functions are necessary (without them, any depth of linear layers collapses to a single linear transformation). Introduces the vanishing gradient problem and how ReLU family activations address it.

**Code Cell:**
- Plots 6 activation functions side-by-side: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softplus
- Overlays the derivative (f'(x)) on each plot using `np.gradient`
- Prints a properties comparison table

**Output:**
Six-panel figure with activation curves and derivatives. Properties table:

```
  Function     Range  Vanishing Grad Risk     Typical Use  Computationally
   Sigmoid     (0,1)                 High  Output (binary)        Moderate
      Tanh   (-1,1)               Medium          Hidden        Moderate
      ReLU    [0,∞)    None (positive)          Hidden            Fast
Leaky ReLU  (-∞,∞)                  Low          Hidden            Fast
       ELU  (-1,∞)                  Low          Hidden        Moderate
```

Saved as `03_activation_functions.png`

---

### 3.2 Perceptron — From Scratch (AND / OR Gate)

**Text Cell:** Introduces the perceptron as the atomic unit of all neural networks. Explains the update rule: `w ← w + η(y − ŷ)x`. Distinguishes linearly separable (AND, OR) from non-separable problems (XOR — motivating multi-layer networks).

**Code Cell:**
- Implements a Perceptron class from scratch (no sklearn, no keras)
- Trains on AND gate (separable) and OR gate (separable)
- Plots the learned decision boundary over the training points
- Annotates each point with true and predicted label

**Output:**
Two-panel plot. AND gate shows a clean diagonal decision boundary. OR gate similar. Both titled `✅ Learned`. Printed predictions confirm 100% accuracy on both.

Saved as `03_perceptron.png`

---

### 3.3 Multi-Layer Perceptron (MLP) — MNIST

**Text Cell:** Explains how stacking layers with non-linear activations enables a network to learn any function (Universal Approximation Theorem). Introduces Dropout as a regularization technique. Uses MNIST as the standard benchmark.

**Code Cell:**
- Loads MNIST (60,000 train / 10,000 test)
- Builds a 3-hidden-layer MLP: 784→256→128→64→10
- Uses Dropout(0.3) and Dropout(0.2) between layers
- Trains with Adam, EarlyStopping on val_accuracy
- Evaluates on test set, generates confusion matrix

**Architecture:**
```
Input (784) → Dense(256, ReLU) → Dropout(0.3)
           → Dense(128, ReLU) → Dropout(0.2)
           → Dense(64,  ReLU)
           → Dense(10,  Softmax)
```

**Output:**
- Training/validation accuracy and loss curves
- 10×10 confusion matrix heatmap
- `✅ MLP Test Accuracy: 98.xx%`

Saved as `03_mlp_mnist.png`, `03_mlp_confusion.png`

---

### 3.x Backpropagation — Manual Forward & Backward Pass

**Text Cell:** Explains the chain rule of calculus applied to neural networks. Walks through the four steps: forward pass → compute loss → backward pass (gradient computation layer by layer) → weight update. Emphasizes that this is what `model.fit()` does internally.

**Code Cell:**
- Implements a 2-layer network entirely from scratch (NumPy only)
- Solves the XOR problem: 2 inputs → 4 hidden → 1 output
- Manually computes `dL/dW2`, `dL/db2`, `dL/dW1`, `dL/db1` using chain rule
- Logs loss and gradient norms every epoch for 5,000 iterations

**Output:**
- Log-scale loss curve (converges from ~0.25 to ~0.001)
- Gradient norm curves per layer (shows gradient flow health)
- Final W1 weight heatmap
- Printed prediction table:

```
 Input  True  Pred  Raw Output
 [0,0]     0     0      0.0023
 [0,1]     1     1      0.9971
 [1,0]     1     1      0.9968
 [1,1]     0     0      0.0031
```

Saved as `03x_backpropagation.png`

---

### 3.4 Convolutional Neural Network (CNN) — MNIST

**Text Cell:** Explains how CNNs address the key weakness of MLPs on images — by sharing weights spatially (convolution), they learn translation-invariant features. Introduces: convolutional filters, max-pooling, batch normalization, and the receptive field concept.

**Code Cell:**
- Builds a 3-conv-layer CNN: Conv(32)→BN→Pool → Conv(64)→BN→Pool → Conv(64) → Dense(128) → Dense(10)
- Trains on full MNIST with EarlyStopping
- Visualizes all 32 learned filters from the first convolutional layer

**Architecture:**
```
Input (28,28,1)
  → Conv2D(32, 3×3) → BatchNorm → MaxPool(2×2)
  → Conv2D(64, 3×3) → BatchNorm → MaxPool(2×2)
  → Conv2D(64, 3×3)
  → Flatten → Dense(128, ReLU) → Dropout(0.4)
  → Dense(10, Softmax)
```

**Output:**
- Filter visualization grid (4×8 of 32 filters, each 3×3)
- `✅ CNN Test Accuracy: 99.xx%`
- Printed MLP vs CNN comparison

Saved as `03_cnn_filters.png`

---

### 3.5 LSTM & RNN — Time Series Prediction

**Text Cell:** Explains why feedforward networks fail on sequential data (no memory of previous inputs). Introduces the RNN hidden state, and the LSTM gating mechanism (input gate, forget gate, output gate) that solves the vanishing gradient problem for long sequences. Adds GRU as a simpler alternative.

**Code Cell:**
- Generates a noisy sine wave (1,000 points)
- Creates overlapping sequences of length 50 as training windows
- Trains three architectures identically: SimpleRNN, LSTM, GRU
- Predicts on held-out test portion and measures MSE

**Output:**
Three-panel forecast plot with true signal (blue) vs prediction (red dashed):
```
SimpleRNN  |  MSE = 0.00842
LSTM       |  MSE = 0.00312
GRU        |  MSE = 0.00298
```

Saved as `03_lstm_rnn.png`

---

### 3.6 Self-Organizing Maps (SOM)

**Text Cell:** Introduces SOMs as an unsupervised neural network that learns a 2D topology-preserving map of high-dimensional data. Explains competitive learning: neurons compete to respond to each input; the winner and its neighbors update toward that input.

**Code Cell:**
- Uses `minisom` library on Iris dataset (4 features)
- Trains a 10×10 SOM for 5,000 iterations
- Renders the U-Matrix (unified distance matrix) — dark areas indicate cluster boundaries
- Renders the activation frequency map

**Output:**
- U-Matrix heatmap with Iris class markers overlaid (showing natural cluster separation)
- Activation frequency map showing which neurons respond most
- Three species visually separated in the SOM topology

Saved as `03_som.png`

---

## Module 4 — Deep Learning

**Text Cell:** Distinguishes Deep Learning from shallow Neural Networks — depth enables automatic hierarchical feature extraction: edges → shapes → objects → concepts. Introduces the eight sub-topics covered: DNNs, Transfer Learning, GANs, Attention, Dropout, Deep RL, Capsule Networks, DBNs.

---

### 4.1 Deep Neural Network Architecture — CIFAR-10

**Text Cell:** Uses CIFAR-10 (60,000 color images, 10 classes) to demonstrate why depth matters for complex data. Motivates the CNN (next section) by first showing the ceiling of pure dense networks on image data.

**Code Cell:**
- Loads CIFAR-10, shows sample images for all 10 classes
- Flattens images to 3,072-dimensional vectors
- Builds and trains DNNs at depths 1, 2, and 4
- Compares parameter counts and test accuracy

**Output:**
Sample image grid (2 rows × 10 classes):
```
Depth  Params  Test Acc
    1  395146    42.14%
    2  412170    42.62%
    4  446218    43.49%
```
> 💡 Demonstrates why CNNs are necessary — adding dense layers barely helps when spatial structure is discarded.

Saved as `04_cifar10_samples.png`

---

### 4.2 Transfer Learning — MobileNetV2 on CIFAR-10

**Text Cell:** Explains that large models pretrained on ImageNet (1.2M images, 1,000 classes) have learned powerful general features that transfer to new tasks. Distinguishes feature extraction (frozen base) from fine-tuning (unfrozen base). Shows MobileNetV2 as an efficient architecture for transfer.

**Code Cell:**
- Loads MobileNetV2 pretrained on ImageNet, freezes base layers
- Adds a classification head: GlobalAveragePooling → Dense(256) → Dropout(0.4) → Dense(10)
- Trains on only 5,000 CIFAR-10 images (images resized to 96×96)
- Evaluates on full 1,000-sample test set

**Output:**
- Accuracy and loss training curves
- `✅ Transfer Learning Test Accuracy: xx.xx% (trained on only 5000 samples!)`

Saved as `04_transfer_learning.png`

---

### 4.3 Generative Adversarial Network (GAN) — MNIST

**Text Cell:** Explains the GAN framework: a Generator that creates fake samples and a Discriminator that distinguishes real from fake. Training is a minimax game — the Generator improves until the Discriminator can no longer tell the difference. Introduces label smoothing as a stabilization technique.

**Code Cell:**
- Builds Generator: Latent(100) → Dense(256)→BN → Dense(512)→BN → Dense(1024)→BN → Dense(784, tanh)
- Builds Discriminator: Dense(512)→LeakyReLU → Dropout(0.4) → Dense(256)→LeakyReLU → Dense(1, sigmoid)
- Trains adversarially for 30 epochs (batch size 128)
- Generates a 5×5 grid of synthetic MNIST digits

**Output:**
- 5×5 grid of generated digit images (black background)
- D/G loss curve showing adversarial training dynamics

Saved as `04_gan_generated.png`, `04_gan_loss.png`

---

### 4.4 Attention Mechanism — Scaled Dot-Product Attention

**Text Cell:** Introduces attention as the ability to selectively focus on relevant parts of the input. Explains the Query/Key/Value formulation: Q and K compute compatibility scores; those scores weight the V vectors. This is the core operation inside every Transformer.

**Code Cell:**
- Implements scaled dot-product attention from scratch in NumPy:  
  `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`
- Creates synthetic 8-token sequences with 16-dimensional embeddings
- Visualizes the full 8×8 attention weight matrix

**Output:**
- Attention weight heatmap (8×8, annotated with 2-decimal scores)
- Attention output matrix visualization
- Printed shape diagnostics:
```
Query shape:   (8, 16)
Key shape:     (8, 16)
Value shape:   (8, 16)
Output shape:  (8, 16)
```

Saved as `04_attention.png`

---

### 4.5 Dropout — Regularization Effect

**Text Cell:** Explains overfitting as the gap between training and test performance. Describes Dropout: during training, each neuron is zeroed with probability p, forcing the network to learn redundant representations. This acts as an implicit ensemble of exponentially many sub-networks.

**Code Cell:**
- Trains identical MLP on MNIST with Dropout=0.0, 0.3, and 0.5
- Tracks train accuracy, val accuracy, and computes overfit gap
- Compares test accuracy and gap across three settings

**Output:**
- Training/validation curves for all three settings on same axes
- Grouped bar chart: test accuracy vs overfit gap
- Printed table:
```
       Model  Test Acc  Overfit Gap
No Dropout     0.979       0.0213
Dropout=0.3    0.978       0.0089
Dropout=0.5    0.977       0.0041
```

Saved as `04_dropout.png`

---

### 4.6 Deep Reinforcement Learning — Q-Learning on GridWorld

**Text Cell:** Explains the RL framework: an Agent observes a State, takes an Action, receives a Reward, and transitions to a new State. Introduces the Q-value (expected cumulative reward) and the Bellman equation update rule. Describes ε-greedy exploration decay.

**Code Cell:**
- Implements a 5×5 GridWorld from scratch with traps (−5), steps (−0.1), and goal (+10)
- Trains tabular Q-Learning for 2,000 episodes with ε decay from 1.0 → 0.05
- Plots smoothed reward curve and learned policy grid

**Output:**
- Smoothed reward curve (shows learning plateau around episode 500)
- Policy grid: arrows show best action per cell, 💣 marks traps, 🏆 marks goal
- `Final avg reward (last 100 episodes): 7.xx`

Saved as `04_qlearning.png`

---

### 4.x Capsule Networks — Routing by Agreement on MNIST

**Text Cell:** Explains the limitation of CNNs that max-pooling discards spatial relationships between features. Introduces Capsules as vector-valued neurons that encode both presence and pose. Describes the dynamic routing-by-agreement algorithm as a replacement for max-pooling.

**Code Cell:**
- Implements `CapsuleLayer` and `squash` activation from scratch in Keras/TF
- Builds a SimpleCapsNet: Conv(64)→Conv(128)→Primary Caps→Digit Caps(10×16D)
- Trains on 10,000 MNIST samples for 8 epochs
- Evaluates on full test set

**Architecture:**
```
Input (28,28,1)
  → Conv2D(64, 9×9) → Conv2D(128, 9×9, stride=2)
  → Primary Capsules (8D vectors, routing=3)
  → Digit Capsules (10 × 16D)
  → Capsule Length → Output (10,)
```

**Output:**
- Accuracy and loss curves
- `✅ CapsNet Test Accuracy: xx.xx%`

Saved as `04x_capsnet.png`

---

### 4.x Deep Belief Networks (DBNs) — Stacked RBMs

**Text Cell:** Explains Restricted Boltzmann Machines (RBMs) as energy-based models that learn a probability distribution over inputs. Describes how stacking RBMs layer-by-layer creates a Deep Belief Network — one of the original deep learning architectures (Hinton, 2006) that sparked the modern deep learning renaissance.

**Code Cell:**
- Uses `sklearn.neural_network.BernoulliRBM`
- Builds a two-layer DBN pipeline: RBM(256) → RBM(128) → LogisticRegression
- Trains on 10,000 MNIST samples
- Visualizes 64 of the 256 learned RBM filters from Layer 1
- Compares against raw-pixel logistic regression baseline

**Output:**
- 4×16 grid of 64 learned RBM filters (Gabor-like edge detectors)
- Comparison table:
```
           Model  Test Accuracy          Feature Source
Raw Pixels + LR          88.xx%        Raw 784 pixels
DBN (RBM×2 + LR)         92.xx%  Unsupervised RBM features
```

Saved as `04x_dbn_filters.png`

---

## Module 5 — Generative AI (Innermost Ring)

**Text Cell:** Introduces Generative AI as the current frontier — models that can produce new text, images, code, and audio indistinguishable from human-created artifacts. Lists the seven topics: Language Modeling, Transformer Architecture, Self-Attention, NLU, Text Generation, Summarization, Dialogue Systems.

---

### 5.1 Language Modeling — N-Gram Model

**Text Cell:** Defines a language model as a probability distribution over token sequences: P(wₙ | w₁...wₙ₋₁). Explains N-gram models as the simplest form — using the previous N−1 tokens as context. Discusses the Markov assumption and sparsity problems at higher N.

**Code Cell:**
- Downloads *Alice in Wonderland* (Project Gutenberg) via `wget`
- Tokenizes ~5,800 words using regex
- Trains trigram (n=3) and 4-gram (n=4) models
- Implements probability-weighted generation with float renormalization
- Plots top-20 token frequencies and next-word probability distribution

**Output:**
```
3-gram: alice was beginning to get very tired of sitting by her sister on
        the bank and of having nothing to do once or twice she had ...

4-gram: alice was beginning to get very tired of sitting by her sister on
        the bank and of having nothing to do once or twice she had ...
```

Token frequency bar chart + next-word probability bar chart. Saved as `05_ngram_lm.png`

---

### 5.2 Transformer Architecture — Built from Scratch

**Text Cell:** Describes the Transformer (Vaswani et al., 2017) as the architecture that replaced RNNs for sequence modeling. Explains multi-head self-attention (parallel attention over different representation subspaces), positional encoding, layer normalization, and the encoder block structure.

**Code Cell:**
- Implements `MultiHeadSelfAttention`, `TransformerBlock`, and `TokenAndPositionEmbedding` as custom Keras layers
- Builds a Transformer encoder for binary sentiment classification on IMDb
- Trains for 8 epochs (25,000 train reviews)

**Architecture:**
```
Input (200 tokens)
  → Token + Position Embedding (vocab=10k, dim=64)
  → TransformerBlock (4 heads, ff_dim=128)
  → GlobalAveragePooling1D
  → Dense(20, ReLU) → Dropout(0.1)
  → Dense(1, Sigmoid)
```

**Output:**
- Training/validation accuracy and loss curves
- `✅ Transformer IMDb Sentiment Accuracy: 8x.xx%`

Saved as `05_transformer.png`

---

### 5.3 Pretrained Transformers — HuggingFace Pipelines

**Text Cell:** Explains that training large language models from scratch requires billions of parameters and massive compute. Transfer learning in NLP means fine-tuning pretrained models (BERT, GPT, BART) on downstream tasks. Introduces the HuggingFace `pipeline` API as the simplest interface to state-of-the-art models.

**Code Cell — Four demos in one cell:**

**1. Sentiment Analysis (NLU)**
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
```
                                              Text    Label  Score
Artificial Intelligence is transforming every... POSITIVE  0.999
This neural network converges too slowly - fr... NEGATIVE  1.000
The transformer architecture is an elegant a... POSITIVE  1.000
            Overfitting ruined my model comp... NEGATIVE  1.000
```

**2. Zero-Shot Classification**
- Model: `facebook/bart-large-mnli`
- Classifies an unseen text into arbitrary labels without any fine-tuning
```
Input: 'This paper proposes a novel quantum-resistant encryption...'
            Label  Score
    cybersecurity  0.812
quantum computing  0.073
         robotics  0.056
 machine learning  0.048
       healthcare  0.011
```

**3. Summarization**
- Model: `facebook/bart-large-cnn` (called via `AutoModelForSeq2SeqLM` directly)
- Summarizes a 68-word paragraph to ~46 words

**4. Text Generation (GPT-2)**
- Model: `gpt2`
- Generates two completions of a given prompt with temperature=0.8

---

### 5.4 Dialogue Systems — Retrieval-Based Chatbot

**Text Cell:** Distinguishes retrieval-based chatbots (match user query to a knowledge base) from generative chatbots (produce novel responses). Explains TF-IDF as a term-weighting scheme and cosine similarity as a distance metric for semantic matching.

**Code Cell:**
- Builds an 8-entry knowledge base of AI Q&A pairs
- Vectorizes questions with `TfidfVectorizer`
- Implements a `chat()` function with confidence threshold
- Runs 6 test queries and returns best-match answers with confidence scores
- Visualizes the full query × knowledge base similarity matrix

**Output:**
```
                     User Query  Confidence                        Bot Response
Can you explain what neural n...       0.412  A neural network is a computati...
    How does the transformer work       0.523  A transformer is a deep learning...
    Tell me about overfitting pr...     0.387  Overfitting occurs when a model ...
  What is the attention mechanism       0.614  An attention mechanism allows a ...
         How do we train models?        0.312  Backpropagation is the algorithm...
      What can generative AI do?        0.441  Generative AI refers to AI syste...
```

TF-IDF similarity heatmap (6 queries × 8 KB questions). Saved as `05_chatbot_similarity.png`

---

## Module 6 —  Summary

**Code Cell:** Generates the complete course summary as a color-coded table. Each row is colored by its ring. All 29 topics show as `✅ Complete`.

**Output:**
Full 29-row summary table with columns: Ring, Topic, Dataset, Key Output, Status. Color-coded by ring matching the concentric diagram colors. Printed stats:

```
✅ Completed demos : 29
📊 Total topics    : 29

Ring breakdown:
Deep Learning              9
Neural Networks            7
Machine Learning           6
Artificial Intelligence    4
Generative AI              4
```

Saved as `06_course_summary_table.png`

---

## Complete Topic Coverage (29 Demos)

| # | Ring | Topic | Dataset | Key Output |
|---|------|-------|---------|------------|
| 1 | Artificial Intelligence | A\* Planning | Synthetic Grid | Path grid chart |
| 2 | Artificial Intelligence | Expert Systems | Custom Rules | Diagnoses table |
| 3 | Artificial Intelligence | Knowledge Rep. | Custom KB | Rule-based answers |
| 4 | Artificial Intelligence | Fuzzy Logic | Custom Control System | Membership function plot |
| 5 | Machine Learning | Feature Engineering | Breast Cancer | Importance + correlation heatmap |
| 6 | Machine Learning | Decision Trees + SVM | Iris | Accuracy comparison table |
| 7 | Machine Learning | Semi-Supervised Learning | Digits (10% labeled) | SSL vs supervised bar chart |
| 8 | Machine Learning | Regression | Synthetic | Poly fit curves |
| 9 | Machine Learning | K-Means + PCA | Iris / Digits | Cluster map + elbow |
| 10 | Machine Learning | Ensemble Learning | Synthetic | Ensemble bar chart |
| 11 | Neural Networks | Activation Functions | Synthetic | 6-function comparison plot |
| 12 | Neural Networks | Perceptron | Logic Gates | Learned decision boundary |
| 13 | Neural Networks | MLP (MNIST) | MNIST | Training curves + confusion matrix |
| 14 | Neural Networks | Backpropagation | XOR Problem | Loss + gradient norm curves |
| 15 | Neural Networks | CNN (MNIST) | MNIST | Learned conv filters |
| 16 | Neural Networks | LSTM/RNN/GRU | Sine Wave | Forecast plot (3 models) |
| 17 | Neural Networks | Self-Organizing Maps | Iris | U-matrix + activation map |
| 18 | Deep Learning | DNN Depth Study | CIFAR-10 | Depth vs accuracy table |
| 19 | Deep Learning | Transfer Learning | CIFAR-10 | Fine-tune accuracy curves |
| 20 | Deep Learning | GAN | MNIST | Generated digit images |
| 21 | Deep Learning | Attention Mechanism | Synthetic Sequences | Attention weight heatmap |
| 22 | Deep Learning | Dropout | MNIST | Overfit gap comparison |
| 23 | Deep Learning | Q-Learning | GridWorld | Learned policy grid |
| 24 | Deep Learning | Capsule Networks | MNIST (10k) | Accuracy + loss curves |
| 25 | Deep Learning | Deep Belief Networks | MNIST (10k) | RBM filter visualization |
| 26 | Generative AI | N-Gram LM | Alice in Wonderland | Token frequency + generation |
| 27 | Generative AI | Transformer (IMDb) | IMDb | Transformer accuracy curves |
| 28 | Generative AI | HuggingFace Pipelines | DistilBERT / BART / GPT-2 | NLU / Summary / Generation |
| 29 | Generative AI | Dialogue System | Custom KB | TF-IDF similarity heatmap |

---

## Technologies Used

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.10+ | Core language |
| TensorFlow / Keras | 2.x | MLP, CNN, LSTM, GAN, Transformer, CapsNet |
| scikit-learn | 1.x | Classical ML, preprocessing, evaluation |
| HuggingFace Transformers | 4.x | DistilBERT, BART, GPT-2 |
| NumPy | 1.x | Numerical computation, backprop from scratch |
| Pandas | 2.x | Data tables and output formatting |
| Matplotlib / Seaborn | 3.x | All visualizations |
| MiniSom | latest | Self-Organizing Maps |
| scikit-fuzzy | latest | Fuzzy Logic inference systems |
| Google Colab | — | Runtime, GPU, Drive integration |

---

## Output Files

All charts are saved to `MyDrive/AI_Universe_Course/` automatically:

```
AI_Universe_Course/
├── 00_ai_universe_overview.png
├── 01_planning_search.png
├── 01_expert_system.png
├── 01x_fuzzy_logic.png
├── 02x_feature_engineering.png
├── 02_ml_classifiers.png
├── 02x_semi_supervised.png
├── 02_regression.png
├── 02_kmeans_pca.png
├── 02_pca_tsne.png
├── 02_ensemble.png
├── 03_activation_functions.png
├── 03_perceptron.png
├── 03_mlp_mnist.png
├── 03_mlp_confusion.png
├── 03x_backpropagation.png
├── 03_cnn_filters.png
├── 03_lstm_rnn.png
├── 03_som.png
├── 04_cifar10_samples.png
├── 04_transfer_learning.png
├── 04_gan_generated.png
├── 04_gan_loss.png
├── 04_attention.png
├── 04_dropout.png
├── 04_qlearning.png
├── 04x_capsnet.png
├── 04x_dbn_filters.png
├── 05_ngram_lm.png
├── 05_transformer.png
├── 05_chatbot_similarity.png
└── 06_course_summary_table.png
```

---

## Next Steps

After completing this notebook, the natural progression is:

| Direction | Resource |
|-----------|---------|
| **Foundational paper** | *Attention Is All You Need* — Vaswani et al. (2017) |
| **Practice competitions** | [Kaggle](https://www.kaggle.com) |
| **Pretrained model hub** | [HuggingFace](https://huggingface.co/models) |
| **Adversarial ML** | Goodfellow et al., *Explaining and Harnessing Adversarial Examples* |
| **Quantum AI** | Biamonte et al., *Quantum Machine Learning* (Nature, 2017) |
| **Federated Learning** | McMahan et al., *Communication-Efficient Learning* |
| **AI Safety & Ethics** | [AI Now Institute](https://ainowinstitute.org) |

---

*Built for CSC-3201 Introduction to Computer Security / AI — California Polytechnic State University, San Luis Obispo*
