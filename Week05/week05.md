# Week 05 — Simple Neural Networks and Neural Language Models

## Overview

This lab bridges classical n-gram language models with modern neural approaches. You will implement a feedforward neural network from scratch (forward pass, activation functions, backpropagation), then build a fixed-context neural language model (Bengio et al., 2003 style) that learns word embeddings jointly with the prediction task. Finally, you will compare the neural LM against the n-gram baselines from Week 04.

## Learning Goals

- Implement a **feedforward neural network** with one hidden layer from scratch using NumPy.
- Understand and apply common **activation functions** (sigmoid, tanh, ReLU) and their derivatives.
- Derive and implement **backpropagation** for a two-layer network.
- Build a **fixed-window neural language model** that maps context word embeddings to next-word probabilities.
- Learn **word embeddings** as part of the language model training.
- Evaluate neural LM **perplexity** and compare with n-gram baselines.
- Reflect on the trade-offs between count-based and neural language models.

## Setup

```bash
pip install numpy nltk matplotlib
```

---

## Part 1 — Activation Functions and Their Derivatives

Activation functions introduce non-linearity into neural networks. The choice of activation affects training dynamics, gradient flow, and expressive power.

### Key Formulas

| Name    | Formula                              | Derivative                        |
|---------|--------------------------------------|-----------------------------------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$    | $\sigma(x)(1-\sigma(x))$         |
| Tanh    | $\tanh(x)$                           | $1 - \tanh^2(x)$                 |
| ReLU    | $\max(0, x)$                         | $\mathbf{1}[x > 0]$              |
| Softmax | $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | — (used at output layer) |

### Coding Task A — Implement Activations

Implement the following functions (element-wise on NumPy arrays):
- `sigmoid(z)` and `sigmoid_grad(z)`
- `tanh_act(z)` and `tanh_grad(z)`
- `relu(z)` and `relu_grad(z)`
- `softmax(z)` — stable version using `z - max(z)` before exponentiating

### Coding Task B — Visualize Activations

Plot all three activations and their derivatives over $x \in [-4, 4]$.

### Discussion Questions

1. For which range of inputs does the sigmoid gradient vanish? Why is this a problem for deep networks?
2. Why is ReLU preferred over sigmoid/tanh in most modern deep networks?
3. Why must we use softmax (not sigmoid) at the output of a multi-class classifier?

---

## Part 2 — Forward Pass in a Two-Layer Network

A feedforward network with one hidden layer computes:

$$
h = f(W_1 x + b_1), \quad \hat{y} = \text{softmax}(W_2 h + b_2)
$$

where $x \in \mathbb{R}^d$, $W_1 \in \mathbb{R}^{H \times d}$, $b_1 \in \mathbb{R}^H$, $W_2 \in \mathbb{R}^{C \times H}$, $b_2 \in \mathbb{R}^C$, and $f$ is the hidden activation.

### Coding Task A — Network Initialization

Implement `init_params(input_dim, hidden_dim, output_dim, seed=42)` that returns a dictionary with keys `W1`, `b1`, `W2`, `b2` initialized with:
- Weights: small random values from $\mathcal{N}(0, 0.01)$
- Biases: zeros

### Coding Task B — Forward Pass

Implement `forward(x, params, activation)` that:
1. Computes the pre-activation $z_1 = W_1 x + b_1$
2. Applies the hidden activation: $h = f(z_1)$
3. Computes the output pre-activation $z_2 = W_2 h + b_2$
4. Applies softmax to get $\hat{y}$
5. Returns a cache dictionary with all intermediate values needed for backprop

### Coding Task C — Cross-Entropy Loss

Implement `cross_entropy_loss(y_hat, y_true)` where `y_true` is a one-hot vector:

$$
\mathcal{L} = -\sum_i y_i \log(\hat{y}_i) = -\log \hat{y}_{y^*}
$$

Use numerical stability clipping: `np.clip(y_hat, 1e-12, 1.0)`.

### Discussion Questions

1. What does the cache returned by `forward` need to contain for backpropagation?
2. Why is numerical stability important when computing log-softmax?

---

## Part 3 — Backpropagation

Backpropagation applies the chain rule to compute gradients of the loss with respect to all parameters.

### Key Gradient Formulas

For the two-layer network:

$$
\frac{\partial \mathcal{L}}{\partial z_2} = \hat{y} - y \quad \text{(softmax + cross-entropy combined)}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_2} = \delta_2 \cdot h^\top, \quad
\frac{\partial \mathcal{L}}{\partial b_2} = \delta_2
$$

$$
\delta_1 = (W_2^\top \delta_2) \odot f'(z_1)
$$

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \delta_1 \cdot x^\top, \quad
\frac{\partial \mathcal{L}}{\partial b_1} = \delta_1
$$

### Coding Task A — Backpropagation

Implement `backward(x, y_true, cache, params, activation)` that returns a gradient dictionary with keys `dW1`, `db1`, `dW2`, `db2`.

**Hint:** Use the cached values from `forward`. The gradient of softmax + cross-entropy w.r.t. $z_2$ is $\hat{y} - y$.

### Coding Task B — Gradient Check

Implement `gradient_check(x, y_true, params, activation, eps=1e-5)`:
- For each parameter $\theta_i$, compute the numerical gradient:

$$
\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \varepsilon) - \mathcal{L}(\theta_i - \varepsilon)}{2\varepsilon}
$$

- Compare with your analytic gradient. The relative error should be $< 10^{-5}$.

### Coding Task C — SGD Update

Implement `sgd_update(params, grads, lr)` that updates parameters in-place:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

### Discussion Questions

1. What is the "vanishing gradient" problem, and which activation functions are most prone to it?
2. Why does gradient checking use a two-sided finite difference rather than a one-sided one?

---

## Part 4 — Neural Language Model (Fixed-Window)

A neural language model (Bengio et al., 2003) predicts the next word given the previous $n-1$ words. Each context word is mapped to an embedding vector; the embeddings are concatenated and fed into a feedforward network.

### Architecture

Given context $(w_{t-(n-1)}, \dots, w_{t-1})$:

$$
x = [e(w_{t-(n-1)}); \dots; e(w_{t-1})] \in \mathbb{R}^{(n-1) \cdot d}
$$

$$
h = \tanh(W_1 x + b_1), \quad P(\cdot \mid \text{context}) = \text{softmax}(W_2 h + b_2)
$$

where $e(w) \in \mathbb{R}^d$ is a learned embedding for word $w$.

### Coding Task A — Embedding Lookup

Implement `lookup_embeddings(context_ids, E)` where:
- `context_ids`: list of integer word IDs (length $n-1$)
- `E`: embedding matrix of shape `(V, d)`
- Returns: concatenated embedding vector of shape `((n-1)*d,)`

### Coding Task B — NLM Forward Pass

Implement `nlm_forward(context_ids, E, params)` that:
1. Calls `lookup_embeddings` to get $x$
2. Calls `forward(x, params, tanh_act)` to get $\hat{y}$ and cache
3. Returns `(y_hat, cache, x)` — `x` is needed for embedding gradients

### Coding Task C — NLM Backward Pass

Implement `nlm_backward(context_ids, y_true, y_hat, cache, x, E, params)` that:
1. Calls `backward` to get parameter gradients
2. Computes embedding gradients: $\frac{\partial \mathcal{L}}{\partial E[w_i]} = \frac{\partial \mathcal{L}}{\partial x}[\text{slice}_i]$ for each context word $i$
3. Returns `(param_grads, embed_grads)` where `embed_grads` is a dict `{word_id: gradient_vector}`

### Coding Task D — Build Training Data

Implement `build_ngram_examples(sentences_tokens, n, word_to_id)` that:
- Prepends $n-1$ `<s>` tokens to each sentence
- Yields `(context_ids, target_id)` tuples for each position

### Coding Task E — Training Loop

Implement `train_nlm(sentences_tokens, word_to_id, n, d, H, lr, epochs)` that:
1. Initializes `E` (embedding matrix) and network `params`
2. Runs gradient descent for `epochs` passes over training examples
3. Records and prints average cross-entropy loss per epoch
4. Returns `E`, `params`, and the loss history

**Hint:** Update embeddings with `E[word_id] -= lr * embed_grads[word_id]` (only update rows that appeared in the batch).

### Discussion Questions

1. How does the neural LM handle unseen words at test time differently from an n-gram model with add-one smoothing?
2. What does the embedding matrix $E$ represent, and how does it differ from a one-hot input representation?
3. Why is the context window size $n$ a hyperparameter in the neural LM, just as in n-gram models?

---

## Part 5 — Perplexity and Comparison with N-gram Models

### Coding Task A — NLM Perplexity

Implement `nlm_perplexity(sentences_tokens, E, params, n, word_to_id)`:
- For each context-target pair in the test sentences, compute $-\log P(\text{target} \mid \text{context})$
- Return $\exp(\text{average negative log-prob})$

### Coding Task B — Comparison Table

Train both a bigram LM (add-one smoothing from Week 04) and the neural LM on the same corpus. Report:

| Model     | Train Perplexity | Test Perplexity |
|-----------|-----------------|-----------------|
| Bigram (add-one) | ? | ? |
| Neural LM (n=2) | ? | ? |
| Neural LM (n=3) | ? | ? |

### Coding Task C — Nearest-Neighbor Words by Embedding

After training, find the 5 nearest neighbors (by cosine similarity over embedding vectors) for at least 3 query words. Compare with the co-occurrence-based nearest neighbors from Week 04.

### Discussion Questions

1. Does the neural LM achieve lower test perplexity than add-one bigram? Under what conditions might it not?
2. How do the embedding-based nearest neighbors differ from the co-occurrence-based ones? What does this tell you about what each method captures?
3. What are two limitations of the fixed-window neural LM compared to recurrent or transformer models?

---

## Part 6 — (Bonus) Mini XOR Network

Train a 2-input, 1-hidden-layer network to learn the XOR function. This is a classic exercise that shows why a linear model cannot learn XOR, but a network with a hidden layer can.

### Coding Task

- XOR data: inputs $\{(0,0),(0,1),(1,0),(1,1)\}$, targets $\{0,1,1,0\}$ (use one-hot for 2 classes).
- Use your `forward`, `backward`, `sgd_update` from Part 3.
- Train until loss $< 0.01$ (or for 5000 steps) and report the predicted class for each input.

### Discussion Question

1. Does a single linear layer (no hidden activation) converge on XOR? Why or why not?

---

## Submission

- Submit your completed notebook/script as instructed by your TA (`.ipynb` or `.py`).
- Written answers must be in your own words.
