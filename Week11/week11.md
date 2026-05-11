# Week 11 — Transformers and Modern NLP Architectures

## Overview

This lab builds the Transformer architecture from the ground up. You will implement **scaled dot-product attention** and **multi-head attention** in pure NumPy/PyTorch, assemble a **Transformer encoder block** with residual connections and layer normalisation, and use pre-trained **BERT** for contextual embeddings. You will then fine-tune BERT for text classification, visualise its attention patterns, and reflect on why the Transformer fundamentally changed what is possible in NLP.

## Learning Goals

- Understand the **limitations of sequential models** (RNNs, LSTMs) that motivated the Transformer.
- Implement **scaled dot-product attention** from scratch: queries, keys, values, and the softmax normalisation.
- Implement **multi-head attention** with learned linear projections.
- Implement **sinusoidal positional encoding** and understand why it is necessary.
- Assemble a **Transformer encoder block**: multi-head attention → add & norm → feed-forward → add & norm.
- Use **BERT** from HuggingFace to extract contextual word embeddings and compare them to static embeddings.
- Fine-tune a pre-trained transformer for **text classification** and evaluate performance.
- Visualise **attention heads** and interpret what different heads attend to.
- Articulate why the Transformer architecture enabled the **pre-train → fine-tune paradigm** and scaling.

## Setup

```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn numpy pandas
```

---

## Part 1 — The Need for Transformers

Before the Transformer (Vaswani et al., 2017), sequence modeling relied on **Recurrent Neural Networks** (RNNs) and **LSTMs**. These architectures process tokens one step at a time, left-to-right, which creates two fundamental bottlenecks: **sequential computation** that cannot be parallelised, and **fixed-size hidden states** that struggle to carry information across long distances.

### The Bottlenecks of Sequential Models

| Property | RNN / LSTM | Transformer |
|----------|-----------|-------------|
| Computation order | Sequential (t depends on t-1) | Fully parallel |
| Path between tokens i and j | O(n) steps | O(1) (direct attention) |
| Maximum context | Fixed hidden size | Full sequence (limited only by compute) |
| Training speed on modern GPUs | Slow (sequential) | Fast (matrix operations) |
| Long-range dependency | Hard (vanishing gradient) | Easy (direct attention weight) |

### Task 1A — Complexity Analysis

For a sequence of length n and model dimension d, answer the following:

| Metric | Self-Attention | Recurrent (LSTM) |
|--------|---------------|-----------------|
| Per-layer computation | O(n²·d) | O(n·d²) |
| Sequential operations | O(1) | O(n) |
| Maximum path length (i→j) | O(1) | O(n) |

Fill in which regime (n² or d²) dominates for long sentences vs wide models, and explain which architecture is preferred when n ≪ d vs n ≫ d.

### Task 1B — Long-Range Dependency Problem

Given the sentence: *"The keys that the student who had studied abroad lost were found near the library entrance."*

1. Identify the long-range grammatical dependency in the sentence.
2. Estimate how many recurrent steps an LSTM would need to propagate information from the subject to its predicate.
3. Explain how the Transformer resolves this with a single attention operation.

### Discussion Questions

1. Why does the vanishing gradient problem particularly afflict RNNs trained on long sequences? How do LSTMs partially address it?
2. What is the **information bottleneck** in an encoder-decoder RNN? How does the attention mechanism (Bahdanau, 2015) partially solve it?
3. What is **teacher forcing** and why is it needed in RNN training but not in Transformer training?

---

## Part 2 — Scaled Dot-Product Attention

Attention is the core operation of the Transformer. Given a set of **queries** (Q), **keys** (K), and **values** (V), attention computes a weighted sum of values, where the weights are determined by the compatibility of each query with each key.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The scaling factor $1/\sqrt{d_k}$ prevents dot products from growing too large in high dimensions, which would push softmax into regions of near-zero gradient.

### Coding Task A — Softmax and Scaled Attention

Implement `softmax(x)` (numerically stable) and `scaled_dot_product_attention(Q, K, V, mask=None)`. The mask (if provided) is an additive mask of −∞ for positions to ignore (used in the decoder for causal masking).

### Coding Task B — Attention Visualisation

Create a toy 5-token sequence and compute self-attention weights. Visualise the 5×5 weight matrix as a heatmap. Observe which tokens attend to which.

### Coding Task C — Effect of Temperature

Replace the fixed $\sqrt{d_k}$ divisor with a temperature parameter τ. Visualise how τ = 0.1 (sharp), τ = 1.0 (standard), and τ = 10.0 (flat) change the attention distribution. Discuss implications for model behaviour.

### Discussion Questions

1. What happens to attention weights when $d_k$ is large and the scaling factor is omitted? Why is this a training problem?
2. In **self-attention**, Q, K, and V all come from the same input. In **cross-attention** (decoder), Q comes from the decoder and K, V from the encoder. What does each mechanism achieve?
3. What is a **causal mask** and why is it essential for autoregressive (left-to-right) language models?

---

## Part 3 — Multi-Head Attention and Positional Encoding

A single attention head collapses all information into one weighted sum. **Multi-head attention** runs h parallel attention heads, each with its own learned projections W_Q^i, W_K^i, W_V^i, then concatenates and projects the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$
$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$

Different heads can specialise: one may attend to syntactic structure, another to coreference, another to local context.

Since the Transformer has no recurrence or convolution, it knows nothing about token order without explicit **positional encoding**. The original paper uses sinusoidal functions:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

### Coding Task A — Multi-Head Attention

Implement a `MultiHeadAttention` class with:
- `__init__(d_model, num_heads)`: create weight matrices W_Q, W_K, W_V, W_O.
- `split_heads(x, batch_size)`: reshape from (batch, seq, d_model) to (batch, heads, seq, d_k).
- `forward(Q, K, V, mask=None)`: project, split heads, attend, concatenate, project out.

### Coding Task B — Sinusoidal Positional Encoding

Implement `positional_encoding(max_len, d_model)` returning a (max_len, d_model) matrix using the sinusoidal formulas above.

### Coding Task C — Visualise Positional Encodings

Plot the positional encoding matrix as a heatmap (positions on the y-axis, dimensions on the x-axis). Also plot the dot-product similarity matrix PE · PEᵀ to verify that nearby positions are more similar. Save to `positional_encoding.png`.

### Discussion Questions

1. Why does multi-head attention learn more useful representations than single-head attention of the same total dimension?
2. The original Transformer uses **fixed** sinusoidal positional encodings. Modern models (BERT, GPT) use **learned** positional embeddings. What are the trade-offs?
3. What is **Rotary Position Embedding (RoPE)** and why is it used in LLaMA and GPT-Neo? (High-level description only.)

---

## Part 4 — The Transformer Encoder Block

A single Transformer encoder layer combines multi-head attention with a **position-wise feed-forward network (FFN)**, connected via **residual connections** and **layer normalisation**:

```
x = LayerNorm(x + MultiHeadAttention(x, x, x))
x = LayerNorm(x + FFN(x))
```

The FFN is a two-layer MLP applied independently to each position:
$$\text{FFN}(x) = \max(0,\, xW_1 + b_1)\,W_2 + b_2$$

with inner dimension typically 4 × d_model (e.g. 512 → 2048 → 512).

Stacking N such layers (typically 6–24) produces the full Transformer encoder.

### Coding Task A — Feed-Forward Network

Implement `PositionwiseFFN(d_model, d_ff)` with ReLU activation, applied identically to every token position.

### Coding Task B — Transformer Encoder Layer

Implement `TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)` combining multi-head attention, FFN, residual connections, and layer normalisation.

### Coding Task C — Full Encoder Forward Pass

Stack 2 encoder layers and run a forward pass on a random batch (batch=2, seq=10, d_model=64). Verify output shape. Count total trainable parameters.

### Coding Task D — PyTorch Built-in Comparison

Use `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` for the same configuration. Confirm output shape matches your implementation. Compare parameter counts.

### Discussion Questions

1. Why are **residual connections** critical for training deep Transformers? What problem do they solve?
2. What does **layer normalisation** do and how does it differ from batch normalisation? Why is LayerNorm preferred in Transformers?
3. The FFN dimension is typically 4× the model dimension. What is the effect of increasing this ratio further? Name a model family that uses a larger ratio.

---

## Part 5 — Pre-trained Transformers with HuggingFace

Pre-training a Transformer from scratch requires massive compute (hundreds of GPU-years). HuggingFace provides thousands of pre-trained models ready for immediate use or fine-tuning.

**BERT** (Bidirectional Encoder Representations from Transformers, Devlin et al. 2018) is pre-trained with two objectives:
1. **Masked Language Modeling (MLM)**: 15% of tokens are masked; the model must predict them from context.
2. **Next Sentence Prediction (NSP)**: the model classifies whether two sentences are consecutive.

The result is contextual embeddings: the word *"bank"* has different representations in *"river bank"* vs *"bank account"*.

### Coding Task A — Load BERT and Extract Embeddings

Load `bert-base-uncased` and extract last-hidden-state embeddings for three sentences. Compare the [CLS] token embedding and the per-token embeddings.

### Coding Task B — Contextual vs Static Embeddings

Compare BERT's contextual embedding for the word *"bank"* in:
1. *"She sat on the bank of the river."*
2. *"He deposited money at the bank."*
3. *"The data bank was encrypted."*

Compute cosine similarities between the three *"bank"* vectors. Verify that contextual embeddings distinguish these senses. Compare to a static embedding (random or mean-pooled) where all three would be identical.

### Coding Task C — Masked Language Modeling

Use `fill-mask` pipeline with `bert-base-uncased`. Test the model's world knowledge and syntactic awareness on five masked sentences covering: grammar (subject-verb agreement), world knowledge, analogy, and idiom completion.

### Discussion Questions

1. What is the difference between **bidirectional** (BERT) and **autoregressive** (GPT) pre-training objectives? Which is better for classification tasks vs generation tasks?
2. Explain why BERT's [CLS] token is used for sequence-level classification. What does it learn during pre-training?
3. What is **sub-word tokenisation** (WordPiece / BPE) and why is it used instead of word-level or character-level tokenisation?

---

## Part 6 — Fine-tuning BERT for Text Classification

Pre-trained Transformers are **fine-tuned** by adding a task-specific head (e.g. a linear classifier on the [CLS] embedding) and training on labeled data. Fine-tuning requires far less data and compute than training from scratch.

### Coding Task A — Zero-Shot Baseline

Use the HuggingFace `text-classification` pipeline with `distilbert-base-uncased-finetuned-sst-2-english` to evaluate on 50 samples from the SST-2 validation set. Record accuracy.

### Coding Task B — Manual Fine-tuning Loop

Fine-tune `distilbert-base-uncased` on 200 SST-2 training examples for 3 epochs. Use AdamW with learning rate 2e-5. Evaluate on the same 50-sample validation set. Compare accuracy before and after fine-tuning.

### Coding Task C — Learning Curve

Fine-tune on 50, 100, 200, and 400 training examples (3 epochs each). Plot validation accuracy vs training set size. Save to `finetune_learning_curve.png`.

### Coding Task D — Confusion Matrix

For the 200-sample fine-tuned model, plot a confusion matrix on the validation set. Identify which sentiment is harder to classify correctly.

### Discussion Questions

1. Why does fine-tuning a 66M-parameter DistilBERT on only 200 examples still achieve reasonable accuracy? What does this tell us about the representations learned during pre-training?
2. What is **catastrophic forgetting** in the context of fine-tuning? How does the small learning rate (2e-5) help?
3. Describe **LoRA** (Low-Rank Adaptation) as an alternative to full fine-tuning. What memory advantage does it offer?

---

## Part 7 — Attention Visualisation and Why Transformers Transformed NLP

### Coding Task A — BERT Attention Head Visualisation

Load `bert-base-uncased` with `output_attentions=True`. For the sentence *"The cat sat on the mat"*, extract attention weights from layer 6 (middle layer). Visualise four different attention heads as heatmaps. Describe what each head appears to attend to (e.g. local context, syntactic structure, [CLS] aggregation).

### Coding Task B — Attention Pattern Classification

For each of the following sentences, run BERT and identify which attention head most strongly attends the subject to its verb:
1. *"The dogs bark loudly."*
2. *"The student who won the award studies biology."*
3. *"Scientists published their results last week."*

Report the (layer, head) indices with the highest subject→verb attention weight.

### The Transformer Revolution: A Timeline

| Year | Milestone | Impact |
|------|-----------|--------|
| 2017 | Transformer (Vaswani et al.) | Replaces RNNs for MT; 25× training speedup |
| 2018 | BERT (Devlin et al.) | Pre-train → fine-tune paradigm; state-of-the-art on 11 tasks |
| 2018 | GPT-1 (Radford et al.) | Autoregressive pre-training for generation |
| 2019 | GPT-2; RoBERTa; XLNet | Scale and training strategy matter enormously |
| 2020 | GPT-3 (175B params) | Few-shot prompting; in-context learning emerges |
| 2021 | CLIP; DALL-E | Transformers conquer vision and multimodal tasks |
| 2022 | ChatGPT; Instruction tuning | RLHF aligns generation to human preferences |
| 2023–24 | GPT-4; Gemini; LLaMA; Claude | Reasoning, coding, multimodal at scale |

### Discussion Questions

1. The Transformer's success rests on three pillars: **parallelism**, **pre-training**, and **scale**. Explain each and why they reinforce each other.
2. What are **scaling laws** (Kaplan et al., 2020)? How do they predict model performance from compute budget alone?
3. Why did BERT's release in 2018 immediately render many task-specific architectures (hand-crafted BiLSTM + CRF for NER, etc.) obsolete?
4. Transformers have been applied to images (ViT), audio (Whisper), protein structures (AlphaFold 2), and code (Codex). What common property of these domains makes the Transformer applicable?
5. What are the key **limitations** of current Transformer-based LLMs? Name at least three (e.g. quadratic attention complexity, context length, hallucination, reasoning).

---

## Submission

- Complete all `TODO` sections in the notebook.
- Answer all written questions in the markdown cells.
- Include in your submission:
  - The **attention heatmap** from Part 2B
  - The **temperature comparison plot** from Part 2C
  - The **positional encoding visualisation** from Part 3C (`positional_encoding.png`)
  - The **fine-tuning learning curve** from Part 6C (`finetune_learning_curve.png`)
  - The **confusion matrix** from Part 6D
  - The **BERT attention head visualisations** from Part 7A
  - Your **attention pattern table** from Part 7B
- Submit your completed `.ipynb` file as instructed.
