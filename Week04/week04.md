# Week 04 — Counting, N-grams, Distributional Semantics, and HMMs

## Overview

This lab covers classic foundations that still show up everywhere in modern NLP: counting-based statistics, smoothed bigram language models, sentence probability/perplexity, vector semantics via co-occurrence, cosine similarity, and sequence modeling with Hidden Markov Models (HMMs).

## Learning Goals

- Understand **word counting** and how preprocessing decisions change counts.
- Train and evaluate a **bigram language model**, and explain **data sparsity**.
- Apply **add-one (Laplace) smoothing** to avoid zero probabilities.
- Compute **sentence probability** and **perplexity**.
- Build **co-occurrence vectors** and compute **cosine similarity**.
- Use distributional vectors for **word similarity** and interpret failures.
- Implement the core algorithms for **Hidden Markov Models** (forward and Viterbi).

## Setup

```bash
pip install numpy nltk
```

---

## Part 1 — Word Counting (Tokenization + Frequency)

Word counting looks simple, but it depends heavily on tokenization choices (case, punctuation, numbers, contractions, etc.). You’ll implement a small pipeline and compare outcomes.

### Coding Task A — Tokenize and Count

Implement:
- `tokenize(text: str) -> list[str]`
- `word_counts(tokens: list[str]) -> dict[str, int]`

**Requirements:**
- Lowercase everything.
- Split on whitespace.
- Remove tokens that are only punctuation.

### Coding Task B — Top-k and Zipf Check

Implement:
- `top_k(counts, k=20)` returning a list of `(token, count)` pairs.
- A short analysis: do the top counts look “Zipf-like” (very common words dominate)?

### Discussion Questions

1. If you keep punctuation as tokens, what changes in the top-20 list?
2. Why can “better tokenization” sometimes hurt a downstream model?

---

## Part 2 — Bigram Language Models

An n-gram language model estimates the probability of a token sequence by assuming a limited context window.

### Key Formula (Chain Rule + Bigram Assumption)

For tokens $w_1, w_2, \dots, w_N$:

$$
P(w_1^N) = P(w_1)\prod_{i=2}^{N} P(w_i \mid w_1^{i-1})
\approx P(w_1)\prod_{i=2}^{N} P(w_i \mid w_{i-1})
$$

### Coding Task A — Build Unigram and Bigram Counts

Given a tokenized corpus, build:
- `unigram_counts[w]`
- `bigram_counts[(w_prev, w)]`
- `vocab` (include special tokens `<s>`, `</s>`, `<UNK>`)

**Hint:** Add sentence boundary tokens (`<s>`, `</s>`) around each sentence before counting.

### Coding Task B — MLE Bigram Probability

Implement MLE (unsmoothed):

$$
P_{\text{MLE}}(w \mid w_{prev}) = \frac{c(w_{prev}, w)}{c(w_{prev})}
$$

Implement `p_bigram_mle(w_prev, w)`.

### Data Sparsity Check

Compute:
- Total number of observed bigram types: \(|\{(w_{prev}, w): c>0\}|\)
- Total possible bigrams: \(V^2\)
- Sparsity ratio: observed / possible

**Question:** Why does \(V^2\) make sparsity explode even for moderate \(V\)?

---

## Part 3 — Add-one (Laplace) Smoothing

Unsmoothed n-gram models assign probability 0 to any unseen bigram, which makes entire sentence probabilities 0.

### Key Formula (Add-one Smoothing)

$$
P_{\text{add1}}(w \mid w_{prev}) = \frac{c(w_{prev}, w) + 1}{c(w_{prev}) + V}
$$

### Coding Task A — Add-one Bigram Probability

Implement `p_bigram_add1(w_prev, w, unigram_counts, bigram_counts, V)`.

### Coding Task B — Unknown Words

Implement `normalize_token(w, vocab)` that maps unseen words to `<UNK>`, and ensure your probability functions use this normalization.

### Discussion Questions

1. What problem does add-one smoothing solve?
2. What undesirable effect does it have on frequent events?
3. If \(V\) is very large, what happens to probabilities under add-one?

---

## Part 4 — Sentence Probability and Perplexity

### Sentence Probability (Log Space Recommended)

Compute the log probability of a sentence (with boundaries):

$$
\log P(w_1^N) \approx \sum_{i=2}^{N} \log P(w_i \mid w_{i-1})
$$

### Coding Task A — Sentence Log Probability

Implement `sentence_logprob(tokens, p_bigram)` that returns the sum of log-probabilities (use `math.log`).

### Perplexity

$$
PP = \exp\left(-\frac{1}{N-1}\sum_{i=2}^{N}\log P(w_i \mid w_{i-1})\right)
$$

### Coding Task B — Perplexity

Implement `perplexity(corpus_sentences, p_bigram)` where `corpus_sentences` is a list of token lists.

**Report:**
- Perplexity of the training corpus vs a held-out corpus (if provided).
- Compare MLE vs add-one.

### Discussion Questions

1. Why is perplexity lower on training data than test data?
2. Why can smoothing increase test performance but worsen training perplexity?

---

## Part 5 — Co-occurrence Vectors

Distributional semantics: “You shall know a word by the company it keeps.” You’ll represent each word by counts of neighboring context words.

### Define a Co-occurrence Matrix

Pick a context window size \(k\) (e.g., 2). For each center word \(w\) and each context word \(c\) within \(k\) positions:
- increment `X[w, c] += 1`

### Coding Task A — Build Co-occurrence Counts

Implement `build_cooccurrence(sentences, window=2, min_count=5)`:
- Build vocabulary of words with frequency ≥ `min_count` (plus `<UNK>` if you want).
- Return a matrix `X` (as `numpy.ndarray`) and mappings `word_to_id`, `id_to_word`.

### Coding Task B — Vector for a Word

Implement `vector_for(word, X, word_to_id)` returning the row vector for `word` (or `<UNK>`).

### Data Sparsity Reflection

Compute and report:
- fraction of zeros in `X` (sparsity of co-occurrence)
- the most frequent context words (column sums)

---

## Part 6 — Word Similarity with Cosine Similarity

### Cosine Similarity

For vectors $u, v$:

$$
\cos(u,v) = \frac{u \cdot v}{\|u\|\|v\|}
$$

### Coding Task A — Cosine Similarity

Implement `cosine(u, v)` (handle zero vectors safely).

### Coding Task B — Most Similar Words

Implement `most_similar(query_word, X, word_to_id, id_to_word, top_k=10)`:
- Compute cosine between the query vector and every other word vector.
- Return top-k most similar (excluding the word itself).

### Analysis Prompts

Try at least 3 query words and answer:
- Which neighbors look like **synonyms** vs **topic-related**?
- Give one example where similarity is clearly wrong and explain why co-occurrence might fail (polysemy, rare words, small corpus, function words, etc.).

---

## Part 7 — Hidden Markov Models (HMMs)

We model a sequence of hidden states \(z_1,\dots,z_T\) (e.g., POS tags) generating observed tokens \(x_1,\dots,x_T\).

### HMM Parameters

- Initial distribution: \(\pi(z_1)\)
- Transition: \(A_{ij} = P(z_t=j \mid z_{t-1}=i)\)
- Emission: \(B_{j}(x) = P(x_t=x \mid z_t=j)\)

### Coding Task A — Estimate HMM Parameters from Labeled Data

Given sequences of `(state, word)` pairs, estimate:
- `pi[state]`
- `A[prev_state, state]`
- `B[state, word]`

**Important:** Use smoothing for emissions (and optionally transitions), or map rare words to `<UNK>`.

### Coding Task B — Forward Algorithm (Sequence Probability)

Implement the forward recursion:

$$
\alpha_1(j)=\pi(j)B_j(x_1),\quad
\alpha_t(j)=B_j(x_t)\sum_i \alpha_{t-1}(i)A_{ij}
$$

Return $P(x_1^T)=\sum_j \alpha_T(j)$.

**Hint:** Use log-space (`logsumexp`) if your sequences are long; otherwise, plain floats are fine for short labs.

### Coding Task C — Viterbi Decoding (Most Likely State Sequence)

Implement Viterbi:

$$
\delta_1(j)=\pi(j)B_j(x_1),\quad
\delta_t(j)=B_j(x_t)\max_i \delta_{t-1}(i)A_{ij}
$$

Also store backpointers to recover the best state path.

### Evaluation (If Gold States Provided)

Compute token-level accuracy of predicted states vs gold states.

### Discussion Questions

1. Why does Viterbi use `max` while forward uses `sum`?
2. What kinds of dependencies does an HMM fail to model in language?
3. Compare where HMMs are still useful vs modern neural sequence models.

---

## Submission

- Submit your completed notebook/script as instructed by your TA (e.g., `.ipynb` or `.py`).
- Your written answers must be in your own words.
