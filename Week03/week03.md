# Week 03 — Language Models: From N-grams to Prompt Engineering

## Overview

This lab takes you from the mathematical foundations of language models to hands-on experimentation with modern transformer-based generation. You will build a statistical bigram model from scratch, then move to GPT-2 to study decoding behavior, hallucination, and prompt engineering.

## Learning Goals

- Understand how sentence probability is decomposed using the **chain rule** of probability.
- Implement a **bigram language model** with **Laplace (add-one) smoothing**.
- Sample text and compute **perplexity** using your model.
- Experiment with **decoding strategies** (greedy, sampling, top-p) in a transformer.
- Observe **hallucination** in LLMs and apply prompting strategies to reduce it.
- Design **structured prompts** and validate model outputs programmatically.

## Setup

```bash
pip install nltk transformers torch
```

---

## Part 1 — Sentence Plausibility and Probability of Text

A language model estimates the likelihood of a sequence of words. Before writing any code, reason about what makes a sentence "natural."

**Tasks:**
- You are given a list of sentences — rank them from most to least natural.
- Identify which sentence has a **grammar error** (syntactic problem) and which has a **meaning/semantic error**.
- Briefly explain (2–3 lines) why a language model might still assign non-zero probability to clearly odd sentences.

> **Hint:** Think about what the chain rule says — each word's probability depends only on its local context, not global coherence.

---

## Part 2 — Build a Bigram Language Model (with Smoothing)

You will train a simple bigram model on a small toy corpus and implement several functions step by step.

### Key Formula

Add-one (Laplace) smoothed bigram probability:

$$
P(w_2 \mid w_1) = \frac{\text{count}(w_1, w_2) + 1}{\text{count}(w_1) + V}
$$

where $V$ is the vocabulary size.

### Coding Task A — Smoothed Bigram Probability
Implement `p_bigram_addone(w1, w2)` that returns the smoothed probability of `w2` given `w1`.

> Use `bigram_counts` and `unigram_counts` already built in the notebook. Make sure you handle the denominator correctly with the vocabulary size $V$.

### Coding Task B — Handle Unknown Tokens
Implement `normalize_token(w)` so that words not in the vocabulary are mapped to `<UNK>`.

> Return the word itself if it exists in `vocab`, otherwise return the `UNK` constant.

### Coding Task C — Next-Word Prediction
Implement `top_k_next(w1, k=5)` that returns the **k most probable next tokens** given a context word.

> Iterate over all vocabulary words, compute their probability given `w1`, and return the top-k sorted by probability descending.

### Coding Task D — Sample Text from the Bigram Model
Implement `sample_next(w1)` that **randomly samples** the next token using the bigram probability distribution.

> Use `random.choices` or manually build a probability distribution over the vocabulary and sample from it.

### Coding Task E — Perplexity
Implement `perplexity(tokens_seq)` using:

$$
PP = \exp\!\left(-\frac{1}{N}\sum_{i=2}^{N}\log P(w_i \mid w_{i-1})\right)
$$

> Iterate over consecutive pairs of tokens in the sequence. Use `math.log` and `math.exp`. Be careful with index bounds — the sum starts at $i=2$ (the second token).

### Discussion Questions (answer 2–3)
1. Compare `PP(test_1)` vs `PP(test_2)`: which is lower and why?
2. What does add-one smoothing fix? What trade-off does it introduce?
3. When would n-gram models struggle compared to transformers?

---

## Part 3 — Transformer Generation and Decoding Controls

Using GPT-2, you will generate text under different decoding configurations and measure the effect on output diversity and repetition.

### Coding Task A — Compare Decoding Strategies
Complete the two missing generator calls:
- `out2`: sampling with `temperature=1.0` and `top_p=0.9`
- `out3`: sampling with `temperature=1.5` and `top_p=0.95`

> Pass `do_sample=True` along with the appropriate `temperature` and `top_p` arguments to `generator(...)`.

### Coding Task B — Trigram Repetition Ratio
Implement `trigram_repetition_ratio(text)` that tokenizes the text and returns the fraction of 3-grams that appear more than once.

> A ratio of 0 means no repeated trigrams. A ratio of 1 means every trigram is repeated. Tokenize simply with `text.split()`.

### Discussion Questions (answer 2–3)
1. Which decoding setting produced the most **creative** output? Which produced the most **stable** output?
2. How did the repetition ratio change across settings?
3. Why might sampling help or hurt factuality?

---

## Part 4 — Hallucination and Self-Check Prompting

You will provoke a hallucination from GPT-2 and then construct a prompt that asks the model to express uncertainty and list what it would need to verify.

### Coding Task — Ask for Uncertainty + Evidence
Complete the `checked` variable by calling `generator` with `prompt_check` already defined for you.

> Use `do_sample=True`, `temperature=0.7`, and a reasonable `max_new_tokens` (e.g. 150). Extract `[0]["generated_text"]` from the result.

### Discussion Questions (answer 2–3)
1. Did the model acknowledge uncertainty? Quote one phrase that signals uncertainty (or note if none appeared).
2. Why do LLMs hallucinate (1–2 sentences)?
3. Give one high-stakes domain where hallucination is especially risky and explain why.

---

## Part 5 — Prompt Engineering with Structured Outputs

You will engineer a prompt that forces the model to return a JSON object with a specific schema, then validate it with Python.

### Coding Task A — Create a Structured Prompt
Complete `ask_structured(question)` so it calls `generator` with the structured prompt and returns the raw generated text string.

> The prompt template is already written — just call `generator(prompt, ...)` and return the text. Use `max_new_tokens=200` and `do_sample=True`.

### Coding Task B — Extract and Validate JSON Output
The extraction and validation helpers are already provided. Your job is to **run** them on the response from Task A and interpret the results.

> Understand what `extract_first_json` and `validate_payload` do. Run the validation cell and report whether the model followed the schema.

### Discussion Questions (answer 2–3)
1. Did the model follow the JSON format? If not, what went wrong?
2. Why does forcing a structure sometimes reduce hallucination?
3. Rewrite this vague prompt to be safer: **"What is the best financial model used today?"**

---

## Submission

- Submit your completed **`.ipynb` notebook**.
- Follow the academic integrity policy — your explanations must be in your own words.
