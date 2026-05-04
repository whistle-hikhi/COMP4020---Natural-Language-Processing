# Week 10 — Machine Translation

## Overview

This lab introduces the theory and practice of **Machine Translation (MT)**. You will implement the BLEU metric from scratch, compute complementary metrics (chrF, TER), use a pre-trained **Neural MT model** (MarianMT) for real translation, analyse and correct MT errors through post-editing, and explore domain adaptation challenges. The lab closes with a discussion of gender bias and ethical considerations in high-stakes MT deployment.

## Learning Goals

- Understand the core challenges in MT and the major approaches (statistical, neural).
- Implement **BLEU score** from scratch: modified n-gram precision and brevity penalty.
- Use `sacrebleu` for standardised BLEU and **chrF** evaluation.
- Implement **TER** (Translation Edit Rate) from a word-level Levenshtein distance.
- Translate text with **MarianMT** from HuggingFace and tune beam search.
- Classify and correct MT errors through **post-editing**.
- Evaluate the **domain gap** between general and specialised (medical) MT.
- Reflect on **gender bias** and ethical deployment of MT systems.

## Setup

```bash
pip install sacrebleu transformers sentencepiece torch nltk
```

---

## Part 1 — MT Fundamentals and Error Taxonomy

Machine Translation must handle a much richer set of linguistic phenomena than classification tasks: morphological agreement, idiomatic expressions, word order differences, discourse-level coherence, and domain-specific terminology.

### MT Error Taxonomy

| Error Type | Description |
|------------|-------------|
| Omission | Source word/phrase missing from the translation |
| Addition | Extra words not present in the source |
| Mistranslation | Incorrect word choice or meaning |
| Word order | Correct words but in wrong sequence |
| Punctuation | Incorrect punctuation marks |
| Morphology | Wrong inflection, tense, or agreement |
| Terminology | Incorrect domain-specific term |
| Fluency | Grammatically awkward though roughly correct |

### Task 1A — Error Classification

Classify five English→Vietnamese MT outputs using the error taxonomy. Identify the error type(s) and write a corrected translation.

### Task 1B — MT System Comparison

Rank three MT outputs for the same source sentence from best to worst, with justification.

### Discussion Questions

1. What makes MT fundamentally harder than sentiment classification? Name three linguistic phenomena.
2. Distinguish between **adequacy** (meaning preserved?) and **fluency** (grammatically natural?).
3. Why is literal word-for-word translation often incorrect? Give an example idiom.

---

## Part 2 — BLEU Score from Scratch

**BLEU** (Bilingual Evaluation Understudy) is the dominant automatic MT metric. It computes a geometric mean of modified n-gram precisions (orders 1–4) multiplied by a brevity penalty to discourage short outputs.

$$\text{BLEU} = \text{BP} \times \exp\!\left(\sum_{n=1}^{4} \tfrac{1}{4} \log p_n\right)$$

$$\text{BP} = \begin{cases} 1 & c > r \\ e^{1 - r/c} & c \leq r \end{cases}$$

### Coding Task A — Modified N-gram Precision

Implement `modified_precision(hypothesis, references, n)`. For each hypothesis n-gram, clip its count to the maximum count of that n-gram across all references, then divide by the total hypothesis n-gram count.

### Coding Task B — Brevity Penalty

Implement `brevity_penalty(hypothesis_length, reference_length)`.

### Coding Task C — Sentence and Corpus BLEU

Implement `sentence_bleu` with add-1 smoothing for zero-count n-grams, and `corpus_bleu` that aggregates counts across all sentences before computing the score.

### Coding Task D — Validate Against sacrebleu

Compare your implementation to `sacrebleu.corpus_bleu` on a five-sentence English→French evaluation set.

### Discussion Questions

1. Why is **modified** precision used instead of standard precision?
2. Why can per-sentence BLEU scores be unreliable?
3. A translation gets BLEU = 0 despite conveying the correct meaning. How is this possible?
4. What does the brevity penalty prevent?

---

## Part 3 — Additional MT Metrics

### Coding Task A — chrF with sacrebleu

Compute corpus chrF and per-sentence chrF alongside BLEU for the EN→FR set.

### Coding Task B — TER from Scratch

Implement `ter_score(hypothesis, reference)` using word-level Levenshtein distance (insertions, deletions, substitutions), normalised by reference length.

### Coding Task C — Metric Degradation Plot

Simulate progressive word-order degradation by swapping adjacent words and plot BLEU, chrF, and (1 − TER) as functions of degradation level. Save to `mt_metrics_degradation.png`.

### Discussion Questions

1. Why is chrF more robust than BLEU for morphologically rich languages?
2. A hypothesis has low BLEU but high chrF — what translation pattern explains this?
3. What does TER > 1.0 mean in practice?
4. Why should you report multiple metrics rather than BLEU alone?

---

## Part 4 — Neural Machine Translation with MarianMT

**MarianMT** models are compact sequence-to-sequence Transformers pre-trained on OPUS parallel corpora. They are available for hundreds of language pairs via HuggingFace.

### Coding Task A — Load the Model

Load `Helsinki-NLP/opus-mt-en-fr` using `MarianTokenizer` and `MarianMTModel`.

### Coding Task B — Translate with Beam Search

Implement `translate(texts, tokenizer, model, num_beams, max_length)` using `model.generate()` and `tokenizer.batch_decode()`.

### Coding Task C — Evaluate with BLEU and chrF

Compare MarianMT translations with the provided imperfect hypotheses from Part 2 using BLEU, chrF, and TER.

### Coding Task D — Beam Size Experiment

Translate with beam sizes 1, 2, 4, 8. Plot BLEU (left axis) and inference time (right axis) against beam size. Save to `bleu_vs_beam.png`.

### Discussion Questions

1. What is beam search and why does it outperform greedy decoding?
2. At what beam size do you observe diminishing BLEU returns?
3. What do the encoder and decoder each learn to do in a seq2seq Transformer?
4. How would back-translation improve BLEU for a low-resource language pair?

---

## Part 5 — Post-Editing Analysis

**Post-editing (PE)** corrects MT output to professional quality. Measuring PE effort guides decisions about when to deploy MT in production.

### Coding Task A — Post-editing Effort with TER

Compute TER between raw MT output and human post-edited versions for five segments.

### Coding Task B — Word-level Diff

Implement `word_diff(mt, pe)` using the Levenshtein traceback to count insertions, deletions, and substitutions separately.

### Coding Task C — Hands-on Post-editing Task

Correct five MT outputs with deliberate errors into fluent French. The errors include prepositional calques, false cognates, and morphological agreement mistakes. Compute TER for your edits.

### Discussion Questions

1. What is the difference between **light post-editing** and **full post-editing**?
2. Why is TER a better proxy for post-editing effort than BLEU?
3. Which error types did you encounter most in Task 5C? Why might they be systematic?

---

## Part 6 — Domain Adaptation and Low-Resource MT

Pre-trained MT models are typically trained on general-domain data (news, parliamentary proceedings). Performance drops in specialised domains due to terminology, syntactic style, and low training coverage.

### Coding Task A — Domain Gap Analysis

Translate three general-domain and three medical-domain sentences with MarianMT. Compute corpus BLEU and chrF for each domain. Inspect medical translations for terminology errors.

### Coding Task B — OOV Subword Rate

Implement `oov_rate(text, tokenizer)` that measures the fraction of whitespace-tokenised words that the MarianMT tokeniser splits into more than one subword token. Compare rates between domains.

### Discussion Questions

1. Why does MT quality drop in the medical domain? Name two domain-specific linguistic challenges.
2. What is **domain adaptation** for NMT? Describe two adaptation strategies.
3. What is **back-translation** and why is it useful for low-resource pairs?
4. Explain **zero-shot cross-lingual transfer** in multilingual models (NLLB, mBART). When would you use it?

---

## Part 7 — MT Ethics and Human Evaluation (Discussion)

### Gender Bias

Translate five English sentences with gender-neutral professional nouns (engineer, nurse, CEO, teacher, programmer) using MarianMT. Record which gender the model defaults to and explain why bias arises from training data imbalance.

### Human Evaluation

Rate the medical domain translations from Part 6A on:
- **Adequacy** (1–5): does the translation convey the full meaning?
- **Fluency** (1–5): is the translation grammatically natural?

### High-Stakes Deployment

1. Identify two risks of deploying unvalidated MT in medical or legal contexts.
2. What safeguards would you recommend before deploying MT in a hospital?
3. Why is back-translation of patient records without human review potentially dangerous?

---

## Submission

- Complete all `TODO` sections in the notebook.
- Answer all written questions in the markdown cells.
- Include in your submission:
  - The **BLEU / chrF / TER comparison table** from Part 2D
  - The **metric degradation plot** from Part 3C (`mt_metrics_degradation.png`)
  - The **beam size vs BLEU plot** from Part 4D (`bleu_vs_beam.png`)
  - The **domain gap table** from Part 6A
  - Your **post-edited translations** from Part 5C
  - Your **gender bias observations** from Part 7
- Submit your completed `.ipynb` file as instructed.
