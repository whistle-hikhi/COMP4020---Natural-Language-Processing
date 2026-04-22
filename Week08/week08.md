# Week 08 — Data Annotation for NLP

## Overview

This lab focuses on the theory and practice of data annotation in NLP. Annotation is the process of labelling raw text with linguistic or semantic information to create training and evaluation datasets. You will explore common annotation schemes, measure label quality with inter-annotator agreement (IAA) metrics, and build a small annotated dataset from scratch.

## Learning Goals

- Understand why **high-quality annotated data** is the foundation of supervised NLP.
- Apply common annotation schemes: **POS tagging**, **Named Entity Recognition (NER)**, and **sentiment**.
- Measure inter-annotator agreement with **Cohen's κ**, **Fleiss's κ**, and **Krippendorff's α**.
- Identify and analyse sources of **annotator disagreement**.
- Simulate a simple **active learning** loop to reduce annotation cost.
- Reflect on ethical and practical challenges in crowdsourced annotation.

## Setup

```bash
pip install nltk scikit-learn pandas matplotlib krippendorff
```

---

## Part 1 — Annotation Schemes and Label Sets

Annotation schemes define what labels annotators apply and what guidelines govern their decisions.

### Task 1A — POS Annotation

Manually assign Penn Treebank POS tags to a short sentence using the reference sheet provided. Compare your tags with those produced by NLTK's default tagger.

### Task 1B — NER Annotation

Given five short sentences, apply BIO (Beginning-Inside-Outside) tags for the entity types PER (person), ORG (organisation), LOC (location), and MISC (miscellaneous).

### Task 1C — Sentiment Annotation

Assign a sentiment label (Positive / Neutral / Negative) to each of ten short reviews. Record your confidence (High / Medium / Low) alongside each label.

### Discussion Questions

1. What is the difference between **BIO** and **BIOES** tagging schemes for NER? When would BIOES be preferred?
2. Why is it important to write explicit annotation guidelines before labelling starts?
3. Give one example of a sentence where the correct sentiment label is genuinely ambiguous.

---

## Part 2 — Inter-Annotator Agreement

IAA measures how consistently multiple annotators apply the same labels. Raw percentage agreement ignores chance agreement; Cohen's κ corrects for this.

### Coding Task A — Cohen's κ

Implement Cohen's κ from scratch for two annotators with categorical labels.

### Coding Task B — Fleiss's κ

Implement Fleiss's κ for three or more annotators.

### Coding Task C — Krippendorff's α

Use the `krippendorff` library to compute α for ordinal annotation (e.g., severity ratings).

### Coding Task D — Confusion Analysis

Build an annotator confusion matrix and identify the most frequently confused label pairs.

### Discussion Questions

1. A pair of annotators achieves 85% raw agreement on a binary task where one label covers 90% of items. Compute the expected chance agreement and the resulting κ. What does this tell you?
2. What κ value is generally considered "substantial" agreement? What threshold is often required before publishing a dataset?
3. When would Krippendorff's α be preferred over Cohen's κ?

---

## Part 3 — Annotating a NER Dataset

Apply BIO-NER annotation to a small corpus and calculate IAA between two simulated annotators.

### Coding Task A — Tokenise and BIO-tag

Tokenise five news sentences and apply BIO-NER tags.

### Coding Task B — Simulate a Second Annotator

Apply a simple rule-based "second annotator" (capitalisation heuristic) and compare its labels with yours.

### Coding Task C — Token-level IAA

Compute Cohen's κ at the token level between the two annotators.

### Coding Task D — Span-level F1

Compute span-level precision, recall, and F1 between the two annotators treating one as gold.

### Discussion Questions

1. Why is token-level κ often misleadingly high for NER even when annotators disagree on entity boundaries?
2. What information is lost when you compute token-level agreement instead of span-level agreement?

---

## Part 4 — Sentiment Annotation and Adjudication

Simulate a three-annotator sentiment annotation task and practise adjudication.

### Coding Task A — Simulate Three Annotators

Generate annotations for 20 sentences from three annotators with different noise levels.

### Coding Task B — Majority Vote

Implement majority-vote label aggregation.

### Coding Task C — Fleiss's κ

Compute Fleiss's κ for the three annotators across all 20 items.

### Coding Task D — Adjudication

Identify items where all three annotators disagree and propose an adjudication rule.

### Discussion Questions

1. How does majority voting differ from Dawid–Skene expectation-maximisation for aggregating crowd labels?
2. What are two strategies for handling items where all annotators disagree after majority voting?
3. How would you design an annotation study to minimise annotator fatigue?

---

## Part 5 — Active Learning for Annotation

Active learning selects the most informative examples to annotate, reducing the total annotation budget needed to reach a target model performance.

### Coding Task A — Train a Baseline Classifier

Train a logistic-regression sentiment classifier on a small seed set.

### Coding Task B — Uncertainty Sampling

Implement least-confidence uncertainty sampling to select the next batch of examples to annotate.

### Coding Task C — Active Learning Loop

Run five rounds of annotation + retraining and plot learning curves (accuracy vs. number of annotated examples) for random sampling vs. uncertainty sampling.

### Discussion Questions

1. What is the **cold-start problem** in active learning and how is it typically addressed?
2. Name two strategies other than least-confidence uncertainty sampling.
3. What is a potential bias introduced by active learning that could hurt model generalisation?

---

## Part 6 — Crowdsourced Annotation Ethics and Quality

### Discussion Tasks

1. Identify **three potential biases** that might arise when crowdsourcing sentiment annotation on a platform like Amazon Mechanical Turk.
2. Explain the concept of **gold standard items** (honeypots) and how they are used to detect low-quality annotators.
3. Describe one fairness concern related to paying annotators minimum wage for complex linguistic judgements.
4. How does **annotation schema design** affect the resulting dataset's usefulness for downstream tasks?

---

## Submission

- Complete all `TODO` sections in the notebook.
- Answer all written questions in the markdown cells.
- Include the IAA summary table from Part 2, the NER span-level F1 table from Part 3, and the active learning learning-curve plot from Part 5.
- Submit your completed `.ipynb` file as instructed.
