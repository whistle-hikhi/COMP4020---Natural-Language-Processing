# Week 06 — NLP Pipeline

## Overview

This lab builds a complete, modular NLP pipeline from raw text to structured linguistic annotations. You will implement each classical stage — sentence segmentation, tokenization, normalization, part-of-speech tagging, named entity recognition, and dependency parsing — first from scratch to understand the mechanics, then compare with production-grade tools (spaCy). The pipeline you build will feed directly into a simple information-extraction system at the end.

## Learning Goals

- Understand the **stages of an NLP pipeline** and their ordering dependencies.
- Implement **rule-based sentence segmentation** and **tokenization**.
- Apply **text normalization** techniques: lowercasing, stop-word removal, stemming, lemmatization.
- Build a **part-of-speech (POS) tagger** using the Viterbi algorithm over a Hidden Markov Model.
- Implement a **rule-based named entity recognizer (NER)** using gazetteers and regex patterns.
- Parse **dependency structure** conceptually and annotate head–dependent relations.
- Compose pipeline stages into an **end-to-end system** and evaluate each stage.
- Compare your hand-built pipeline against **spaCy's industrial-strength pipeline**.

## Setup

```bash
pip install numpy nltk spacy matplotlib
python -m spacy download en_core_web_sm
```

---

## Part 1 — Sentence Segmentation

Raw text arrives as a flat string. Before any further analysis, it must be split into sentences. A rule-based segmenter looks for **sentence boundary markers** (`.`, `!`, `?`) while handling common exceptions (abbreviations, initials, ellipsis).

### Key Rules

| Pattern | Example | Boundary? |
|---------|---------|-----------|
| `.` followed by uppercase + space | `"...done. The next..."` | Yes |
| `.` after known abbreviation | `"Dr. Smith"`, `"U.S.A."` | No |
| `!` or `?` | `"Really? Yes!"` | Yes |
| `...` (ellipsis) | `"Well... maybe"` | No |

### Coding Task A — Rule-Based Segmenter

Implement `segment_sentences(text: str) -> list[str]` using regex and a set of abbreviations:

```python
ABBREVIATIONS = {"mr", "mrs", "dr", "prof", "sr", "jr", "vs", "etc", "u.s", "u.k"}
```

The function should:
1. Use a regex to find candidate sentence boundaries (`.`, `!`, `?` followed by whitespace and a capital letter or end-of-string).
2. Skip boundaries where the preceding token is a known abbreviation or a single uppercase letter (initial).
3. Return a list of stripped sentence strings.

### Coding Task B — Evaluate Against NLTK

Use `nltk.sent_tokenize` on the same text and compare sentence counts and boundaries. Report:
- Number of sentences (yours vs NLTK)
- Any sentences split differently

### Discussion Questions

1. Why must sentence segmentation come before tokenization in most pipelines?
2. Give two examples where a period does **not** mark a sentence boundary.
3. How would you handle multilingual text (e.g., `。` in Chinese, `।` in Hindi)?

---

## Part 2 — Tokenization and Normalization

Tokenization splits a sentence into **tokens** (words, punctuation, numbers). Normalization prepares tokens for downstream tasks by removing noise and reducing vocabulary size.

### Tokenization Strategies

| Strategy | Example input | Output |
|----------|--------------|--------|
| Whitespace | `"don't worry"` | `["don't", "worry"]` |
| Punctuation split | `"don't worry"` | `["don", "'", "t", "worry"]` |
| Regex word | `"don't worry"` | `["don't", "worry"]` |

### Normalization Pipeline

$$\text{raw token} \xrightarrow{\text{lowercase}} \xrightarrow{\text{stop-word removal}} \xrightarrow{\text{stemming/lemmatization}} \text{normalized token}$$

### Coding Task A — Regex Tokenizer

Implement `tokenize(sentence: str) -> list[str]` using the regex pattern:

```
r"[A-Za-z]+(?:[''][A-Za-z]+)*|\d+(?:[.,]\d+)*|[^\w\s]"
```

This captures: contractions, numbers with decimal separators, and punctuation.

### Coding Task B — Stop-Word Filter

Implement `remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]`. Use NLTK's English stop-word list. Note which words are removed and discuss the impact on downstream tasks.

### Coding Task C — Stemming vs Lemmatization

For a list of tokens, apply:
- Porter stemmer (`nltk.stem.PorterStemmer`)
- NLTK WordNet lemmatizer (`nltk.stem.WordNetLemmatizer`)

Show the output side-by-side for at least 10 varied words (e.g., `running`, `better`, `studies`, `flies`).

### Coding Task D — Full Normalization Function

Implement `normalize(sentence: str, remove_sw: bool = True, method: str = "lemma") -> list[str]` that chains tokenization → lowercasing → stop-word removal → stemming/lemmatization.

### Discussion Questions

1. Give an example where stemming produces a non-word and lemmatization does not.
2. When should you **keep** stop words (name two downstream tasks)?
3. What is the difference between a type and a token? How does tokenization affect type-token ratio?

---

## Part 3 — Part-of-Speech Tagging

POS tagging assigns a grammatical category (noun, verb, adjective, …) to each token. A **Hidden Markov Model (HMM)** tagger treats the tag sequence as hidden states and words as observations.

### HMM Components

| Symbol | Meaning | Formula |
|--------|---------|---------|
| $\pi_t$ | Initial tag probability | $P(\text{tag}_1)$ |
| $A_{t_i, t_j}$ | Transition probability | $P(t_j \mid t_i)$ |
| $B_{t, w}$ | Emission probability | $P(w \mid t)$ |

### Viterbi Algorithm

For each position $i$ and tag $t$:

$$
v_t(i) = \max_{t'} \left[ v_{t'}(i-1) \cdot A_{t', t} \cdot B_{t, w_i} \right]
$$

Backtrack from the highest-scoring final state to recover the tag sequence.

### Coding Task A — Estimate HMM Parameters

Given a list of `(word, tag)` tuples from a tagged corpus, implement:
- `estimate_transitions(tagged_sents) -> dict`: $P(t_j \mid t_i)$ with add-one smoothing
- `estimate_emissions(tagged_sents) -> dict`: $P(w \mid t)$ with add-one smoothing
- `estimate_initial(tagged_sents) -> dict`: $P(t_1)$ from sentence-initial tags

Use the Penn Treebank tagset available via `nltk.corpus.treebank`.

### Coding Task B — Viterbi Decoder

Implement `viterbi(tokens, transitions, emissions, initial, tagset) -> list[str]` that:
1. Initializes the trellis with `initial[t] * emissions[t].get(w_0, unk_prob)`.
2. Fills each column using the recurrence above.
3. Backtracks to return the best tag sequence.

Handle unknown words with a small uniform probability `unk_prob = 1e-6`.

### Coding Task C — Evaluate Tagger

Evaluate accuracy on a held-out 20% of the treebank:

$$
\text{Accuracy} = \frac{\text{# correctly tagged tokens}}{\text{# total tokens}}
$$

Report overall accuracy and accuracy broken down by: known words vs unknown words.

### Coding Task D — Compare with spaCy

Run `spacy_nlp(sentence).doc` and extract `.pos_` for each token. Compare tags (mapping Penn Treebank → Universal POS if needed) on 10 sample sentences.

### Discussion Questions

1. Why does the Viterbi algorithm use **max** (not sum) over previous states?
2. What is the **label bias problem** and which type of model suffers from it (HMM or CRF)?
3. How does an HMM tagger handle a word it has never seen during training?

---

## Part 4 — Named Entity Recognition

NER identifies spans of text that refer to real-world entities: persons (`PER`), organizations (`ORG`), locations (`LOC`), and others (`MISC`).

### Approaches

| Method | Strengths | Weaknesses |
|--------|----------|-----------|
| Gazetteer (lookup list) | Precise for known entities | Misses new/misspelled entities |
| Regex patterns | Good for structured entities (dates, phone numbers) | Cannot generalize |
| HMM / CRF | Learns from data, handles context | Needs labeled training data |

### BIO Tagging Scheme

Each token receives a label:
- `B-TYPE` — beginning of an entity of TYPE
- `I-TYPE` — inside (continuation) of an entity of TYPE
- `O` — outside any entity

Example: `"Barack Obama visited Paris"` → `B-PER I-PER O B-LOC`

### Coding Task A — Gazetteer NER

Build `gazetteer_ner(tokens: list[str], gazetteers: dict[str, set[str]]) -> list[str]`:
- `gazetteers` maps entity type → set of entity strings (may be multi-token).
- Use a **longest-match-first** strategy: try to match the longest possible span starting at each position.
- Return a BIO tag sequence.

Provide gazetteers with at least 20 entries each for PER, ORG, and LOC.

### Coding Task B — Regex NER

Implement `regex_ner(text: str) -> list[tuple[str, str]]` that extracts and labels:
- **Dates**: `January 15, 2024`, `15/01/2024`, `2024-01-15`
- **Times**: `3:45 PM`, `15:00`
- **Email addresses**: `user@example.com`
- **Phone numbers**: `+1-800-555-0199`, `(800) 555-0199`

Return a list of `(matched_string, entity_type)` tuples.

### Coding Task C — Merge and Evaluate

Combine gazetteer and regex NER into `ner_pipeline(text: str, gazetteers: dict) -> list[tuple[int, int, str]]` that returns `(start_char, end_char, entity_type)` spans, de-duplicated and sorted.

Evaluate on 20 manually annotated sentences using:
$$
\text{Precision} = \frac{|\text{predicted} \cap \text{gold}|}{|\text{predicted}|}, \quad
\text{Recall} = \frac{|\text{predicted} \cap \text{gold}|}{|\text{gold}|}, \quad
F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$

### Coding Task D — spaCy NER

Run spaCy's NER on the same 20 sentences. Compare F1 scores per entity type.

### Discussion Questions

1. Why is the BIO scheme preferred over a simple binary (entity/not-entity) label?
2. What causes **boundary errors** in NER (where the entity type is correct but the span is wrong)?
3. How would you extend this pipeline to handle **nested entities** (e.g., `[Bank of [America]_LOC]_ORG`)?

---

## Part 5 — Dependency Parsing (Conceptual + Annotation)

A **dependency parse** represents the syntactic structure of a sentence as a directed tree where each arc connects a **head** word to a **dependent** word, labelled with a grammatical relation.

### Relation Labels (Universal Dependencies)

| Label | Meaning | Example |
|-------|---------|---------|
| `nsubj` | Nominal subject | *She* runs |
| `obj` | Object | runs *marathons* |
| `amod` | Adjectival modifier | *fast* runner |
| `det` | Determiner | *the* runner |
| `prep` / `nmod` | Prepositional modifier | runs *in* the park |
| `ROOT` | Root of the sentence | She **runs** marathons |

### Coding Task A — Manual Annotation

For each of the following sentences, draw or list the dependency arcs (head → dependent, label):

1. `"The quick brown fox jumps over the lazy dog."`
2. `"Alice gave Bob a beautiful gift."`
3. `"The researchers from VinUniversity published a paper on NLP."`

### Coding Task B — Extract Arc Statistics with spaCy

Parse 50 sentences from a corpus of your choice with spaCy. Count:
- The 10 most frequent dependency relation labels
- The average dependency distance (|head index − dependent index|)
- The fraction of arcs that are **left-arc** vs **right-arc** in English

### Coding Task C — Projectivity Check

Implement `is_projective(arcs: list[tuple[int, int]]) -> bool` where `arcs` is a list of `(head, dependent)` index pairs. An arc $(i, j)$ is non-projective if there exists an arc $(k, l)$ such that $i < k < j < l$ or $l < i < k < j$ (the arcs cross).

### Discussion Questions

1. What is the difference between **constituency parsing** and **dependency parsing**?
2. Why is **projectivity** a useful property for parsing algorithms?
3. How would dependency parses help in the information extraction task in Part 6?

---

## Part 6 — End-to-End Pipeline and Information Extraction

### The Pipeline

```
Raw Text
   │
   ▼
Sentence Segmentation
   │
   ▼
Tokenization + Normalization
   │
   ▼
POS Tagging
   │
   ▼
NER
   │
   ▼
Dependency Parsing  (via spaCy)
   │
   ▼
Relation Extraction
```

### Coding Task A — Build the Pipeline Class

Implement a `NLPPipeline` class:

```python
class NLPPipeline:
    def __init__(self, use_spacy_pos=False, use_spacy_ner=False): ...
    def process(self, text: str) -> list[dict]: ...
    #   Returns a list of sentence dicts:
    #   {
    #     "sentence": str,
    #     "tokens": list[str],
    #     "pos_tags": list[str],
    #     "entities": list[(start, end, type)],
    #     "dep_arcs": list[(head_idx, dep_idx, label)]   # spaCy only
    #   }
```

Use your implementations from Parts 1–4 (with optional spaCy override flags).

### Coding Task B — Relation Extraction

Implement `extract_relations(sentence_dict: dict) -> list[tuple]` that applies simple **pattern-based relation extraction**:

- **`works_for(PER, ORG)`**: pattern `PER (is | was | joined | works at | CEO of) ORG`
- **`located_in(ORG, LOC)`**: pattern `ORG (in | based in | headquartered in) LOC`
- **`born_in(PER, LOC)`**: pattern `PER (born in | from | native of) LOC`

Return a list of `(subject_entity, relation, object_entity)` triples.

### Coding Task C — Evaluate the Full Pipeline

Run `NLPPipeline` on the following test paragraph and report:

```
Dr. Jane Smith, a researcher at VinUniversity in Hanoi, published a landmark paper on
transformer-based NLP models. She was born in Ho Chi Minh City. Her collaborator,
Prof. John Doe from MIT in Cambridge, contributed to the section on dependency parsing.
The paper was released on March 15, 2024 at 9:00 AM and received over 500 citations.
```

Expected extractions (partial):
- `works_for(Jane Smith, VinUniversity)`
- `located_in(VinUniversity, Hanoi)`
- `born_in(Jane Smith, Ho Chi Minh City)`
- `works_for(John Doe, MIT)`
- DATE entity: `March 15, 2024`

### Coding Task D — Pipeline Benchmark

Time each pipeline stage on a 100-sentence corpus and report:
- Stage name
- Average time per sentence (ms)
- Cumulative time

Which stage is the bottleneck?

### Discussion Questions

1. What are the **error propagation** implications of a sequential pipeline? Give a concrete example.
2. How does **joint modeling** (e.g., a single neural model that does NER + RE together) address the error propagation problem?
3. Name two scenarios where a **rule-based pipeline** is preferable to a neural end-to-end model.

---

## Part 7 — Custom spaCy Component

Add a custom spaCy pipeline component that detects **Vietnamese person names** using a simple rule: sequences of 2–4 title-cased tokens where the first token is one of `{"Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Phan"}`.

### Coding Task

```python
import spacy
from spacy.language import Language

@Language.factory("viet_person_ner")
def create_viet_ner(nlp, name):
    return VietPersonNER(nlp)

class VietPersonNER:
    FAMILY_NAMES = {"Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Phan"}

    def __call__(self, doc):
        # TODO: identify spans matching the pattern
        # TODO: add them as PERSON entities without overwriting existing ones
        pass
```

Test on: `"Nguyen Van An and Tran Thi Bich work at VinUniversity."` Expected: two PERSON entities.

### Discussion Question

1. How would you handle name ambiguity (e.g., `"Le"` as a Vietnamese family name vs the English word *le*)?

---

## Submission

- Submit your completed notebook/script as instructed by your TA (`.ipynb` or `.py`).
- Written answers must be in your own words.
- Include timing results for Part 6D as a table or plot.
