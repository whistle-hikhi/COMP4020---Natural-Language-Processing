## Setup

```bash
pip install nltk spacy transformers datasets scikit-learn
python -m spacy download en_core_web_sm
```

## Sample Dataset

```python
data = [
    {"text": "Scientists discovered a new planet in the solar system.", "label": 0},
    {"text": "Shocking! Aliens built the pyramids!", "label": 1},
    {"text": "Government releases official economic report.", "label": 0},
    {"text": "Doctors hate him! Cure cancer with lemon water!", "label": 1},
]
```

Convert to a Pandas `DataFrame`.

## Part 1 — NLTK

**Tasks:**

- Tokenization
- Stopword removal
- Stemming

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text):
    # Your code here
    # Tokenization
    # Removing Stopword
    # Stemming
    return stems


print(preprocess(data[1]["text"]))
```

### 💡 Discussion

- What information is lost?
- Is stemming always good?

## Part 2 — spaCy

- POS tagging
- Named Entity Recognition
- Dependency parsing

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(data[0]["text"])

for token in doc:
    # Your code here
    # Print out text and it POS


doc = nlp("Apple bought a UK startup for $1 billion")
for ent in doc.ents:
    # Your code here
    # Print out NER
```

### 💡 Discussion

- How is spaCy different from NLTK?
- Why is NER useful in fake news?

## Part 3 — Transformers 

We use:

- `distilbert-base-uncased`

```python
from transformers import pipeline

classifier = pipeline(
    # Your code here
)

for item in data:
    result = classifier(item["text"])
    print(item["text"])
    print(result)
```

### 💡 Discussion

- Why does this require no preprocessing?
- Where is tokenization happening?
- Why does it perform better?

# Submission
- Submit a pdf file report (pdf file should show the results and answered questions)
- Submit a .ipynb notebook or .py answer