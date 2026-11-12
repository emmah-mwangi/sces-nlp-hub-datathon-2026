# Team 07 — Datalynk
## Lingala Intent Classification

**Strathmore NLP Hub Datathon 2026 | Track A: Intent Classification**

| Project detail | Information |
|---|---|
| Team number | 07 |
| Team name | Datalynk |
| Track | A — Intent Classification |
| Language | Lingala |
| Dataset size | 250 accepted records |
| Number of intent classes | 10 |
| Application | Streamlit |
| Project type | Small-data NLP research prototype |

---

## 1. Project Overview

Lingala Intent Classification is a small-data Natural Language Processing (NLP) research prototype developed by Team 07 — Datalynk for the Strathmore NLP Hub Datathon 2026.

The project explores how machine learning can identify the intended action behind a Lingala text input. Given a sentence, the system predicts the intent category that best represents the user's request.

The project focuses on building and evaluating a reproducible intent-classification pipeline using a manually assembled, reviewed dataset and a machine-learning baseline.

The accompanying Streamlit application provides an interactive interface for demonstrating the classifier and exploring its predictions.

**Research objective:** To investigate how a lightweight NLP model can support intent recognition in Lingala while documenting the challenges of limited training data, language variation and the need for human linguistic review.

## 2. Intent Classification Categories

The dataset covers ten intent categories:

| Intent label | Description |
|---|---|
| `weather_query` | Asking about weather conditions |
| `datetime_query` | Asking about the current date or time |
| `alarm_set` | Requesting an alarm |
| `calendar_set` | Creating or scheduling a calendar event |
| `calendar_query` | Asking about calendar events |
| `lists_createoradd` | Creating a list or adding an item |
| `lists_query` | Asking about list contents |
| `play_music` | Requesting music playback |
| `news_query` | Asking for news or current affairs |
| `qa_definition` | Asking for a definition or explanation |

The labels distinguish between different user actions rather than simply grouping sentences by their vocabulary.

## 3. Dataset and Data Preparation

The project uses a dataset of **250 accepted Lingala records** distributed across the ten intent categories.

The dataset is maintained in:

`data/contribution.csv`

The dataset follows the starter kit's contribution and validation workflow. Structural validation was executed using:

```bash
python validate.py --track intent --data data/contribution.csv
```

**Validation result:** PASS — 250 records.

The validation process checks dataset structure and applicable contribution requirements. It does not independently establish linguistic correctness, translation quality, volunteer consent or semantic equivalence. These require human review and supporting documentation.

### Data quality and research considerations

- Intent labels are based on the project's ten-category taxonomy.
- Lingala wording and spelling require appropriate fluent-speaker review.
- Volunteer contributions are referenced using pseudonymous contributor codes.
- Duplicate and semantically similar records should be reviewed carefully.
- Dataset provenance, consent and contribution records should be maintained separately from private permission evidence.

## 4. Methodology and Model

The project uses the starter kit's CPU-based machine-learning baseline.

The pipeline includes:

1. Dataset validation.
2. Development and test evaluation using the project's established data splits.
3. Character-level TF-IDF feature extraction.
4. Logistic Regression classification.
5. Comparison against a majority-class baseline.
6. Evaluation using accuracy, macro-F1 and per-class F1 scores.

Character-level features provide a practical baseline for text classification where spelling variation and limited training data are important considerations.

The model and evaluation outputs are generated under:

```text
artifacts/
└── intent/
    ├── model.joblib
    ├── dev_metrics.json
    ├── test_metrics.json
    ├── test_predictions.csv
    └── per_label_report.csv
```

These are generated outputs. Their presence in the local project does not necessarily mean they are included in the Git repository or final submission package.

## 5. Evaluation Results

The project pipeline was executed on the contribution dataset.

### Development results

| Metric | Result |
|---|---:|
| Development macro-F1 | 0.921 |

### Held-out test results

| Metric | Result |
|---|---:|
| Test records | 50 |
| Accuracy | 0.960 |
| Macro-F1 | 0.960 |
| Majority baseline macro-F1 | 0.018 |

### Per-class test F1

| Intent | F1 score |
|---|---:|
| `weather_query` | 1.000 |
| `datetime_query` | 0.889 |
| `alarm_set` | 0.909 |
| `calendar_set` | 1.000 |
| `calendar_query` | 1.000 |
| `lists_createoradd` | 0.909 |
| `lists_query` | 0.889 |
| `play_music` | 1.000 |
| `news_query` | 1.000 |
| `qa_definition` | 1.000 |

**Interpretation:** The baseline achieved a test accuracy of 96% and macro-F1 of approximately 0.960 on the 50-record held-out test set. These results demonstrate promising performance on this particular dataset.

However, the dataset is relatively small. The results should not be interpreted as evidence of production-level performance or generalisation to all Lingala speakers, dialects and conversational contexts.

The held-out test set should remain reserved for final evaluation rather than repeated model tuning.

## 6. Application and Demonstration

The project includes a Streamlit application that provides an interactive interface to the intent-classification prototype.

The application is intended to demonstrate the model's classification workflow and communicate its research outputs.

Run the application locally using:

```bash
python -m streamlit run app.py
```

The application runs locally in a web browser using Streamlit's default local address.

**Note:** Application features should be described according to the functionality implemented and verified in the submitted version.

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- Internet access for initial dependency installation.

### Step 1: Clone the repository

```bash
git clone https://github.com/emmah-mwangi/Lingala-NLP-Datathon.git
cd Lingala-NLP-Datathon
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Validate the contribution dataset

```bash
python validate.py --track intent --data data/contribution.csv
```

### Step 6: Train the model

```bash
python train.py --track intent --data data/contribution.csv
```

### Step 7: Evaluate the model

```bash
python evaluate.py --track intent --data data/contribution.csv
```

### Step 8: Launch the application

```bash
python -m streamlit run app.py
```

The training and evaluation commands generate model and metric artifacts in the `artifacts/intent/` directory.

## 8. Project Structure

```text
Lingala-NLP-Datathon/
│
├── app.py
├── validate.py
├── train.py
├── evaluate.py
├── requirements.txt
│
├── data/
│   └── contribution.csv
│
├── artifacts/
│   └── intent/
│
├── tests/
│
├── templates/
│
├── examples/
│
├── taxonomy.csv
├── TRACKS.md
├── NEXT_STEPS.md
├── NATURAL_DATA.md
│
└── README.md
```

The structure above describes the main project components. Additional files may be present in the repository.

## 9. Team Roles and Contributions

| Member | Contributor code | Responsibility |
|---|---|---|
| Caroline | T07-A | Language and documentation lead: Lingala authoring, spelling conventions, annotation guide, dataset card, quality report, linguistic reviews and error analysis. |
| Emmah | T07-B | Frontend and data-upkeep lead: Streamlit demonstration, Record Log and contribution dataset upkeep, README, backup recording and volunteer permissions. |
| Tatyana | T07-C | Backend and pipeline lead: repository management, validation, training, evaluation, dataset splits, final test execution, frozen commit and submission package. |

### Volunteer contributions

Lingala volunteer contributions are credited using pseudonymous identifiers:

- T07-V1
- T07-V2
- T07-V3

Volunteer identities and private permission evidence are not published in the public repository.

Further contribution and AI-use disclosures are maintained in the project's supporting documentation.

## 10. Limitations and Ethical Considerations

This project is an academic research prototype and not a production-ready conversational AI system.

Its main limitations include:

- **Limited dataset size:** 250 accepted records cannot represent the full range of Lingala expressions and conversational contexts.
- **Language variation:** Spelling, dialect and everyday conversational differences may affect classification.
- **Human linguistic review:** Automated validation cannot replace fluent-speaker assessment.
- **Classification scope:** The model recognises ten predefined intent categories and cannot reliably identify requests outside those categories.
- **Model confidence:** Model probability scores are not necessarily calibrated measures of prediction certainty.
- **Generalisation:** Performance on the held-out test set does not guarantee similar results on unseen real-world conversations.

The project acknowledges the importance of transparent data provenance, informed volunteer contributions, appropriate attribution and responsible use of AI-assisted development tools.

## 11. Acknowledgements

Team 07 — Datalynk acknowledges the Strathmore NLP Hub for providing the Datathon framework and starter kit.

We also acknowledge the Lingala contributors who supported the development of the dataset and the linguistic review process.

The project builds upon the Strathmore NLP Hub Datathon Starter Kit v2.0, with project-specific data, implementation, evaluation and documentation.

---

**Team 07 — Datalynk**  
Strathmore NLP Hub Datathon 2026  
Track A — Lingala Intent Classification

VIDEO LINK :
https://docs.google.com/videos/d/1w63A-1fD8AFmrUtkqal7YBuWIwtVdri3uIvYZT48nds/play?usp=sharing

<video controls src="20261112-1311-52.8790558-1.mp4" title="Title"></video>
