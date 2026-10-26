# Customer Support Intent Classifier

A multi-class NLP project that classifies customer support messages into 77 banking-related intents using a fine-tuned BERT model (`bert-base-uncased`).

## Project Overview

Customer support teams receive messages about card payments, bank transfers, cash withdrawals, account access, and other banking services. This project uses a Transformer-based text classifier to predict the most likely intent from a customer's message.

## Features

- Fine-tuned BERT text classification
- 77 customer support intent categories
- Top-3 intent predictions with confidence scores
- Validation and official test-set evaluation
- Per-class precision, recall, and F1 analysis
- Misclassification and confidence-threshold analysis
- Local inference using a saved model

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Tokenizers
- Pandas, NumPy
- Scikit-learn

## Dataset

The project uses the Banking77 dataset, containing customer support questions assigned to 77 intent categories.

The training dataset contains 10,003 examples, and the official test dataset contains 3,080 examples.

The training data was split into:
- Training: 8,002 examples
- Validation: 2,001 examples

The validation split was stratified by category using `random_state=42`. The official test set was kept separate for final evaluation.

## Model Training

- Base model: `bert-base-uncased`
- Maximum sequence length: 64 tokens
- Training batch size: 16
- Epochs: 3
- Learning rate: `2e-5`
- Training environment: Google Colab with a Tesla T4 GPU

The trained model and tokenizer were saved for local inference.

## Results

### Official test set

| Metric | Result |
|---|---:|
| Accuracy | 89.32% |
| Macro F1 | 89.24% |
| Weighted F1 | 89.24% |
| Correct predictions | 2,751 / 3,080 |
| Incorrect predictions | 329 / 3,080 |

### Validation set

| Metric | Result |
|---|---:|
| Accuracy | 90.05% |
| Macro F1 | 90.02% |
| Validation examples | 2,001 |

The similar validation and official test scores are encouraging evidence of generalization.

## Confidence-Based Routing Experiment

Predictions were evaluated at different minimum confidence thresholds on the validation set.

| Threshold | Coverage | Accuracy on accepted predictions |
|---|---:|---:|
| 50% | 86.0% | 96.0% |
| 60% | 78.9% | 97.4% |
| 70% | 69.5% | 98.2% |
| 80% | 49.4% | 99.6% |
| 90% | 4.1% | 100.0% |

Coverage is the percentage of messages whose predictions meet the threshold. In a potential support workflow, lower-confidence predictions could be routed for human review.

These are validation-set results, not a guarantee of production performance. Model confidence is not necessarily a calibrated probability of correctness.

## Error Analysis

The main errors involve semantically similar intents, including:

- Failed transfers versus transfers not received
- Declined transfers versus declined card payments
- Failed, pending, and reverted top-ups
- Unrecognized direct debit payments versus unrecognized card payments

These cases demonstrate why per-class evaluation and manual error inspection are important in addition to overall accuracy.

## Run Locally

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure the model is available

The saved model and tokenizer should be located at:

```text
models/banking77-bert/
```

The directory must contain the model weights, tokenizer files, configuration, and label mapping required by the prediction script.

### 4. Run a prediction

From the project root:

```powershell
python src/predict.py
```

The prediction script loads the local model and returns the most likely intents with confidence scores.

### 5. Run evaluation or analysis

```powershell
python src/evaluate.py
python src/inspect_errors.py
python src/analyze_confidence.py
```

These scripts evaluate the saved model and inspect its errors; they do not retrain it.

## Project Structure

```text
customer-support-classifier/
├── data/
│   └── train.csv
├── models/
│   └── banking77-bert/
├── reports/
│   ├── validation_misclassifications.csv
│   ├── validation_classification_report.csv
│   └── validation_confidence_analysis.csv
├── src/
│   ├── predict.py
│   ├── evaluate.py
│   ├── inspect_errors.py
│   ├── inspect_intent_examples.py
│   └── analyze_confidence.py
├── requirements.txt
└── README.md
```

## Limitations

- Some intent categories overlap semantically.
- A high confidence score does not guarantee a correct prediction.
- Results may vary on real-world messages that differ from the training dataset.
- The model is a prototype and should not independently make consequential financial decisions.

## Future Improvements

Potential improvements include confidence calibration, better handling of ambiguous intents, targeted training-data quality checks, and deployment through a REST API.

## Summary

This project demonstrates a practical NLP workflow: dataset preparation, Transformer fine-tuning, model saving, inference, quantitative evaluation, per-class error analysis, and confidence-based routing.
