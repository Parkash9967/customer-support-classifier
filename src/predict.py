
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "banking77-bert"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64


# ==========================================
# 2. Load the trained model and tokenizer
# ==========================================

def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Check that you extracted the model into models/banking77-bert/"
        )

    print(f"Loading model from: {MODEL_DIR}")
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ==========================================
# 3. Predict a customer-support intent
# ==========================================


def predict_intent(text, tokenizer, model):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a non-empty message.")

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    # Get the three categories with the highest probabilities
    top_probabilities, top_ids = torch.topk(
        probabilities,
        k=3
    )

    id2label = model.config.id2label
    top_predictions = []

    for probability, label_id in zip(
        top_probabilities.tolist(),
        top_ids.tolist()
    ):
        label = id2label.get(
            label_id,
            id2label.get(str(label_id), str(label_id))
        )

        top_predictions.append({
            "intent": label,
            "confidence": probability,
        })

    return {
        "message": text,
        "intent": top_predictions[0]["intent"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a non-empty message.")

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())

    # The trained model's configuration stores the category mapping.
    id2label = model.config.id2label
    predicted_label = id2label.get(
        predicted_id,
        id2label.get(str(predicted_id), str(predicted_id))
    )

    return {
        "message": text,
        "intent": predicted_label,
        "confidence": confidence,
    }


# ==========================================
# 4. Run example predictions
# ==========================================

def main():
    tokenizer, model = load_model()

    test_messages = [
        # Lost or stolen card
        "I lost my card.",
       "My card has been stolen.",
       "I need to block my lost card.",
       "I lost my card and need to block it.",

       # Account access
       "I forgot my passcode.",
       "I cannot log in to my account.",

       # Transfer problems
       "My bank transfer is still pending.",
       "My bank transfer has not arrived yet.",

       # Virtual card
       "My virtual card is not working.",
       "How can I get a virtual card?",
    ]

    print("\n" + "=" * 55)
    print("BANKING77 CUSTOMER SUPPORT INTENT CLASSIFIER")
    print("=" * 55)

    for message in test_messages:
        result = predict_intent(message, tokenizer, model)

        print(f"\nMessage: {result['message']}")
        print("Top predictions:")
        for prediction in result["top_predictions"]:
            print(
                f"  {prediction['intent']}: "
                f"{prediction['confidence']:.2%}"
                
            )


if __name__ == "__main__":
    main()
