from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_DIR = "models/roberta"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model.push_to_hub(
    "Thong419/fake-news-roberta"
)

tokenizer.push_to_hub(
    "Thong419/fake-news-roberta"
)

print("Upload completed")