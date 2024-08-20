import json
import re
import string
from datetime import datetime

from transformers import AutoTokenizer

TOKENIZER_MODEL = "afbudiman/distilled-indobert-classification"
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

class Helper:
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                # Format datetime object to a string
                return obj.isoformat()
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)
        
    def CleanText(text):
        # Lowercase the text
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user @ references and '#' from text
        text = re.sub(r'\@\w+|\#', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_fn(examples):
        texts = examples['clean content']
        result = TOKENIZER(texts, padding='max_length', truncation=True, max_length=512)
        return result
        