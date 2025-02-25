import azure.functions as func
import logging
'''from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json'''

import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import TFAutoModelForSequenceClassification
import numpy as np
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
'''
logger.info("Initialisation du modèle NER...")
model_name ="Jean-Baptiste/camembert-ner-with-dates"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
logger.info("Modèle NER initialisé avec succès.")
'''
logger.info("Initialisation du modèle NER...")
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
logger.info("Modèle NER initialisé avec succès.")

def predict_sentiment(comment):
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=1).tolist()[0]
        labels = ["1 étoile", "2 étoiles", "3 étoiles", "4 étoiles", "5 étoiles",]
        sentiment = labels[np.argmax(scores)]
        return sentiment


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        query = req_body.get('text')

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "No query provided in request body"}),
                mimetype="application/json",
                status_code=400
            )
        
        result = predict_sentiment(query)

        return func.HttpResponse(
            json.dumps({"response": result}),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
