import azure.functions as func
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Initialisation du modèle NER...")
model_name ="Jean-Baptiste/camembert-ner-with-dates"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
logger.info("Modèle NER initialisé avec succès.")

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
        
        result = nlp(query)

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
