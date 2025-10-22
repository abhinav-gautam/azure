from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


def main():
    try:
        # Get configuration settings
        load_dotenv()
        ai_endpoint = os.getenv("AI_SERVICE_ENDPOINT")
        ai_key = os.getenv("AI_SERVICE_KEY")

        # Create client using endpoint and key
        credential = AzureKeyCredential(ai_key)
        ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

        # Analyse each text file in the reviews folder
        reviews_folder = "reviews"
        for file_name in os.listdir(reviews_folder):
            # Read the file contents
            print("\n-------------\n", file_name)
            text = open(
                os.path.join(reviews_folder, file_name), encoding="utf-8"
            ).read()
            print("\n", text)

            # Get language
            detectedLanguage = ai_client.detect_language(documents=[text])[0]
            print(f"\nLanguage: {detectedLanguage.primary_language.name}")

            # Get sentiment
            sentimentAnalysis = ai_client.analyze_sentiment(documents=[text])[0]
            print(f"\nSentiment: {sentimentAnalysis.sentiment}")

            # Get key phrase
            keyPhrases = ai_client.extract_key_phrases(documents=[text])[0]
            print(f"\nKey phrases: {keyPhrases.key_phrases}")

            # Get entities
            entities = ai_client.recognize_entities(documents=[text])[0].entities
            print("\nEntities:")
            for entity in entities:
                print(f"{entity.text} - {entity.category}")

            # Get linked entities
            linkedEntities = ai_client.recognize_linked_entities(documents=[text])[
                0
            ].entities
            print("\nLinked entities:")
            for entity in linkedEntities:
                print(f"{entity.name} - {entity.url}")

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
