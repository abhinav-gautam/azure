from dotenv import load_dotenv
import os

# import namespaces
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv("AI_SERVICE_ENDPOINT")
        ai_key = os.getenv("AI_SERVICE_KEY")
        project_name = os.getenv("PROJECT")
        deployment_name = os.getenv("DEPLOYMENT")

        # Create client using endpoint and key
        credential = AzureKeyCredential(key=ai_key)
        client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

        # Read each text file in the ads folder
        batchedDocuments = []
        ads_folder = "ads"
        files = os.listdir(ads_folder)
        for file_name in files:
            # Read the file contents
            text = open(os.path.join(ads_folder, file_name), encoding="utf8").read()
            batchedDocuments.append(text)

        # Extract entities
        operation = client.begin_recognize_custom_entities(
            documents=batchedDocuments,
            project_name=project_name,
            deployment_name=deployment_name,
        )

        document_results = operation.result()

        for doc, custom_entities_result in zip(files, document_results):
            print(doc)
            if custom_entities_result.kind == "CustomEntityRecognition":
                for entity in custom_entities_result.entities:
                    print(
                        f"\tEntity '{entity.text}' has category '{entity.category}' with confidence score of '{entity.confidence_score}'"
                    )
            elif custom_entities_result.is_error is True:
                print(
                    f"\tError with code '{custom_entities_result.error.code}' and message '{custom_entities_result.error.message}'"
                )

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
