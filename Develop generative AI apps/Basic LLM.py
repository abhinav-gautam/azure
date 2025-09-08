from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

try:
    project_endpoint = ""
    project_client = AIProjectClient(
        endpoint=project_endpoint, credential=DefaultAzureCredential()
    )

    connections = project_client.connections

    print("List of connections")
    for connection in connections.list():
        print(f"{connection.name} - {connection.type}")

    # Get a chat client
    chat_client = project_client.get_openai_client(api_version="2024-10-21")

    # Get a chat completion based on a user-provided prompt
    user_prompt = input("Enter a question:")

    response = chat_client.chat.completions.create(
        model="Ministral-3B",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_prompt},
        ],
    )
    print(response.choices[0].message.content)
except Exception as ex:
    print(ex)
