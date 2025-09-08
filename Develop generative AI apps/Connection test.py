from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

try:
    # Update project URL below
    project_endpoint = ""
    project_client = AIProjectClient(
        endpoint=project_endpoint, credential=DefaultAzureCredential()
    )

    connections = project_client.connections

    print("List of connections")
    for connection in connections.list():
        print(f"{connection.name} - {connection.type}")
except Exception as ex:
    print(ex)
