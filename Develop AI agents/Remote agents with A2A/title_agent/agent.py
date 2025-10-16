from typing import List
import os
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import Agent, ListSortOrder, MessageRole


class TitleAgent:
    def __init__(self):
        # Create the agents client
        self.client = AgentsClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True,
            ),
        )

        self.agent: Agent | None = None

    async def create_agent(self) -> Agent:
        if self.agent:
            return self.agent

        # Create the title agent
        self.agent = self.client.create_agent(
            model=os.environ["MODEL_DEPLOYMENT"],
            name="title-agent",
            instructions="""
                You are a helpful writing assistant.
                Given a topic the user wants to write about, suggest a single clear and catchy blog post title.
            """,
        )

        return self.agent

    async def run_conversation(self, user_message: str) -> List[str]:
        if not self.agent:
            await self.create_agent()

        # Create a thread for the chat session
        thread = self.client.threads.create()

        # Send user message
        self.client.messages.create(
            thread_id=thread.id, role=MessageRole.USER, content=user_message
        )

        # Create and run the agent
        run = self.client.runs.create_and_process(
            thread_id=thread.id, agent_id=self.agent.id
        )

        if run.status == "failed":
            print(f"Title Agent: Run failed = {run.last_error}")
            return [f"Error: {run.last_error}"]

        # Get response messages
        messages = self.client.messages.list(
            thread_id=thread.id, order=ListSortOrder.DESCENDING
        )
        responses = []
        for msg in messages:
            if msg.role == MessageRole.AGENT and msg.text_messages:
                for text_msg in msg.text_messages:
                    responses.append(text_msg.text.value)
                break

        return responses if responses else ["No response received"]


async def create_foundry_title_agent() -> TitleAgent:
    agent = TitleAgent()
    await agent.create_agent()
    return agent
