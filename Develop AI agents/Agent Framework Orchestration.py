from typing import List, cast
import asyncio
from azure.identity import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import SequentialBuilder, WorkflowOutputEvent, ChatMessage, Role


async def main():
    # Agent instructions
    summarizer_instructions = """
    Summarize the customer's feedback in one short sentence. Keep it neutral and concise.
    Example output:
    App crashes during photo upload.
    User praises dark mode feature.
    """

    classifier_instructions = """
    Classify the feedback as one of the following: Positive, Negative, or Feature request.
    """

    action_instructions = """
    Based on the summary and classification, suggest the next action in one short sentence.
    Example output:
    Escalate as a high-priority bug for the mobile team.
    Log as positive feedback to share with design and marketing.
    Log as enhancement request for product backlog.
    """

    # Create the chat client
    credential = AzureCliCredential()
    async with AzureAIAgentClient(async_credential=credential) as chat_client:
        # Create agents
        summarizer_agent = chat_client.create_agent(
            instructions=summarizer_instructions, name="summarizer"
        )

        classifier_agent = chat_client.create_agent(
            instructions=classifier_instructions, name="classifier"
        )

        action_agent = chat_client.create_agent(
            instructions=action_instructions, name="action"
        )

        # Initialize the current feedback
        feedback = """
            I use the dashboard every day to monitor metrics, and it works well overall. 
            But when I'm working late at night, the bright screen is really harsh on my eyes. 
            If you added a dark mode option, it would make the experience much more comfortable.
            """

        # Build sequential orchestration
        workflow = (
            SequentialBuilder()
            .participants([summarizer_agent, classifier_agent, action_agent])
            .build()
        )

        # Run and collect outputs
        outputs: List[List[ChatMessage]] = []
        async for event in workflow.run_stream(f"Customer feedback: {feedback}"):
            if isinstance(event, WorkflowOutputEvent):
                outputs.append(cast(List[ChatMessage], event.data))

        # Display outputs
        if outputs:
            for i, message in enumerate(outputs[-1], start=1):
                name = message.author_name or (
                    "assistant" if message.role == Role.ASSISTANT else "user"
                )
                print(f"{'-' * 60}\n{i:02d} [{name}]\n{message.text}")


if __name__ == "__main__":
    asyncio.run(main())
