import os
import asyncio
from pathlib import Path

from azure.identity.aio import AzureCliCredential
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from pydantic import Field
from typing import Annotated


async def main():
    # Clear the console
    os.system("cls" if os.name == "nt" else "clear")

    # Load the expenses data file
    script_dir = Path(__file__).parent
    file_path = script_dir / "data.txt"
    with file_path.open("r") as file:
        data = file.read() + "\n"

    # Ask for a prompt
    user_prompt = input(
        f"Here is the expense data in your file:\n\n{data}\n\nWhat would you like me to do with it?\n\n"
    )

    # Run the async agent code
    await process_expense_data(user_prompt, data)


async def process_expense_data(prompt, expenses_data):
    # Create a chat agent
    async with (
        AzureCliCredential() as cred,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=cred),
            name="expenses_agent",
            instructions="""You are an AI assistant for expense claim submission.
                        When a user submits expenses data and requests an expense claim, use the plug-in function to send an email to expenses@contoso.com with the subject 'Expense Claim`and a body that contains itemized expenses with a total.
                        Then confirm to the user that you've done so.""",
            tools=send_email,
        ) as agent,
    ):

        # Use the agent to process the expenses data
        try:
            # Add the input prompt to a list of messages to be submitted
            prompt_messages = [f"{prompt}: {expenses_data}"]

            # Invoke the agent for the specified thread with the messages
            response = await agent.run(prompt_messages)

            # Display the response
            print(f"\n# Agent:\n{response}")

        except Exception as e:
            print(e)


# Create a tool function for the email functionality
def send_email(
    to: Annotated[str, Field(description="Who to send the email to")],
    subject: Annotated[str, Field(description="The subject of the email")],
    body: Annotated[str, Field(description="The text body of the email")],
):
    print("\nTo:", to)
    print("Subject:", subject)
    print(body)


if __name__ == "__main__":
    asyncio.run(main())
