from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Create Groq model
model = ChatGroq(
   model="openai/gpt-oss-120b",
    temperature=0
)

# 2. Chat history (memory)
messages = [
    SystemMessage(content="You are a helpful assistant.")
]

# 3. Infinite chat loop
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    # 4. Add user message to history
    messages.append(
        HumanMessage(content=user_input)
    )

    # 5. Get AI response using FULL history
    response = model.invoke(messages)

    ai_message = response.content

    print("\nAI:", ai_message)

    # 6. Save AI response in memory
    messages.append(
        AIMessage(content=ai_message)
    )