from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt


# -----------------------------
# 1. TOOLS
# -----------------------------

@tool
def read_email(email_id: str) -> str:
    """Read an email by ID"""
    return f"""
From: boss@company.com
Subject: Meeting Reminder

Please attend the meeting at 3 PM.
Email ID: {email_id}
"""


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email"""

    print("\n📧 EMAIL SENT")
    print("To:", recipient)
    print("Subject:", subject)
    print("Body:", body)

    return f"Email sent to {recipient}"


# -----------------------------
# 2. GROQ MODEL
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# -----------------------------
# 3. CHECKPOINTER (required)
# -----------------------------

checkpointer = InMemorySaver()

thread_config = {
    "configurable": {
        "thread_id": "email-thread-1"
    }
}


# -----------------------------
# 4. HUMAN-IN-THE-LOOP MIDDLEWARE
# -----------------------------

middleware = [
    HumanInTheLoopMiddleware(
        interrupt_on={
            "send_email": {
                "allowed_decisions": ["approve", "edit", "reject"]
            },
            "read_email": False   # no approval needed
        }
    )
]


# -----------------------------
# 5. CREATE AGENT
# -----------------------------

agent = create_agent(
    model=llm,
    tools=[read_email, send_email],
    checkpointer=checkpointer,
    middleware=middleware,
)


# -----------------------------
# 6. LOOP WRAPPER FUNCTION
# -----------------------------

def run_agent_loop(user_input):

    state = {
        "messages": [("user", user_input)]
    }

    while True:

        result = agent.invoke(state, config=thread_config)

        # ---------------------------
        # DONE CASE
        # ---------------------------
        if "__interrupt__" not in result:
            print("\n✅ FINAL RESULT:")
            print(result)
            break

        interrupt = result["__interrupt__"][0]

        # 🔥 SAFE: do NOT assume structure
        payload = getattr(interrupt, "value", interrupt)

        print("\n⚠️ HUMAN APPROVAL REQUIRED")

        # ALWAYS inspect first time
        print("\nDEBUG INTERRUPT PAYLOAD:")
        print(payload)

        # ---------------------------
        # USER DECISION
        # ---------------------------
        decision = input("\napprove / edit / reject: ").strip().lower()

        if decision == "approve":
            state = Command(resume={"type": "approve"})

        elif decision == "reject":
            state = Command(resume={"type": "reject"})

        elif decision == "edit":

            new_body = input("New email body: ")

            # try safe fallback access
            args = payload.get("args", {}) if isinstance(payload, dict) else {}

            state = Command(
                resume={
                    "type": "edit",
                    "args": {
                        **args,
                        "body": new_body
                    }
                }
            )

        else:
            print("Invalid → rejecting")
            state = Command(resume={"type": "reject"})
# 7. RUN SYSTEM
# -----------------------------

if __name__ == "__main__":

    user_query = input("Enter your request: ")

    run_agent_loop(user_query)