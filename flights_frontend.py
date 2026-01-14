import streamlit as st
import uuid
from langchain_core.messages import AIMessage
from flights import agent, logger
# ---------- Page config ----------
st.set_page_config(
    page_title="UpgradeVIP Flight Bot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat" not in st.session_state:
    st.session_state.chat = [
        ("assistant", 
         "UpgradeVIP Flight Bot - Welcome!\n\n"
         "Please provide your **flight number** (e.g., LY001) and **flight date** (MM/DD/YYYY format) to get started."
        )
    ]

# ---------- Header ----------
st.markdown("""
<div class="main-header">
  <h1>✈️ UpgradeVIP Flight Bot</h1>
  <p>Get your flight details instantly.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("#### Session")
    st.code(st.session_state.session_id)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New chat"):
            st.session_state.chat = [
                ("assistant", 
                 "UpgradeVIP Flight Bot - Welcome!\n\n"
                 "Please provide your **flight number** (e.g., LY001) and **flight date** (MM/DD/YYYY format) to get started."
                )
            ]
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    with col_b:
        transcript = "\n".join(
            [("You: " if r == "user" else "Bot: ") + c for r, c in st.session_state.chat]
        )
        st.download_button("Download", transcript.encode("utf-8"), file_name="flight_chat.txt")

    st.markdown("---")
    st.markdown("#### Tips")
    st.info("""
**How to use:**  
1. Provide your **flight number** (e.g., LY001)
2. Provide your **flight date** (MM/DD/YYYY format)
3. The bot will fetch your flight details
4. Choose between arrival or departure

*Example: "My flight is LY001 on 12/25/2025"*
""")

# ---------- Helpers ----------
def extract_text_from_ai(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if "text" in p:
                    parts.append(p["text"])
                elif p.get("type") == "text" and "text" in p:
                    parts.append(p["text"])
        return "\n".join([t for t in parts if t]) or ""
    return str(content or "")

def md_with_linebreaks(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "  \n")

def call_agent(user_text: str) -> str:
    messages_to_send = [{"role": "user", "content": user_text}]
    logger.info(f"📥 Messages sending to llm: {messages_to_send}")
    
    try:
        result = agent.invoke(
            {"messages": messages_to_send},
            config={
                "configurable": {"thread_id": st.session_state.session_id},
                "max_iterations": 5,
            },
        )
        logger.info(f"🔄 Agent response: {result}")

        messages = result.get("messages", [])
        bot_reply = None
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                bot_reply = getattr(m, "content", None)
                break

        if isinstance(bot_reply, list) and bot_reply and isinstance(bot_reply[0], dict) and "text" in bot_reply[0]:
            bot_reply_text = bot_reply[0]["text"]
        else:
            bot_reply_text = extract_text_from_ai(bot_reply)

        if not bot_reply_text:
            bot_reply_text = "Sorry, I couldn't process that. Please try again."
        
        logger.info(f"✅ Bot reply: {bot_reply_text}")
        return bot_reply_text
    except ValueError as ve:
        logger.error(f"❌ Validation error: {ve}")
        return "I need more information before I can help you. Let's start from the beginning - which service would you like to book: Airport VIP or Transfer?"
    except Exception as e:
        logger.error(f"❌ Agent invocation failed: {e}")
        return "Sorry, something went wrong. Please try again later."
    
# ---------- Chat history display ----------
with st.container():
    for role, content in st.session_state.chat:
        avatar = "🧑" if role == "user" else "✈️"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                st.markdown(md_with_linebreaks(str(content)))
            else:
                st.write(content)

# ---------- Input ----------
prompt = st.chat_input("Type your message and press Enter...")
if prompt:
    st.session_state.chat.append(("user", prompt))
    
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)
    
    with st.chat_message("assistant", avatar="✈️"):
        with st.spinner("Processing..."):
            reply = call_agent(prompt)
        st.markdown(md_with_linebreaks(reply))

    st.session_state.chat.append(("assistant", reply))