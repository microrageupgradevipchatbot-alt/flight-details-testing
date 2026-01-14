import streamlit as st
import uuid

from complete import agent,logger
from langchain_core.messages import AIMessage
# ---------- Page config ----------
st.set_page_config(
    page_title="UpgradeVIP Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "system_sent" not in st.session_state:
    st.session_state.system_sent = False
if "chat" not in st.session_state:
    st.session_state.chat = [
("assistant", 
         "Good day. Welcome to UpgradeVIP – where seamless luxury travel is our standard.\n"
         "I'm here to ensure every detail of your journey is impeccably arranged.\n\n How may I be of service today?\n\n"
         "**Airport VIP Services** – Fast-track security, lounge access, and meet & greet  \n"
         "**Airport Transfer Services** – Chauffeur-driven transfers tailored to your schedule\n\n"
         "What may I arrange to ensure a seamless journey?"
        )]

# ---------- Header ----------
st.markdown("""
<div class="main-header">
  <h1>UpgradeVIP Chatbot</h1>
  <p>Your premium concierge for Airport VIP and Transfers.</p>
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
         "Good day. Welcome to UpgradeVIP – where seamless luxury travel is our standard.\n"
         "I'm here to ensure every detail of your journey is impeccably arranged.\n\n How may I be of service today?\n\n"
         "**Airport VIP Services** – Fast-track security, lounge access, and meet & greet  \n"
         "**Airport Transfer Services** – Chauffeur-driven transfers tailored to your schedule\n\n"
         "What may I arrange to ensure a seamless journey?"
        )
            ]
            st.session_state.system_sent = False
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    with col_b:
        transcript = "\n".join(
            [("You: " if r == "user" else "Assistant: ") + c for r, c in st.session_state.chat]
        )
        st.download_button("Download", transcript.encode("utf-8"), file_name="upgradevip_chat.txt")

    st.markdown("---")
    st.markdown("#### Tips")
    st.info("""
**Customer Assistance:**  
Ask anything about UpgradeVIP – services, contact info, airport list, or general questions.
- Note: Chatbot will not answer out of scope questions i.e capital of france etc

**Booking Flow:**  
1. Tell the bot which service you want: **airport VIP** or **transfer**.  
2. Provide your **flight number** (e.g. LY001) and **date** (MM/DD/YYYY).  
3. Choose **Arrival/Departure**(choose departure)
4. then select your **class** (Economy/Business/First).  
5. Enter **passenger** and **luggage count** (range 1-10).  
6. Pick your **preferred currency**.  
7. Select a service card by entering card no. or title. 
8. enter prefer time 
9. enter msg for steward 
10. give email.  
11. For multi-service, the bot will ask for other service you want to book.
12 Yes or no 
- if no
    - then invoice will given by bot confirm it
    - email will be send to you
- if yes
    - then repeat from step 1            
*Tip: For airport list, just ask “Show airports list”.*
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
    # Normalize Windows CRLF and turn single newlines into Markdown line breaks
    return text.replace("\r\n", "\n").replace("\n", "  \n")

def call_agent(user_text: str) -> str:
    messages_to_send = []
    messages_to_send.append({"role": "user", "content": user_text})
    
    logger.info(f"📥 Messages sending to llm: {messages_to_send}")
    logger.info(f"🔑 Session ID: {st.session_state.session_id}")
    
    try:
        result = agent.invoke(
            {"messages": messages_to_send},
            config={
                "configurable": {"thread_id": st.session_state.session_id},
                "max_iterations": 5,
            },
        )
        logger.info(f"🔄 Agent response: {result}")

        # Find latest AI message
        messages = result.get("messages", [])
        bot_reply = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                bot_reply = getattr(m, "content", None)
                break

        # Handle string or list response
        if isinstance(bot_reply, list) and bot_reply and isinstance(bot_reply[0], dict) and "text" in bot_reply[0]:
            bot_reply_text = bot_reply[0]["text"]
        else:
            bot_reply_text = extract_text_from_ai(bot_reply)

        if not bot_reply_text:
            bot_reply_text = "Sorry, I am encountering some issues. Please try again later."
            
        logger.info(f"✅ Agent invocation successful. Reply: {bot_reply_text}")
        return bot_reply_text
        
    except Exception as e:
        logger.error(f"❌ Agent invocation failed: {str(e)}", exc_info=True)
        logger.error(f"❌ Error type: {type(e).__name__}")
        return f"Sorry, I encountered an error: {str(e)}. Please try again or contact support."
# ---------- Chat history display ----------
with st.container():
    for role, content in st.session_state.chat:
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                st.markdown(md_with_linebreaks(str(content)))
            else:
                st.write(content)



# Add pending_reply to session state (put this after your other session state initializations)
if "pending_reply" not in st.session_state:
    st.session_state.pending_reply = None


# ---------- Input ----------
prompt = st.chat_input("Type your message and press Enter...")
if prompt:
    # Add user message to chat
    st.session_state.chat.append(("user", prompt))
    
    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)
    
    # Show assistant typing with spinner
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Typing..."):
            reply = call_agent(prompt)
        st.markdown(md_with_linebreaks(reply))


    # Add assistant reply to chat
    st.session_state.chat.append(("assistant", reply))