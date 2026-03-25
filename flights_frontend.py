import streamlit as st
import uuid
from langchain_core.messages import AIMessage
from flights import agent, logger
import streamlit.components.v1 as components
# ---------- Page config ----------
st.set_page_config(
    page_title="UpgradeVIP Chatbot",
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
         "Good day. Welcome to UpgradeVIP – where seamless luxury travel is our standard.\n"
         "I'm here to ensure every detail of your journey is impeccably arranged.\n\n How may I be of service today?\n\n"
         "**Airport VIP Services** – Fast-track security, lounge access, and meet & greet  \n"
         "**Airport Transfer Services** – Chauffeur-driven transfers tailored to your schedule\n\n"
         "What may I arrange to ensure a seamless journey?"
        )
    ]

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
    except Exception as e:
        logger.error(f"❌ Agent invocation failed: {e}")
        return f"Sorry, something went wrong. Please try again later."
    
# ---------- Chat history display ----------
with st.container():
    for role, content in st.session_state.chat:
        avatar = "🧑" if role == "user" else "✈️"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                st.markdown(md_with_linebreaks(str(content)), unsafe_allow_html=True)
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
        st.markdown(md_with_linebreaks(reply), unsafe_allow_html=True)

    st.session_state.chat.append(("assistant", reply))

# ---------- Floating Contact Buttons ----------
st.markdown("""
<style>
.floating-buttons {
    position: fixed;
    bottom: 140px;
    right: 30px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    z-index: 9999;
}

.floating-buttons a {
    text-decoration: none;
}

.floating-btn {
    width: 70px;
    height: 70px;
    background: linear-gradient(145deg, #25D366 60%, #1ebe57 100%);
    color: white;
    border-radius: 50%;
    font-size: 14px;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25), 0 1.5px 4px rgba(0,0,0,0.18);
    border: 2px solid #fff;
    transition: box-shadow 0.2s, filter 0.2s;
    padding: 0;
    word-break: break-word;
    cursor: pointer;
    user-select: none;
}

.floating-btn.live-btn {
    background: linear-gradient(145deg, #007bff 60%, #0056b3 100%);
}

.floating-btn:hover {
    box-shadow: 0 10px 24px rgba(0,0,0,0.35), 0 2px 8px rgba(0,0,0,0.22);
    filter: brightness(1.08);
}
</style>
<div class="floating-buttons">
    <a href="https://wa.me/447414246103" target="_blank">
        <div class="floating-btn">WhatsApp</div>
    </a>
    <a href="#" onclick="window.Tawk_API && window.Tawk_API.maximize(); return false;">
    <div class="floating-btn live-btn">Live Agent</div>
</a>
</div>
""", unsafe_allow_html=True)

components.html("""
<!--Start of Tawk.to Script-->
<script type="text/javascript">
var Tawk_API=Tawk_API||{};
Tawk_API.onLoad = function() {
    window.parent.Tawk_API = Tawk_API;  // 👈 THIS IS THE FIX
};

(function(){
var s1=document.createElement("script"),
s0=document.getElementsByTagName("script")[0];
s1.async=true;
s1.src='https://embed.tawk.to/69c3da4935e8d61c3a87571f/1jkigpbv8';
s1.charset='UTF-8';
s1.setAttribute('crossorigin','*');
s0.parentNode.insertBefore(s1,s0);
})();
</script>
<!--End of Tawk.to Script-->
""", height=0)
