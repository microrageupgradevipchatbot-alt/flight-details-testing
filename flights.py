# Standard library
import logging
import os
import smtplib
import uuid
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#RAG
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#langgraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent



#============================================== .env  =========================================================

load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def setup_logging():
    CURRENT_DIR = Path(__file__).parent
    LOGS_DIR = CURRENT_DIR / 'Logs'
    LOGS_DIR.mkdir(exist_ok=True)
    LOG_FILE = LOGS_DIR / 'flights.log'
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger("Upgrade-Vip")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()

if GOOGLE_API_KEY:
    logger.info("GOOGLE_API_KEY loaded successfully.")
else:
    logger.info("GOOGLE_API_KEY not found. Please set it in your environment or .env file.")

if not os.getenv("DEV_URL"):  
    logger.warning("DEV_URL not found. Please set it in your environment or .env file.")
if not os.getenv("API_KEY"):  
    logger.warning("API_KEY not found. Please set it in your environment or .env file.")  
dev_url = os.getenv("DEV_URL")
api_key = os.getenv("API_KEY")
#===============================================Flight Functions================================================================

def get_flight_details_from_api(flight_number: str, flight_date: str) -> dict:
    logger.info(f"🛫 Calling flight details API with parameters:")
    logger.info(f"   - Flight Number: {flight_number}")
    logger.info(f"   - Flight Date: {flight_date}")
    

    endpoint = f"{dev_url}get_flight_details?flight_number={flight_number}&flightdate={flight_date}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        logger.info(f"🌐 Calling Production endpoint: {endpoint}")
        response = requests.get(endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        try:
            result = response.json()
            logger.info(f"✅ Successfully connected to flight details API (Production) : {result}")
            # Filter only required fields
            filtered_flights = []
            for flight in result.get("data", []):
                filtered_flight = {
                    "origin_iata_code": flight.get("origin_iata_code"),#"TLV",
                    "originName": flight.get("originName"),#Ben Gurion Airport TLV
                    "origin_airport": flight.get("origin_airport"),
                    "origin_time":flight.get("origin_time"),
                    "destination_iata_code": flight.get("destination_iata_code"),
                    "destinationName": flight.get("destinationName"),
                    "destination_airport": flight.get("destination_airport"),
                    "destination_time": flight.get("destination_time"),
                    "date_departure": flight.get("date_departure"),
                    "date_arrival": flight.get("date_arrival"),
                 
                }
                filtered_flights.append(filtered_flight)
            return {
                "code": result.get("code", 1),
                "message": result.get("message", "success"),
                "data": filtered_flights
            }
        except Exception as e:
            logger.error(f"❌ Invalid JSON response from Production: {e}")
            return {"error": "Invalid JSON response", "message": "API returned invalid data format"}
    except Exception as e:
        return {"error": "Request failed", "message": str(e)}  
    
def format_flight_choice_message_impl(flight_data: dict) -> str:
    logger.info(f"🚪 Inside Formatting flight choice message function")
    if not flight_data or "data" not in flight_data or not flight_data["data"]:
        return "Sorry, flight details could not be retrieved make sure your flight number is correct and date is in MM/DD/YYYY format like 10/29/2025."
    info = flight_data["data"][0]
    msg = (
        f"Departure:\n"
        f"{info.get('origin_iata_code', '')}   {info.get('origin_time', '')}\n"
        f"{info.get('date_departure', '')}\n"
        f"{info.get('originName', '')}\n\n"
        f"Arrival:\n"
        f"{info.get('destination_iata_code', '')}   {info.get('destination_time', '')}\n"
        f"{info.get('date_arrival', '')}\n"
        f"{info.get('destinationName', '')}\n\n"
        f"Which do you want to choose: arrival or departure?"
    )
    logger.info(f"✈️ Formatted flight choice message: {msg}")
    return msg

#==============================================flight tools=========================================================


@tool
def flight_details_tool(flight_number: str, flight_date: str):
    """
    Call external API to fetch flight details.

    Parameters:
      flight_number (str): The user's flight number (e.g., 'LY001').
      flight_date (str): The user's flight date (e.g., '11/31/2025' in mm/dd/yyyy format).

    Call this tool as soon as both flight number and flight date are provided by the user.
    """
    return get_flight_details_from_api(flight_number, flight_date)

@tool
def format_flight_choice_tool(flight_data: dict) -> str:
    """Format flight details into a user-friendly message after calling flight_details_tool."""
    # Unwrap if the LLM nested it or sent a single object instead of {data:[...]}
    if isinstance(flight_data, dict) and "flight_details_tool_response" in flight_data:
        flight_data = flight_data["flight_details_tool_response"]
    # If the LLM sent a single-flight dict, wrap it as {data: [dict]}
    if isinstance(flight_data, dict) and "data" not in flight_data and "flight_no" in flight_data:
        flight_data = {"data": [flight_data]}
    return format_flight_choice_message_impl(flight_data)


tools = [   flight_details_tool,
            format_flight_choice_tool
        ]

#================================================ LLM_AND_REACT_AGENT_Setup =========================================================================


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=4,
    google_api_key=GOOGLE_API_KEY,
)

memory = InMemorySaver()
SYSTEM_PROMPT = """
when user gives flight number and date, 
call flight_details_tool(flight_number: str, flight_date: str) 
"""

agent = create_react_agent(
    llm,
    tools=tools,
    checkpointer=memory,  # save convo in RAM
    prompt=SYSTEM_PROMPT,
  #  pre_model_hook=trim_messages,  # run trimming before each model call     
)

#=============================================== PROMPT =======================================================================


#=============================================== FastAPI setup =======================================================================

# class MessageRequest(BaseModel):
#     message: str
#     session: Optional[str] = None

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True,
#     allow_methods=["*"], allow_headers=["*"],
# )
# #==========================root endpoint to check if server is running==========================

# @app.get("/")
# def root():
#     return {"okk": True}

# #========================================  MAIN  ================================================

# @app.post("/message")
# async def message_endpoint(req: MessageRequest):
#     user_input = req.message.strip()
#     session_id = req.session or str(uuid.uuid4())
#     logger.info(f"💬 User({session_id}): {user_input}")

#     # Only send the user message; agent will handle session memory
#     messages_to_send = [{"role": "user", "content": user_input}]
#     logger.info(f"📥 Messages sending to llm: {messages_to_send}")

#     result = await agent.ainvoke(
#         {"messages": messages_to_send},
#         config={
#             "configurable": {"thread_id": session_id},
#             "max_iterations": 5
#         }
#     )
#     logger.info(f"🔄 Agent response: {result}")

#     # Extract the latest AI message
#     messages = result.get("messages", [])
#     bot_reply = None
#     for m in reversed(messages):
#         if isinstance(m, AIMessage):
#             bot_reply = getattr(m, "content", None)
#             break
#     logger.info(f"🤖 Bot({session_id}): {bot_reply}")
#     return {"bot_reply": bot_reply, "session": session_id}
    
#========================================= Main for local testing ============================================================
import asyncio

async def main():
    print("UpgradeVIP Flight Bot (type 'exit' to quit)")
    session_id = str(uuid.uuid4())
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        messages_to_send = [{"role": "user", "content": user_input}]
        logger.info(f"📥 Messages sending to llm: {messages_to_send}")

        try:
            result = await agent.ainvoke(
                {"messages": messages_to_send},
                config={
                    "configurable": {"thread_id": session_id},
                    "max_iterations": 5
                }
            )
            logger.info(f"🔄 Agent response: {result}")

            messages = result.get("messages", [])
            bot_reply = None
            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    bot_reply = getattr(m, "content", None)
                    break
            logger.info(f"\n\n🤖 Bot({session_id}): {bot_reply}")
             # Fix: Extract text if bot_reply is a list of dicts
            if isinstance(bot_reply, list) and bot_reply and isinstance(bot_reply[0], dict) and "text" in bot_reply[0]:
                print(f"🤖 Bot: {bot_reply[0]['text']}")
            else:
                print(f"🤖 Bot: {bot_reply}")

        except Exception as e:
            logger.error(f"❌ Error during agent invocation: {e}")
            print("🤖 Bot: Sorry, something went wrong. Please try again later.")

if __name__ == "__main__":
    asyncio.run(main())
