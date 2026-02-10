# Standard library
import logging
import os
import smtplib
import uuid
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
# Third-party
import requests
import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#RAG
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader,UnstructuredMarkdownLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#langgraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent



#============================================== .env  =========================================================
# load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
def setup_paths():
    current_dir = Path(__file__).parent
    data_dir = current_dir / "DATA"
    db_dir = data_dir / "DB"
    doc_dir = data_dir / "Docs"
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    db_dir.mkdir(exist_ok=True)
    doc_dir.mkdir(exist_ok=True)

    return current_dir, db_dir, doc_dir
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

CURRENT_DIR, DB_DIR, DOC_DIR = setup_paths()
logger = setup_logging()

if GOOGLE_API_KEY:
    logger.info("GOOGLE_API_KEY loaded successfully.")
else:
    logger.info("GOOGLE_API_KEY not found. Please set it in your environment or .env file.")

if not st.secrets.get("DEV_URL"):  
    logger.warning("DEV_URL not found. Please set it in your environment or .env file.")
if not st.secrets.get("API_KEY"):  
    logger.warning("API_KEY not found. Please set it in your environment or .env file.")  
dev_url = st.secrets.get("DEV_URL")
api_key = st.secrets.get("API_KEY")
# if not os.getenv("DEV_URL"):  
#     logger.warning("DEV_URL not found. Please set it in your environment or .env file.")
# if not os.getenv("API_KEY"):  
#     logger.warning("API_KEY not found. Please set it in your environment or .env file.")  
# dev_url = os.getenv("DEV_URL")
# api_key = os.getenv("API_KEY")
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

#============================================== VIP Services Functions ==========================================================
def get_vip_services(airport_id: str, travel_type: str, currency: str, service_id: str = None) -> dict:
    # Validate currency parameter
    if not currency or currency.strip() == "":
        logger.error("❌ Currency parameter is required for VIP services API")
        return {"error": "Currency not specified", "message": "Please select a currency preference first."}
    
    logger.info(f"🛎️ Calling VIP services API - airport_id={airport_id}, travel_type={travel_type}, currency={currency}")
    

    endpoint = f"{dev_url}vip_services?airport_id={airport_id}&travel_type={travel_type}&currency={currency}&service_id={service_id}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        logger.info(f"🌐 Calling Production endpoint: {endpoint}")
        response = requests.get(endpoint, headers=headers, timeout=8)
        response.raise_for_status()
        try:
            result = response.json()
            logger.info(f"✅ Successfully connected to VIP services API (Production){result}")
            # Filter out unnecessary fields from each service card
            filtered_services = []
            for service in result.get("data", []):
                filtered_service = {
                    "title": service.get("title"),
                    "price": service.get("price"),
                    "currency": service.get("currency"),
                    "airport_name": service.get("airport_name"),
                    "words": service.get("words"),
                    "adults_1": service.get("adults_1"),
                    "adults_2": service.get("adults_2"),
                    "adults_3": service.get("adults_3"),
                    "adults_4": service.get("adults_4"),
                    "adults_5": service.get("adults_5"),
                    "adults_6": service.get("adults_6"),
                    "adults_7": service.get("adults_7"),
                    "adults_8": service.get("adults_8"),
                    "adults_9": service.get("adults_9"),
                    "adults_10": service.get("adults_10"),
                    "refund_text": service.get("refund_text"),
                    "price_mutiple": service.get("price_mutiple"),
                    "meeting_point": service.get("meeting_point"),
                    "cancellation_policy": service.get("cancellation_policy"),
                    "supplierservicename": service.get("supplierservicename"),
                    # Add any other required fields here
                }
                filtered_services.append(filtered_service)
            return {
                "code": result.get("code", 1),
                "message": result.get("message", "success"),
                "data": filtered_services
            }
        except Exception as e:
            logger.error(f"❌ Invalid JSON response from Production: {e}")
            return "Sorry Invalid JSON response from VIP services API"
    except Exception as e:
        logger.error(f"❌ VIP services API request failed: {e}")
        return "Sorry VIP services API request failed"

def format_vip_services_message(vip_data, passenger_count, preferred_currency):
    logger.info(f"🚪 Inside Formatting vip_services function")
    services = vip_data.get("data", []) if vip_data else []
    if not services:
        return "Sorry, no VIP services are available for your selected flight and requirements."
    
    try:
        passenger_count_int = int(passenger_count) if passenger_count is not None else 0
        if passenger_count_int <= 0:
            return "Invalid passenger count. Please provide a valid number of adults (1-10)."
        if passenger_count_int > 10:
            return "Sorry, we cannot accommodate more than 10 adults. Please enter a number between 1 and 10."
    except (ValueError, TypeError):
        return "Invalid passenger count. Please provide a valid number of adults (1-10)."
    
    currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = currency_symbols.get(preferred_currency, "$")
    
    msg = "Available VIP Services:\n\n"
    for i, service in enumerate(services, 1):
        title = service.get("title", "VIP Service")
        price_key = f"adults_{passenger_count}"
        price = service.get(price_key, service.get("price", 0))
        refund_text = service.get("refund_text", "")
        words = service.get("words", "")
        
        msg += f"{i}. Title: **{title}**\n"
        msg += f"   PRICE: {symbol}{price}\n\n"
        
        msg += "**Cancellations and modifications:**\n"
        if refund_text:
            msg += f"  {refund_text}\n\n"
        else:
            msg += "   Not specified\n\n"
        
        msg += "**Description:**\n"
        if words:
            details = [detail.strip() for detail in words.split(",")]
            for detail in details:
                if detail:
                    msg += f"  {detail}\n"
        else:
            msg += "   Not specified\n"
        
        msg += "\n" + "="*50 + "\n\n"
    logger.info(f"✈️ Formatted VIP services message: {msg}")
    return msg

#======================================invoice====================================
def build_single_invoice_html(
    title: str,
    service_type: str,
    service_name: str,
    price: float,
    currency: str,
    email: str,
    description: str,
    refund_policy: str
) -> str:
    logger.info("🚪 Inside build_single_invoice_html")
    currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = currency_symbols.get(currency, "$")
    service_type_display = "Airport VIP" if service_type == "vip" else "Airport Transfer"
    invoice = f"""
    <h2><b>{title} - {service_type_display}</b></h2>
    <br>
    <b>Service:</b> {service_name}
    <br><b>Price:</b> {symbol}{price} {currency}
    <br><b>Email:</b> {email}
    <br><b>Description:</b> {description}
    <br><b>Refund Policy:</b> {refund_policy}
    """
    logger.info(f"🎫 Single Invoice Output:\n{invoice}")
    return invoice

def build_combined_invoice_html(
    title: str,
    primary_service_type: str,
    primary_service_name: str,
    primary_price: float,
    primary_currency: str,
    primary_description: str,
    primary_refund_policy: str,
    secondary_service_type: str,
    secondary_service_name: str,
    secondary_price: float,
    secondary_currency: str,
    secondary_description: str,
    secondary_refund_policy: str,
    email: str
) -> str:
    logger.info("🚪 Inside build_combined_invoice_html")
    currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    primary_symbol = currency_symbols.get(primary_currency, "$")
    secondary_symbol = currency_symbols.get(secondary_currency, "$")
    primary_type_display = "Airport VIP" if primary_service_type == "vip" else "Airport Transfer"
    secondary_type_display = "Airport VIP" if secondary_service_type == "vip" else "Airport Transfer"

    if primary_currency == secondary_currency:
        total = primary_price + secondary_price
        total_line = f"<b>Total Price:</b> {primary_symbol}{total} {primary_currency}"
    else:
        total_line = (
            f"<b>Total Price:</b><br>"
            f"Primary: {primary_symbol}{primary_price} {primary_currency}<br>"
            f"Secondary: {secondary_symbol}{secondary_price} {secondary_currency}"
        )

    invoice = f"""
    <h2><b>{title} - Combined Booking</b></h2>
    <br>
    <b>Primary Service:</b> {primary_service_name} ({primary_type_display})<br>
    <b>Price:</b> {primary_symbol}{primary_price} {primary_currency}<br>
    <b>Description:</b> {primary_description}<br>
    <b>Refund Policy:</b> {primary_refund_policy}
    <hr>
    <b>Secondary Service:</b> {secondary_service_name} ({secondary_type_display})<br>
    <b>Price:</b> {secondary_symbol}{secondary_price} {secondary_currency}<br>
    <b>Description:</b> {secondary_description}<br>
    <b>Refund Policy:</b> {secondary_refund_policy}
    <hr>
    {total_line}
    <br>
    <b>Email:</b> {email}
    """
    logger.info(f"🎫 Combined Invoice Output:\n{invoice}")
    return invoice

#====================transfer service functions=======================

def get_transport_services(airport_id: str, currency: str) -> dict:
    # Validate currency parameter
    if not currency or currency.strip() == "":
        logger.error("❌ Currency parameter is required for transport services API")
        return {"error": "Currency not specified", "message": "Please select a currency preference first."}
    
    logger.info(f"🚗 Calling Transport services API - airport_id={airport_id}, currency={currency}")
    

    
    endpoint = f"{dev_url}get_vehicles?is_arrival=1&airport_id={airport_id}&currency={currency}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        logger.info(f"🌐 Calling Production endpoint: {endpoint}")
        response = requests.get(endpoint, headers=headers, timeout=8)
        response.raise_for_status()
        try:
            result = response.json()
            logger.info(f"✅ Successfully connected to transport services API (Production:{result})")
            # Filter out unnecessary fields from each vehicle card
            filtered_vehicles = []
            for vehicle in result.get("data", []):
                filtered_vehicle = {
                    "name": vehicle.get("name"),
                    "price": vehicle.get("price"),
                    "ten": vehicle.get("ten"),
                    "thirty": vehicle.get("thirty"),
                    "plus": vehicle.get("plus"),
                    "currency": vehicle.get("currency"),
                    "words": vehicle.get("words"),
                    "capacity": vehicle.get("capacity"),
                    "price_mutiple": vehicle.get("price_mutiple"),
                    "org_price": vehicle.get("org_price"),
                }
                filtered_vehicles.append(filtered_vehicle)
            return {
                "code": result.get("code", 1),
                "message": result.get("message", "success"),
                "data": filtered_vehicles
            }
        except Exception as e:
            logger.error(f"❌ Invalid JSON response from Production: {e}")
            return "Invalid Json response "
    except Exception as e:
        logger.error(f"❌ Transport services API request failed: {e}")
        return "Sorry API request for vehicles failed "
def format_transport_services_message(transport_data,  preferred_currency):
    logger.info(f"🚪 Inside Formatting transport services message function")
    vehicles = transport_data.get("data", []) if transport_data else []
    if not vehicles:
        return "Sorry, no transport services are available for your selected flight and requirements."

    
    currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = currency_symbols.get(preferred_currency, "$")

    msg = "Available Transport Services:\n\n"

    for i, vehicle in enumerate(vehicles, 1):
        
        name = vehicle.get("name", "Transport Service")
        price = vehicle.get("price", 0)
        msg += f"{i}. **Title:** {name}\n"
        msg += f"   **Price:** {symbol}{price}\n"
        msg += f"**Description:**\n"
        words = vehicle.get("words", "")
        if words:
            details = [detail.strip() for detail in words.split(",")]
            for detail in details:
                if detail:
                    msg += f"{detail}\n"
        else:
            msg += "Not specified\n"
        msg += "\n" + "="*50 + "\n\n"

    logger.info(f"🚗 Formatted transport services message: {msg}")
    return msg

#================================== Email Function ==============================
def send_email(to_email, subject, message):
    logger.info(f"🚪 Inside send email function")
    
    # Replace with your SMTP server details
    smtp_server = st.secrets["SMTP_SERVER"]
    smtp_port = int(st.secrets["SMTP_PORT"])
    smtp_user = st.secrets["SMTP_USER"]
    smtp_pass = st.secrets["SMTP_PASS"]
    # smtp_server = os.getenv("SMTP_SERVER")
    # smtp_port = int(os.getenv("SMTP_PORT", "587"))
    # smtp_user = os.getenv("SMTP_USER")
    # smtp_pass = os.getenv("SMTP_PASS")
    msg = MIMEText(message,'html')
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [to_email], msg.as_string())
        server.quit()
        logger.info(f"✔️🎊Email sent successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

#=============================airports list function===========================
def get_airports_from_api() -> dict:
    """
    Fetch airports list from external API and return only id and airport_name for each entry.
    """
    endpoint = f"{dev_url}get_airports"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        logger.info(f"🌐 Calling airports endpoint: {endpoint}")
        response = requests.get(endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        try:
            result = response.json()
            logger.info(f"🔗 Successfully connected to airports API: {result}")

            # Filter only id and airport_name for each airport
            filtered_airports = []
            for airport in result.get("data", []):
                filtered_airports.append({
                    "id": airport.get("id", ""),
                    "airport_name": airport.get("airport_name", "")
                })
            
            logger.info(f"✂️ After trimming : {filtered_airports}")
            return {
                "code": result.get("code", 1),
                "message": result.get("message", "success"),
                "data": filtered_airports
            }
        except Exception as e:
            logger.error(f"❌ Invalid JSON response from airports API: {e}")
            return {"error": "Invalid JSON response", "message": "API returned invalid data format"}
    except Exception as e:
        logger.error(f"❌ Airports API request failed: {e}")
        return {"error": "Request failed", "message": str(e)}
def format_airports_message(airports_data: dict) -> str:
    logger.info(f"🚪 Inside Formatting airports message function")
    if not airports_data or "data" not in airports_data or not airports_data["data"]:
        return "Sorry, airport list could not be retrieved."
    return "\n".join(f"{i+1}) {a['airport_name']}" for i, a in enumerate(airports_data["data"]))


#==============================RAG functions===============
def load_documents(folder_path):
    documents = []
    
    # Check if folder exists and has files
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"⚠️ Documents folder does not exist: {folder_path}")
    
    files = os.listdir(folder_path)
    if not files:
        raise ValueError(f"⚠️ No files found in documents folder: {folder_path}")
    
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        logger.info(f"full filename {file_path}")

        if filename.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
            logger.info(f"loaded {filename}")
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            logger.info(f"loaded {filename}")
        else:
            logger.warning(f"unsupported file type : {filename}")
            continue

        documents.extend(loader.load())
    
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=450
        )
    docs = text_splitter.split_documents(documents)
    logger.info(f"Created {len(docs)} chunks from {len(documents)} documents.")
    return docs
def create_faiss_db(text_chunks):
    '''Index Documents (store embeddings) in FAISS vector store'''
    embeddings = get_gemini_embeddings()
    logger.info(f"✅ Indexing {len(text_chunks)} chunks into FAISS vector store...")
    faiss_db = FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )
    # Save to disk
    faiss_db.save_local(str(DB_DIR))
    logger.info("✅ FAISS vector store created and saved successfully.")
    return faiss_db

def load_vector_store():
    """Load the FAISS vector store from disk."""
    try:
        embeddings = get_gemini_embeddings()
        faiss_db = FAISS.load_local(
            str(DB_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("✅ FAISS vector store loaded from disk.")
        return faiss_db
    except Exception as e:
        logger.error(f"❌ Failed to load vector store: {e}")
        raise
def create_vector_store():
    """Create the Chroma vector store if not present, else load it."""
  
    documents = load_documents(DOC_DIR)
    text_chunks = create_chunks(documents)
    faiss_db = create_faiss_db(text_chunks)

    return faiss_db

def checking_vector_store():
    '''Check if vector store exists, if not create it'''
    
    # Check for FAISS index file
    faiss_index_file = DB_DIR / "index.faiss"
    if faiss_index_file.exists():
        logger.info("✅ Existing FAISS vector store found. Loading it...")
        faiss_db = load_vector_store()
    else:
        logger.info("🛑 No existing vector store found. Creating a new one...")
        faiss_db = create_vector_store()    

    return faiss_db

def retrieve_docs(query, faiss_db):
    '''Retrieve Docs from Vector Store using similarity search'''
    retrieveDocs = faiss_db.similarity_search(query, k=2)
    return retrieveDocs
def get_context_from_docs(documents):
  '''Get pag_content from documents to create context'''

  context = "\n\n".join([doc.page_content for doc in documents])
  for i, doc in enumerate(documents):
      logger.info(f"📋 Chunk {i+1}: {doc.page_content}")
  return context

def get_context(query, faiss_db):
    retrieved_docs = retrieve_docs(query, faiss_db)
    context= get_context_from_docs(retrieved_docs)
    return context


def build_prompt_v5(query, context, chat_history):
    """
    Elite conversational prompt with natural follow-ups and first-person engagement.
    """
    # Format last 4 exchanges as readable text
    history_text = ""
    if chat_history:
        for turn in chat_history[-4:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    return f"""
You are the UpgradeVIP Agent, assisting elite travelers with airport services.

**Chat History:**
{history_text}

**User Question:**
{query}

**Context:**
{context}

---

**CORE IDENTITY:**
- Always use first-person ("I can help you", not "our chatbot can")
- Tone: Warm, professional, refined vocabulary for elite clientele
- Be intelligent, formal, understanding, and conversational
- Add natural phrases ("Great question!", "Absolutely!", "I'd be delighted to help!" etc) to enhance user experience

---

**OUR SERVICES:**
Always present our two core services in numbered format:
1) Airport VIP
2) Airport Transfers

---

**RESPONSE STRUCTURE:**

1. **GREETINGS:**
   - Mirror the user's greeting style (hi → hi, hello → hello, good morning → good morning)
   - For enthusiastic greetings (hiiii, heyyyy), acknowledge enthusiasm but respond professionally
   - For repeated greetings, handle gracefully (e.g., "Hi again!")
   - **Standard Opening:**
     "Welcome, I'm here to assist you with our two premium services:
     1) Airport VIP
     2) Airport Transfers
     Which one would you like me to help you with today?"

2. **CONCISENESS:**
   - Match answer length to query specificity
   - Email query → Email only
   - Services query → Service names only (initially)
   - Always follow with a contextual follow-up question

3. **FOLLOW-UP QUESTIONS:**
   - End responses with relevant follow-ups based on user's query/history
   - Guide unmotivated users gently toward booking

4. **VARIETY:**
   - Never repeat previous answers verbatim
   - Paraphrase responses even for identical queries

---

**CONTENT RULES:**
- No metadata, internal labels, or formatting clutter from Context
- Include links only when highly relevant or explicitly requested
- Out-of-scope queries (e.g., "capital of London", "Elon Musk's salary"):
  → "Apologies, that's outside my expertise. I'm here to assist with UpgradeVIP's airport services. How can I help you today?"

---

**GOAL:**
Make chatting effortless and pleasant. Be relatable, helpful, polite, and professional while subtly guiding users toward booking our services.
"""

def get_gemini_embeddings():
    #api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return embeddings
def get_gemini_llm():
    #api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )
    return llm
def get_gemini_response(query,context,chat_history):
    #api_key = os.getenv("GOOGLE_API_KEY")
    prompt = build_prompt_v5(query, context,chat_history)
    llm = get_gemini_llm()
    return llm.invoke(prompt).content


#=========================rag tool ===========
@tool
def rag_query_tool(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:

    """Retrieve relevant documents from vector store and generate a response using LLM."""
    logger.info(f"🚪 Inside RAG query tool function")
    
    print("🚀 RAG pipeline started...")
    print("Type 'exit' to quit.\n")
    chromaDB = checking_vector_store()
    context = get_context(query,chromaDB)
    answer = get_gemini_response(query,context,chat_history)
    
    logger.info(f"🤖----> Assistant by rag is: {answer}\n")
    return answer

#=============================invoice tools========================

@tool
def build_single_invoice_html_tool(
    title: str,
    service_type: str,
    service_name: str,
    price: float,
    currency: str,
    email: str,
    description: str,
    refund_policy: str
) -> str:
    """
    Generate a single-service HTML invoice.
    Parameters match SINGLE_INVOICE_SCHEMA.
    """
    return build_single_invoice_html(
        title, service_type, service_name, price, currency, email, description, refund_policy
    )

@tool
def build_combined_invoice_html_tool(
    title: str,
    primary_service_type: str,
    primary_service_name: str,
    primary_price: float,
    primary_currency: str,
    primary_description: str,
    primary_refund_policy: str,
    secondary_service_type: str,
    secondary_service_name: str,
    secondary_price: float,
    secondary_currency: str,
    secondary_description: str,
    secondary_refund_policy: str,
    email: str
) -> str:
    """
    Generate a combined-services HTML invoice.
    Parameters match COMBINED_INVOICE_SCHEMA.
    """
    return build_combined_invoice_html(
        title,
        primary_service_type,
        primary_service_name,
        primary_price,
        primary_currency,
        primary_description,
        primary_refund_policy,
        secondary_service_type,
        secondary_service_name,
        secondary_price,
        secondary_currency,
        secondary_description,
        secondary_refund_policy,
        email
    )



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


#==================================vip service tool===================

@tool
def vip_services_tool(airport_id: str, travel_type: str, currency: str, service_id: str = None) -> dict:
    """
    Call external API to fetch available Airport VIP services based on flight details and user preferences.

    Parameters:
      airport_id (str): The airport ID from the flight details object.
        - For 'Departure', use the 'origin_airport' field from flight details (e.g., flight_details["origin_airport"]).
        - For 'Arrival', use the 'destination_airport' field from flight details (e.g., flight_details["destination_airport"]).
      travel_type (str): "Arrival" or "Departure" as selected by the user.
      currency (str): User's preferred currency (USD, EUR, GBP).
      service_id (str, optional): keep it null as we are not taking it from user.

    The agent must extract airport_id from the correct field in primary_flight_details or secondary_flight_details,
    depending on whether the user is booking a primary or secondary service, and based on the user's arrival/departure selection.
    """
    return get_vip_services(airport_id, travel_type, currency, service_id)
@tool
def format_vip_services_tool(
    vip_data: Optional[dict] = None,
    passenger_count: Optional[int] = None,
    currency: Optional[str] = None,
) -> str:
    """Format VIP services into a user-friendly message after calling vip_services_tool."""
    # Defensive guards
    if not all([vip_data, passenger_count, currency]):
        return ("Missing information. Please call vip_services_tool first, "
                "then provide passenger count and currency.")
    if isinstance(vip_data, dict) and "vip_services_tool_response" in vip_data:
        vip_data = vip_data["vip_services_tool_response"]
    return format_vip_services_message(vip_data, passenger_count, currency)

#==============================================Transport tools=========================================================
@tool
def transport_services_tool(airport_id: str, currency: str) -> dict:
    """ Call external API to fetch available transport services for the selected airport and currency.

    Parameters:
      airport_id (str): The airport code. For 'departure', use origin_airport from flight details. For 'arrival', use destination_airport from flight details.
      currency (str): User's preferred currency (USD, EUR, GBP).

    The agent must extract airport_id from the correct field in primary_flight_details or secondary_flight_details,
    depending on whether the user is booking a primary or secondary service, and based on the user's arrival/departure selection.
   """
    return get_transport_services(airport_id, currency)

@tool
def format_transport_services_tool(
    transport_data: Optional[dict] = None,
    currency: Optional[str] = None,
) -> str:
    """Format transfer services into a user-friendly message after calling transport_services_tool."""
    # Defensive guards
    if not all([transport_data, currency]):
        return ("Missing information. Please call transport_services_tool first, "
                "then provide currency.")
    if isinstance(transport_data, dict) and "transport_services_tool_response" in transport_data:
        transport_data = transport_data["transport_services_tool_response"]
    return format_transport_services_message(transport_data, currency)


#========================================== Email Sending tool ===============================================================
@tool
def send_email_tool(to_email: str, subject: str, message: str) -> bool:
    """
    Send booking/invoice email to the user.

    Parameters:
      to_email (str): Recipient email address.
      subject (str): Email subject (e.g. "Your UpgradeVIP Booking Invoice").
      message (str): Email body (HTML or plain text).

    Returns:
      bool: True if email was sent successfully, False otherwise.
    """
    return send_email(to_email, subject, message)
#========================================== Airports List tool ===============================================================
@tool
def airports_tool():
    """Fetch and format airport list for user selection."""
    airports_data = get_airports_from_api()
    return format_airports_message(airports_data)

@tool
def airports_raw_tool() -> dict:
    """
    Fetch the raw airport list from the external API.
    Returns the full JSON/dict response from get_airports_from_api().
    """
    return get_airports_from_api()

@tool
def only_vip_services_tool(airport_id: str, travel_type: str, currency: str, passenger_count: int, flight_data: dict = None, service_id: str = None) -> str:
    """
    Fetch and format VIP services in one step.

    Parameters:
      airport_id (str): The airport ID.
      travel_type (str): "Arrival" or "Departure".
      currency (str): User's preferred currency (USD, EUR, GBP).
      passenger_count (int): Number of adults.
      flight_data (dict, optional): Flight details object (can be None).
      service_id (str, optional): Service ID (default None).

    Returns:
      str: Formatted VIP services message.
    """
    vip_data = get_vip_services(airport_id, travel_type, currency, service_id)
    return format_vip_services_message(vip_data, passenger_count, currency)

@tool
def only_transfer_services_tool(airport_id: str, currency: str) -> str:
    """
    Fetch and format transfer services in one step.

    Parameters:
      airport_id (str): The airport ID.
      currency (str): User's preferred currency (USD, EUR, GBP).

    Returns:
      str: Formatted transfer services message.
    """
    transport_data = get_transport_services(airport_id, currency)
    return format_transport_services_message(transport_data, currency)

#============================================== TOOLS_LIST =========================================================================
tools = [   flight_details_tool,
            format_flight_choice_tool,

            vip_services_tool,
            format_vip_services_tool,
            
            transport_services_tool,
            format_transport_services_tool,
            
            
            send_email_tool,
            build_single_invoice_html_tool,
            build_combined_invoice_html_tool,

            airports_tool,
            airports_raw_tool,

            only_vip_services_tool,
            only_transfer_services_tool,

            rag_query_tool
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
-> For out-of-scope questions or greetings or general queries etc which are not in this prompt call `rag_query_tool(query, chat_history)` where query: string containing the user's current message and chat_history[-4:]: list of the last 4 conversation turns, each as a dict with keys "user" and "assistant" For Example: [{"user": "Hello", "assistant": "Hi there!"}, {"user": "What services?", "assistant": "We offer..."}].

**ROLE**:
Booking only (Airport VIP or Transfers).

HARD GATE:
- First capture `primary_interested` ∈ {"vip","transfer"}.
- Recognize variations: "airport vip", "vip service", "vip lounge" → "vip" | "transfer", "transport", "car service" → "transfer" etc.
- If user clearly states interest (e.g., "I want airport vip"), extract it immediately, acknowledge, and reply
  "Would you like to book on our website or here in chat?
   1) Website: https://order.upgradevip.com/
   2) Chat: Continue here
   Reply with 1 or 2."
IF user replies "1" → provide website link and end conversation.
Else:
proceed to STEP-1 of BOOKING FLOW.
- If user provides unrelated details first (flight number/date before stating service type), STORE them, acknowledge ("Noted."), then ask: "Which service: Airport VIP or Transfer?"
- After `primary_interested` is set, SKIP questions for already-provided details (do not re-ask).


'extracted_info' is a dict in which you will store all the information you collect from the user during the conversation these are the only entities you have store in extracted_info dict:
    "primary_interested":  "vip" or "transfer",
    "primary_flight_number": ly001 or BA111 etc,
    "primary_flight_date": ,
    "primary_flight_details": flight details object returned by flight_details_tool,
    "primary_Arrival_or_departure": arrival or departure,
    "primary_flight_class": economy/plus, business, first,
    "primary_passenger_count": integer number,
    "primary_luggage_count": integer number,
    "primary_preferred_currency": USD, EUR, GBP,
    "primary_get_services": vip services object returned by vip_services_tool,
    "primary_airport_transfer_details": transport services object returned by transport_services_tool,
    "primary_service_selected": service name or number,
    "primary_price": price of selected service,
    "primary_preferred_time": ,
    "primary_msg_for_steward": ,
    "primary_email": ,
    "primary_address": ,
    "primary_confirmation": yes or no,
    "primary_asked_second": yes or no,
    "secondary_interested": "vip" or "transfer",
    "secondary_flight_number": ly001 or BA111 etc,
    "secondary_flight_date": ,
    "secondary_flight_details": flight details object returned by flight_details_tool,
    "secondary_Arrival_or_departure": arrival or departure,
    "secondary_flight_class": economy/plus, business, first,
    "secondary_passenger_count": integer number,
    "secondary_luggage_count": integer number,
    "secondary_preferred_currency": USD, EUR, GBP,
    "secondary_get_services": vip services object returned by vip_services_tool,
    "secondary_airport_transfer_details": transport services object returned by transport_services_tool,
    "secondary_service_selected": service name or number,
    "secondary_price": price of selected service,
    "secondary_preferred_time": ,
    "secondary_msg_for_steward": ,
    "secondary_email": ,
    "secondary_address": ,
    "secondary_confirmation": yes or no

NOTE: when its time to pass extracted_info to any tool always pass the full dict and make value NULL of those entities which you have not collected yet.    

**MULTI SERVICE SELECTION (STRICT)**
- Ask for add-on service .
- ask:
  - If `primary_interested` == "vip": "Do you want to book airport transfer service as well? (yes/no)"
  - Else: "Do you want to book airport VIP service as well? (yes/no)"
- Save reply to `primary_asked_second` ("yes" or "no").

IF `primary_asked_second` == "yes":
- Set `secondary_interested` to the opposite service.
- Follow **BOOKING FLOW** again for the secondary service and store answers as `secondary_*` in 'extracted_info'.
- IMPORTANT:
  - Collect secondary STEP-1 → STEP-10a completely (do NOT skip preferred time or address).
  - Do NOT generate any invoice until secondary required fields are collected.
  - Set `secondary_email = primary_email` .

IF `primary_asked_second` == "no":
- Proceed to invoice (STEP-11).

        
**BOOKING FLOW**
-> after getting user interest 'vip' or 'transfer' extracted in 'primary_interested' then follow the respective flow below.
STEP-1: After getting 'primary_interested', ask for flight number (show example: LY001, BA111). 
STEP-2: ask for date (show example: 10/31/25, in MM/DD/YYYY format).
STEP-3: After having primary_interested, primary_flight_number,primary_flight_date, IMMEDIATELY call `flight_details_tool(primary_flight_number, primary_flight_date)` to fetch flight details.
        - Present the flight details to the user using `format_flight_choice_tool`.
STEP-4: Then ask user for travel type (Arrival or Departure).
STEP-5: Then ask user for number of adults (also showing user : children above 2 also included ; range is 1-10).
STEP-6: Then ask for and luggage (range is 1-10).
STEP-7: Then ask user for preferred currency (USD, EUR, GBP).
STEP-8: Fetch and display services:
        IF primary_interested is 'transfer':
              Call `only_transfer_services_tool(airport_id, currency)`:
                 - airport_id: `origin_airport` (Departure) or `destination_airport` (Arrival) from primary_flight_details
                 - currency: primary_preferred_currency
              Display output → Ask: "Please select a service by name or number i.e 1,2 etc."
              If tool returns "Sorry, no transport services are available...":**
                 - Display the exact message to user
                
        ELSE IF primary_interested is 'vip':
              Call `only_vip_services_tool(airport_id, travel_type, currency, passenger_count)`:
                 - airport_id: `origin_airport` (Departure) or `destination_airport` (Arrival) from primary_flight_details
                 - travel_type: primary_Arrival_or_departure
                 - currency: primary_preferred_currency
                 - passenger_count: primary_passenger_count
              Display output → Ask: "Please select a service by name or number i.e 1,2 etc."
              If tool returns "Sorry, no VIP services are available...":**
                 - Display the exact message to user
                
STEP-9: Ask for message for steward. 
STEP-10: After that Then you have to ask for preferred time of meeting to the user.
STEP-10a:  Only Ask for address from user if 'primary_interested' is 'transfer' otherwise skip it.
STEP-11: After that Then you have to Ask for Email id of the user.
STEP-11b (ADD-ON CHECK):
- After STEP-11, run **MULTI SERVICE SELECTION (STRICT)**.
STEP-12: After you have collected the user's email:
1) Immediately assemble ALL previously collected booking info into a dict called `extracted_info` (include ALL required keys; set missing to null).
2) Create invoice schema(s) (mapping-only; do NOT invent values):

SINGLE_INVOICE_SCHEMA = {
  "title": "UpgradeVIP",
  "service_type": "primary_interested",
  "service_name": "primary_service_selected (matched title/name)",
  "price": "primary_price",
  "currency": "primary_preferred_currency",
  "email": "primary_email",
  "description": "matched service words/description",
  "refund_policy": "matched service refund_text/refund policy"
}

COMBINED_INVOICE_SCHEMA = {
  "title": "UpgradeVIP",
  "primary_service": {
    "service_type": "primary_interested",
    "service_name": "primary_service_selected (matched title/name)",
    "price": "primary_price",
    "currency": "primary_preferred_currency",
    "description": "primary matched service words/description",
    "refund_policy": "primary matched service refund_text/refund policy"
  },
  "secondary_service": {
    "service_type": "secondary_interested",
    "service_name": "secondary_service_selected (matched title/name)",
    "price": "secondary_price",
    "currency": "secondary_preferred_currency",
    "description": "secondary matched service words/description",
    "refund_policy": "secondary matched service refund_text/refund policy"
  },
  "email": "primary_email",
  "total_price": "primary_price + secondary_price (ONLY if same currency; otherwise show both separately)",
  
}

 IF 'primary_interested' is set but secondary_interested is NOT set
    - call 'build_single_invoice_html_tool' with 'SINGLE_INVOICE_SCHEMA'.
    - display invoice to user.
 Else
    - call 'build_combined_invoice_html_tool' with 'COMBINED_INVOICE_SCHEMA'.
    - display invoice to user.

3) Ask confirmation: skip a line then :"Please confirm (yes/no)."
- If YES: call `send_email_tool(primary_email, "UpgradeVIP Booking Invoice", invoice)`
- If NO: ask what to change and resume from that step.



**AIRPORTS LIST FLOW:**
- If the user asks to see all airports, requests an airport list, or asks for available airports, call `airports_tool()` to fetch and format the airport list.
- And display output to the user in readable format.
- Do not ask for flight details or service selection until the user selects an airport or continues with booking.   


**SERVICES BY AIRPORT NAME FLOW:**
- Trigger when user asks for services in a city/airport (e.g., "services in Dubai", "VIP at Heathrow", "transfers at JFK", "Abu Dhabi International Airport").

**Step 1: Extract and Match Airport**
1. Extract the airport name from the user's message (e.g., "Abu Dhabi", "Dubai", "Heathrow").
2. **Immediately call `airports_raw_tool()`** to fetch the full airport list.
3. **Search the returned list** for a matching airport by comparing the user's input (case-insensitive) against the `airport_name` field.
   - Match logic: Check if the user's input appears anywhere in the `airport_name` string.
   - Example: User says "Abu Dhabi" → Match "Abu Dhabi International Airport AUH" → Extract `"id": "204"`.
4. **Extract the `id` field**  from the matched airport object and store it internally as airport_ID (do NOT display the ID or technical details to the user).
5. If **no match is found**:
   - Reply: "I couldn't find an airport matching '[user's input]'. Could you clarify or provide the IATA code (e.g., AUH, DXB)?"
   - Do NOT proceed until a valid airport is identified.
   If **match is found**:
  - Reply: "I've found [full airport name]."   

**Step 2:**
1. "What type of service are you interested in: **Airport VIP** or **Transfer**?" → Store as `service_type`.
**Step 3:**
2. "Is this for **Arrival** or **Departure**?" → Store as `Travel_Type`.

**Step 4: Fetch Services**
- **If service_type is "vip":**
  - Call `only_vip_services_tool(airport_id=airport_ID, travel_type=Travel_Type, currency="USD", passenger_count=1, flight_data=None, service_id=None)`
- **If service_type is "transfer":**
  - Call `only_transfer_services_tool(airport_id=airport_ID, currency="USD", passenger_count=1, flight_data=None, arrival_or_departure=Travel_Type)`

**Step 5: Display Results Only**
- Display the exact output from the tool (service cards with prices, descriptions, cancellation policies).
- **DO NOT ask the user to select a service** in this flow.
- **DO NOT add "Please select any service card by name or by number."**
- Simply show the services

**REMEMBER:**
- When calling single_generate_invoice_tool or generate_combined_invoice_tool, always pass the full extracted_info dict with all required keys. For any value not collected, set it to null (None). Never omit required fields.
- Always ask for missing required information before calling a tool according to the conversation flows.
- Never invent or assume values.
  

"""


agent = create_react_agent(
    llm,
    tools=tools,
    checkpointer=memory,  # save convo in RAM
    prompt=SYSTEM_PROMPT,
  #  pre_model_hook=trim_messages,  # run trimming before each model call     
)
