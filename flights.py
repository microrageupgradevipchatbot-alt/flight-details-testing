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
load_dotenv()
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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

if  GOOGLE_API_KEY:
    logger.info("GOOGLE_API_KEY loaded successfully.")
else:
    logger.info("GOOGLE_API_KEY not found. Please set it in your environment or .env file.")

# if not st.secrets.get("DEV_URL"):  
#     logger.warning("DEV_URL not found. Please set it in your environment or .env file.")
# if not st.secrets.get("API_KEY"):  
#     logger.warning("API_KEY not found. Please set it in your environment or .env file.")  
# dev_url = st.secrets.get("DEV_URL")
# api_key = st.secrets.get("API_KEY")
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
            return {"error": "Invalid JSON response", "message": "VIP services API returned invalid data format", "data": []}
    except Exception as e:
        logger.error(f"❌ VIP services API request failed: {e}")
        return {"error": "Request failed", "message": str(e), "data": []}

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
    
    msg = "✈️ **Available VIP Services:**\n\n"
    
    for i, service in enumerate(services, 1):
        title = service.get("title", "VIP Service")
        price_key = f"adults_{passenger_count}"
        price = service.get(price_key, service.get("price", 0))
        refund_text = service.get("refund_text", "")
        words = service.get("words", "")
        
        # Header with visual separator
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"**{i}. {title}**\n"
        msg += f"💰 **Price:** {symbol}{price} {preferred_currency}\n\n"
        
        # Description with bullets
        msg += "📋 **Included Services:**\n"
        if words:
            details = [detail.strip() for detail in words.split(",")]
            for detail in details:
                if detail:
                    msg += f"   • {detail}\n"
        else:
            msg += "   • Not specified\n"
        msg += "\n"
        
        # Cancellation policy
        msg += "🔄 **Cancellation Policy:**\n"
        if refund_text:
            msg += f"   {refund_text}\n"
        else:
            msg += "   Not specified\n"
        
        msg += "\n"
    
    logger.info(f"✈️ Formatted VIP services message: {msg}")
    return msg
#======================================invoice====================================

def build_dynamic_invoice_html(title: str, bookings: List[Dict[str, Any]], email: str) -> str:
    """Build one invoice for any number of bookings (cart-style)."""
    logger.info("🚪 Inside build_dynamic_invoice_html")

    if not bookings:
        return "<h2><b>UpgradeVIP</b></h2><p>No bookings found.</p>"

    currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    totals_by_currency: Dict[str, float] = {}
    sections: List[str] = [f"<h2><b>{title} - Combined Booking</b></h2>", "<br>"]

    for i, booking in enumerate(bookings, 1):
        service_type = str(booking.get("service_type", "")).lower()
        service_type_display = "Airport VIP" if service_type == "vip" else "Airport Transfer"

        selected = booking.get("selected_service") or {}
        service_name = selected.get("name") or booking.get("service_name") or "N/A"
        service_currency = selected.get("currency") or booking.get("preferred_currency") or booking.get("currency") or "USD"

        raw_price = selected.get("price", booking.get("price", 0))
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            price = 0.0

        totals_by_currency[service_currency] = totals_by_currency.get(service_currency, 0.0) + price
        symbol = currency_symbols.get(service_currency, "")

        description = selected.get("description") or booking.get("description") or "N/A"
        refund_policy = selected.get("refund_policy") or booking.get("refund_policy") or "N/A"
        address = booking.get("address")
        address_html = f"<b>Address:</b> {address}<br>" if address else ""

        sections.append(f"<h3>🎫 SERVICE #{i}</h3>")
        sections.append("<h4>📋 Booking Details:</h4>")
        sections.append(f"<b>Service Type:</b> {service_type_display}<br>")
        sections.append(f"<b>Flight Number:</b> {booking.get('flight_number', 'N/A')}<br>")
        sections.append(f"<b>Flight Date:</b> {booking.get('flight_date', 'N/A')}<br>")
        sections.append(f"<b>Travel Type:</b> {booking.get('arrival_or_departure', 'N/A')}<br>")
        sections.append(f"<b>Passengers:</b> {booking.get('passenger_count', 'N/A')}<br>")
        sections.append(f"<b>Luggage:</b> {booking.get('luggage_count', 'N/A')}<br>")
        sections.append(f"<b>Meeting Time:</b> {booking.get('preferred_time', 'N/A')}<br>")
        sections.append(address_html)
        sections.append("<h4>Service Details:</h4>")
        sections.append(f"<b>Service:</b> {service_name}<br>")
        sections.append(f"<b>Price:</b> {symbol}{price:.2f} {service_currency}<br>")
        sections.append(f"<b>Description:</b> {description}<br>")
        sections.append(f"<b>Refund Policy:</b> {refund_policy}<br>")
        if booking.get("message_for_steward"):
            sections.append(f"<b>Message for Steward:</b> {booking.get('message_for_steward')}<br>")
        sections.append("<hr>")

    sections.append("<h3>Total</h3>")
    if len(totals_by_currency) == 1:
        only_currency = list(totals_by_currency.keys())[0]
        symbol = currency_symbols.get(only_currency, "")
        sections.append(f"<b>Grand Total:</b> {symbol}{totals_by_currency[only_currency]:.2f} {only_currency}<br>")
    else:
        sections.append("<b>Grand Total by Currency:</b><br>")
        for cur, amount in totals_by_currency.items():
            symbol = currency_symbols.get(cur, "")
            sections.append(f"- {symbol}{amount:.2f} {cur}<br>")

    sections.append("<br>")
    sections.append(f"<b>📧 Email:</b> {email}")

    invoice = "\n".join(sections)
    logger.info(f"🎫 Dynamic Invoice Output:\n{invoice}")
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
            return {"error": "Invalid JSON response", "message": "Transport services API returned invalid data format", "data": []}
    except Exception as e:
        logger.error(f"❌ Transport services API request failed: {e}")
        return {"error": "Request failed", "message": str(e), "data": []}
def format_transport_services_message(transport_data, preferred_currency):
    logger.info(f"🚪 Inside Formatting transport services message function")
    vehicles = transport_data.get("data", []) if transport_data else []
    if not vehicles:
        return "Sorry, no transport services are available for your selected flight and requirements."
    
    currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = currency_symbols.get(preferred_currency, "$")

    msg = "🚗 **Available Transport Services:**\n\n"

    for i, vehicle in enumerate(vehicles, 1):
        name = vehicle.get("name", "Transport Service")
        price = vehicle.get("price", 0)
        capacity = vehicle.get("capacity", "N/A")
        words = vehicle.get("words", "")
        
        # Header with visual separator
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"**{i}. {name}**\n"
        msg += f"💰 **Price:** {symbol}{price} {preferred_currency}\n"
        msg += f"👥 **Capacity:** {capacity} passengers\n\n"
        
        # Features with bullets
        msg += "📋 **Features:**\n"
        if words:
            details = [detail.strip() for detail in words.split(",")]
            for detail in details:
                if detail:
                    msg += f"   • {detail}\n"
        else:
            msg += "   • Not specified\n"
        
        msg += "\n"

    logger.info(f"🚗 Formatted transport services message: {msg}")
    return msg
#================================== Email Function ==============================
def send_email(to_email, subject, message):
    logger.info(f"🚪 Inside send email function")
    
    # Replace with your SMTP server details
    # smtp_server = st.secrets["SMTP_SERVER"]
    # smtp_port = int(st.secrets["SMTP_PORT"])
    # smtp_user = st.secrets["SMTP_USER"]
    # smtp_pass = st.secrets["SMTP_PASS"]
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
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
    Neutral support prompt for concise, factual RAG answers.
    """
    print("Building prompt v7...")
    
    history_text = ""
    if chat_history:
        for turn in chat_history[-4:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    return f"""
You are the UpgradeVIP support assistant.

**Chat History:**
{history_text}

**Current Query:**
{query}

**Context:**
{context}

---

**ROLE:**
- Answer in first person when needed, but keep the tone neutral and factual.
- Act like a support assistant, not a concierge or salesperson.
- Use short, clear sentences.
- Prioritize accuracy from the provided context.

---

**GREETING RULES:**
- Only greet if the user greets first or if this is the first interaction.
- Keep greetings brief and neutral.
- Do not use repeated greetings such as "Hello again", "Hi there", "Welcome back", or similar variations.
- If the conversation is already underway, skip the greeting and answer directly.

---

**RESPONSE RULES:**

1. Keep answers short and factual.
   - Default to 1 to 3 sentences.
   - For policy or FAQ questions, answer directly from context without extra promotion.
   - For simple contact questions, provide only the requested detail.

2. Keep the tone neutral.
   - Do not use marketing language, luxury language, or sales phrasing.
   - Do not exaggerate benefits.
   - Do not persuade the user to book unless they explicitly ask about booking.

3. Avoid unnecessary closings.
   - Do not add fillers such as "Anything else?", "Let me know if you need more", or similar closings unless a follow-up is required to answer correctly.
   - Ask at most one follow-up question, and only when required to clarify missing information.

4. Use chat history carefully.
   - If the user repeats a question, paraphrase the reply but keep the same facts.
   - If the user changes topic, answer the new topic directly.

5. Use the context as the source of truth.
   - Extract only the actual answer from context.
   - Do not expose metadata, section names, keyword lists, or internal formatting.
   - Do not invent details that are not present in context.

---

**SPECIAL CASES:**

**Out of scope:**
"Sorry, I can only help with UpgradeVIP services and airport assistance information."

**Pricing:**
If pricing is present in context, provide it briefly.
If pricing is not present, say: "Pricing depends on the airport and service requested."

**Complaints:**
For example : "I'm sorry about that. Please share the issue and I will help with the next step." etc

---

**NEVER:**
- Use repeated or chatty greetings.
- Add promotional wording.
- Add unnecessary follow-up lines.
- Use phrases like "As an AI".
- Include context metadata in the final answer.

**GOAL:**
Provide concise, accurate, support-style answers grounded in the dataset.
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
def build_dynamic_invoice_html_tool(title: str, bookings: List[Dict[str, Any]], email: str) -> str:
    """Generate a single combined invoice for any number of bookings."""
    return build_dynamic_invoice_html(title, bookings, email)


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
# @tool
# def send_email_tool(to_email: str, subject: str, message: str) -> bool:
#     """
#     Send booking/invoice email to the user.

#     Parameters:
#       to_email (str): Recipient email address.
#       subject (str): Email subject (e.g. "Your UpgradeVIP Booking Invoice").
#       message (str): Email body (HTML or plain text).

#     Returns:
#       bool: True if email was sent successfully, False otherwise.
#     """
#     return send_email(to_email, subject, message)

@tool
def checkout_and_send_invoice_tool(title: str, bookings: List[Dict[str, Any]], email: str) -> dict:
    """
    Build invoice and send email in one atomic tool call.
    This avoids passing large HTML payload between separate tool calls.
    """
    if not bookings:
        return {"ok": False, "message": "No bookings found.", "email_sent": False}

    invoice = build_dynamic_invoice_html(title, bookings, email)
    sent = send_email(email, "UpgradeVIP Booking Invoice", invoice)

    return {
        "ok": bool(sent),
        "email_sent": bool(sent),
        "email": email,
        "message": "Invoice emailed successfully." if sent else "Failed to send invoice email.",
        "invoice": invoice,
    }
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
            
            
            # send_email_tool,
            build_dynamic_invoice_html_tool,
            checkout_and_send_invoice_tool,

            # airports_tool,
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

**ROLE:**
Booking assistant for Airport VIP and Airport Transfer.

**SMART FIELD REUSE (IMPORTANT):**
- Extract any booking fields already provided by user in the same or previous messages (service type, flight number, flight date, arrival/departure, passengers, luggage, currency, etc.) and store them in extracted_info.current_booking.
- Ask only missing fields.
- Never re-ask a field that already has a valid value unless the user asks to change it.
- Example: if user says "book vip, flight date is 10/03/2026", do not ask for flight date again; ask next missing field (like flight number).

**MANDATORY GATE:**
- When booking intent is detected, always ask:
    "Would you like to book on our website or here in chat?
     1) Website: https://order.upgradevip.com/
     2) Chat: Continue here
     Reply with 1 or 2."
- If user replies 1: share link and stop booking flow.
- If user replies 2: continue booking flow.

STATE (use this shape in extracted_info):
{
    "email": null,
    "bookings": [],
    "current_booking": {
        "service_type": null,
        "flight_number": null,
        "flight_date": null,
        "flight_details": null,
        "arrival_or_departure": null,
        "passenger_count": null,
        "luggage_count": null,
        "preferred_currency": null,
        "selected_service": {
            "name": null,
            "price": null,
            "currency": null,
            "description": null,
            "refund_policy": null
        },
        "preferred_time": null,
        "message_for_steward": null,
        "address": null
    }
}

**BOOKING FLOW (LOOP):**
1) Ask service type (vip/transfer) only if missing.
2) Ask flight number (LY001, BA111) only if missing.
3) Ask flight date (MM/DD/YYYY) only if missing.
4) Call flight_details_tool, then call format_flight_choice_tool and return its output exactly as-is (no rewrite, no summary, no extra text).
5) Ask Arrival or Departure only if missing.
6) Ask passenger count (1-10) only if missing.
7) Ask luggage count (1-10) only if missing.
8) Ask currency (USD/EUR/GBP) only if missing.
9) Fetch services:
     - vip: only_vip_services_tool(airport_id, travel_type, currency, passenger_count)
     - transfer: only_transfer_services_tool(airport_id, currency)
     airport_id from flight_details:
     - Departure -> origin_airport
     - Arrival -> destination_airport
    - Return the selected tool output exactly as-is (no paraphrase or reformat).
10) Ask service selection by name or number, then map to selected_service with name/price/currency/description/refund_policy.
11) Ask message for steward.
12) Ask preferred meeting time.
13) If transfer, ask address. If vip, skip address.
14) Append current_booking to bookings.
15) Ask: "Do you want to add another service? (yes/no)"
        - yes: reset current_booking and repeat flow.
        - no: proceed checkout.

**CHECKOUT:**
1) Ask email once if missing.
2) Show brief summary of all bookings.
3) Ask: "Is this correct? (yes/no)"
4) If yes:
     - ask: "Please confirm (yes/no)."
    - if yes: call checkout_and_send_invoice_tool("UpgradeVIP", bookings, email)
    - if tool result ok=true: confirm email sent successfully
    - if tool result ok=false: apologize and ask user to retry or update email
     - if no: ask what to change and continue

**RULES:**
- Never invent values.
- Never generate invoice when bookings is empty.
- Keep totals per currency if mixed currencies.
- If no services returned, tell user and ask to change travel type or currency.
- For output from format_flight_choice_tool, format_vip_services_tool, format_transport_services_tool, only_vip_services_tool, only_transfer_services_tool: return verbatim tool text only.
- Do not paraphrase, reorder, shorten, add emojis, or add extra commentary around these tool outputs.
- If tool returns an error or empty-result message, return that message exactly as-is.
- Keep replies concise outside mandatory booking questions and verbatim tool output.

**AIRPORT QUERIES:**
- Count/general: "Upgrade VIP operates at over 430 airports worldwide. Please tell me the airport or city you'd like to check."
- Specific airport/city: use airports_raw_tool with case-insensitive matching.

**SERVICES BY AIRPORT NAME FLOW:**
- Trigger when user asks for services in a city/airport (examples: "services in Dubai", "VIP at Heathrow", "transfers at JFK", "Abu Dhabi International Airport").

Step 1: Extract and Match Airport
1) Extract airport/city name from user message.
2) Immediately call airports_raw_tool().
3) Search returned list by case-insensitive contains match against airport_name.
4) If matched: store matched airport id internally as airport_ID and reply "I've found [full airport name]."
5) If no match: reply "I couldn't find an airport matching '[input]'. Could you clarify or provide the IATA code (e.g., AUH, DXB)?" and stop this flow until clarified.

Step 2:
1) Ask: "What type of service are you interested in: Airport VIP or Transfer?"

Step 3:
1) Ask: "Is this for Arrival or Departure?"

Step 4: Fetch Services
- If service_type is vip:
    call only_vip_services_tool(airport_id=airport_ID, travel_type=Travel_Type, currency="USD", passenger_count=1, flight_data=None, service_id=None)
- If service_type is transfer:
    call only_transfer_services_tool(airport_id=airport_ID, currency="USD")

Step 5: Display Results Only
- Display the tool output as-is.
- Do not ask user to select service in this informational flow.
- Do not convert this informational flow into booking unless user explicitly asks to book.

"""


agent = create_react_agent(
    llm,
    tools=tools,
    checkpointer=memory,  # save convo in RAM
    prompt=SYSTEM_PROMPT,
  #  pre_model_hook=trim_messages,  # run trimming before each model call     
)
