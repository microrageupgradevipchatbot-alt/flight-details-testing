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
import streamlit as st


#RAG
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader,UnstructuredMarkdownLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#langgraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent


# setup paths and logging
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

    LOGS_DIR = CURRENT_DIR / 'DATA' / 'Logs'
    LOGS_DIR.mkdir(exist_ok=True)
    LOG_FILE = LOGS_DIR / 'UPGRADEVIP_COMPLETE.log'
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

#====================llm key==============
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    logger.info("GOOGLE_API_KEY loaded from Streamlit secrets.")
except Exception as e:
    logger.error("GOOGLE_API_KEY not found in Streamlit secrets.")
    st.error("API key not configured. Please contact the administrator.")
    st.stop()

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

#==========================RAG========================
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
        print(f"full filename {file_path}")

        if filename.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
            print(f"loaded {filename}")
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            print(f"loaded {filename}")
        else:
            print(f"unsupported file type : {filename}")
            continue

        documents.extend(loader.load())
    
    return documents
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=450
        )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks from {len(documents)} documents.")
    return docs
def create_chroma_db(text_chunks):
  '''Index Documents (store embeddings) in vector store'''
  
  embeddings = get_gemini_embeddings()
  collection_name = "upgrade_collection"
  print(f"✅Indexing {len(text_chunks)} chunks into Chroma vector store '{collection_name}'...")
  chromaDB =  Chroma.from_documents(
      collection_name=collection_name,
      documents=text_chunks,
      embedding=embeddings,
      persist_directory=DB_DIR
  )
  print("✅Chroma vector store created successfully.")
  return chromaDB

def load_vector_store():
    """Create the Chroma vector store if not present, else load it."""
    embeddings = get_gemini_embeddings()
    chromaDB = Chroma(
        collection_name="upgrade_collection",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    print("✅Chroma vector store loaded from disk.")

    return chromaDB
def create_vector_store():
    """Create the Chroma vector store if not present, else load it."""
  
    documents = load_documents(DOC_DIR)
    text_chunks = create_chunks(documents)
    chromaDB = create_chroma_db(text_chunks)

    return chromaDB
def checking_vector_store():
    '''Check if vector store exists, if not create it'''
    
    if (DB_DIR / "chroma.sqlite3").exists():
        print("✅ Existing vector store found. Loading it...")
        chromaDB = load_vector_store()
    else:
        print("🛑 No existing vector store found. Creating a new one...")
        chromaDB = create_vector_store()    

    return chromaDB

def retrieve_docs(query,chromaDB):
  '''Retrieve Docs from Vector Store using similarity search'''
  retrieveDocs = chromaDB.similarity_search(query,k=2)

  return retrieveDocs
def get_context_from_docs(documents):
  '''Get pag_content from documents to create context'''

  context = "\n\n".join([doc.page_content for doc in documents])
  for i, doc in enumerate(documents):
      logger.info(f"📋 Chunk {i+1}: {doc.page_content}")
  return context
def get_context(query,chromaDB):
    retrieved_docs = retrieve_docs(query,chromaDB)
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

def get_gemini_response(query,context,chat_history):
    prompt = build_prompt_v5(query, context,chat_history)
    llm = get_gemini_llm()
    return llm.invoke(prompt).content

@tool
def rag_query_tool(query: str, chat_history: List[Dict[str, str]]):
    """Retrieve relevant documents from vector store and generate a response using LLM."""
    try:
        logger.info(f"🚪 Inside RAG query tool function")
        logger.info(f"📝 Query: {query}")
        logger.info(f"📜 Chat history: {chat_history}")
        
        print("🚀 RAG pipeline started...")
        chromaDB = checking_vector_store()
        context = get_context(query, chromaDB)
        answer = get_gemini_response(query, context, chat_history)
        
        logger.info(f"🤖----> Assistant by rag is: {answer}\n")
        return answer
    except Exception as e:
        logger.error(f"❌ RAG tool failed: {str(e)}", exc_info=True)
        return f"I encountered an error while processing your request: {str(e)}"
#===========================
tools=[rag_query_tool]
SYSTEM_PROMPT = """
-> For out-of-scope questions or greetings or general queries etc which are not in this prompt call `rag_query_tool(query, chat_history)` where query: string containing the user's current message and chat_history[-4:]: list of the last 4 conversation turns, each as a dict with keys "user" and "assistant" For Example: [{"user": "Hello", "assistant": "Hi there!"}, {"user": "What services?", "assistant": "We offer..."}].

**ROLE**:
Booking only (Airport VIP or Transfers).

HARD GATE:
- First capture `primary_interested` ∈ {"vip","transfer"}.
- Do NOT ask for flight number, date, class, passengers, luggage, currency, or any other field until `primary_interested` is set.
- If the user provides details out of sequence (e.g., flight number/date first), STORE them, acknowledge ("Noted."), then ask: "Which service would you like to book: Airport VIP or Transfer?"
- After `primary_interested` is set, continue the BOOKING FLOW and SKIP questions for any details already provided (do not re-ask).

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

**MULTI SERVICE SELECTION**
- if you are following 'BOOKING FLOW' 
    - when you reach that step where you ask for email,first you should ask user if he wants to book airport transfer if 'primary_interested' is 'vip' else ask user if he wants to book airport vip service as well. 
      - and extract user response and save it in 'primary_asked_second' as yes or no. 
        if its yes:
            than secondary_interested value will automatically be transfer if 'primary_interested' is vip else secondary_interested will become vip.
            and start following 'BOOKING FLOW' from start and from now on extract user responses as secondary_* as it is given in the extracted_info dict above.
            and the step where it ask for email we will copy that secondary_email value in primary_email and then we have top call combined_generate_invoice_tool(extracted_info) instead of single_generate_invoice_tool(extracted_info)
            and have to copy secondary_confirmation value in primary_confirmation where it will ask for confirmation from the user in the flow
        else: keep following 'BOOKING FLOW' from where you left it.

        
**BOOKING FLOW**
-> after getting user interest 'vip' or 'transfer' extracted in 'primary_interested' then follow the respective flow below.
STEP-1: After getting 'primary_interested', ask for flight number (show example: LY001, BA111). 
STEP-2: ask for date (show example: 10/31/25, in MM/DD/YYYY format).
STEP-3: After having primary_interested, primary_flight_number,primary_flight_date, IMMEDIATELY call `flight_details_tool(primary_flight_number, primary_flight_date)` to fetch flight details.
        - Present the flight details to the user using `format_flight_choice_tool`.
STEP-4: Then ask user for travel type (Arrival or Departure).
STEP-5: Then ask for flight class (Economy/plus, Business, First).
STEP-6: Then ask user for number of adults (also showing user : children above 2 also included ; range is 1-10).
STEP-7: Then ask for and luggage (range is 1-10).
STEP-8: Then ask user for preferred currency (USD, EUR, GBP).
STEP-9: 
        IF primary_interested is 'transfer' :
              Firstly Use `transport_services_tool(airport_id, currency)` to fetch available transport services after collecting primary_flight_details, primary_Arrival_or_departure, primary_passenger_count, primary_luggage_count and primary_preferred_currency.
                - For Departure, use `origin_airport`; for Arrival, use `destination_airport` from the primary_flight_details as airport_id.
              And After this Immediately use `format_transport_services_tool(transport_data, flight_data, passenger_count, preferred_currency, arrival_or_departure)` and show exact output to user .
                - When calling format_transport_services_tool, always pass:
                - transport_data: the exact dict returned by transport_services_tool (no extra nesting)
                - flight_data: the dict returned by flight_details_tool
                - passenger_count: as provided by the user primary_passenger_count
                - preferred_currency: as provided by the user primary_preferred_currency
                - arrival_or_departure: as provided by the user primary_Arrival_or_departure
              Right After that present transport options to user and ask user to select any service card by name or by number.  

        ELSE IF primary_interested is 'vip' :
              Firstly Use `vip_services_tool(airport_id, travel_type, currency, service_id)` after collecting primary_flight_details, primary_Arrival_or_departure, primary_flight_class, ptimary_passenger_count, primary_luggage_count, and primary_preferred_currency to fetch available VIP services.
                - For Departure, use `origin_airport`; for Arrival, use `destination_airport` from the primary_flight_details as airport_id.
              And After this Immediately call `format_vip_services_tool(vip_data, flight_data, travel_type, passenger_count, preferred_currency)` and show exact output to user .
                - When calling format_vip_services_tool, always pass:
                - vip_data: the exact dict returned by vip_services_tool (no extra nesting)
                - flight_data: the dict returned by flight_details_tool
                - travel_type: as provided by the user primary_Arrival_or_departure
                - passenger_count: as provided by the user primary_passenger_count
                - preferred_currency: as provided by the user primary_preferred_currency
               Right After that present VIP service options to user and ask user to select any service card by name or by number.  
       
STEP-10: Ask for message for steward. 
STEP-11: After that Then you have to ask for preferred time of meeting to the user.
STEP-11a:         Only Ask for address from user if 'primary_interested' is 'transfer' otherwise skip it.
STEP-12: After that Then you have to Ask for Email id of the user.
STEP-13: After you have collected the user's email, immediately assemble ALL previously collected booking information into a dict called `extracted_info` and immediateky call `single_generate_invoice_tool(extracted_info)`. 
         - after generating the invoice, ask user for confirmation.
- if user confirms then :
   - Use `send_email_tool(to_email, subject, message)` to send booking confirmations or invoices after user confirmation.
- else if user does not confirm then :
   - politely ask user what changes he want to make in the booking and restart the flow from there.


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
     

**REMEMBER:**
- When calling single_generate_invoice_tool or generate_combined_invoice_tool, always pass the full extracted_info dict with all required keys. For any value not collected, set it to null (None). Never omit required fields.
- Always ask for missing required information before calling a tool according to the conversation flows.
- Never invent or assume values.
  

"""

#==================================== Create LangGraph Agent =========================================
# At the bottom, replace the duplicate agent code with this:
try:
    llm = get_gemini_llm()
    memory = InMemorySaver()
    
    agent = create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
    logger.info("✅ Agent created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create agent: {str(e)}", exc_info=True)
    raise

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

