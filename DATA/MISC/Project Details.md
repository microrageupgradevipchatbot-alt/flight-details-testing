# UpgradeVIP AI Chatbot Project
Short description of project

🚀 UpgradeVIP AI Chatbot - Documentation

# ✨ Key Features (Need Revision )

- RAG-Powered FAQ: Answers general questions using ChromaDB knowledge base

- Multi-Service Booking: Book Airport VIP + Transfer in a single conversation


- Real-Time Flight Integration: Live flight details via external API

- Multi-Currency Support: USD, EUR, GBP with automatic price calculation

- Automated Invoice Generation: Professional HTML invoices for single/ combined bookings

- Email Automation: Sends booking confirmations via SMTP (Gmail)

- Session Management: Maintains conversation history per user

- Airport Directory: Lists all available airports for user selection

# Tech Stack & Tools

Core Technologies

- AI Model: Google Gemini 2.5 Flash (via langchain-google-genai)
- Framework: LangChain + LangGraph (ReAct Agent)
- Backend: FastAPI (async endpoints, CORS enabled)
- Frontend: React
- Vector Database: ChromaDB (semantic search, embeddings)
- Memory: InMemorySaver (conversation state management)
- Email: SMTP (Gmail app password)

## imports

- standard imports
- third party imports
- rag
- langgraph

# load .env file
**API_KEYS**
- google api key
- upgradevip url
- upgradevip bareer token

# Seting up directories
- vector db
- docs

# Setting up logger file

# Functions
1. get_airports_from_api
2. format_airports_message
3. get_flight_details_from_api
4. format_flight_choice_message_impl
5. get_vip_services
6. format_vip_services_message
7. get_transport_services
8. format_transport_services_message
9. generate_single_invoice
10. send_email
11. generate_combined_invoice
Rag
12. get_gemini_embeddings
13. get_gemini_response
14. get_gemini_response

15. load_documents
16. create_chunks
17. create_chroma_db

18. load_vector_store
19. create_vector_store
20. checking_vector_store

21. retrieve_docs
22. get_context_from_docx
23. get_context
24. build_prompt

# langgraph

- create agent (create_react_agent)
it will act like this:

System_prompt->user->agent->tool_call
->tool_response->agent(reply)->user->.....

# FAST-API
- pydentic class
  having variables
  - message 
  - session

- Endpoints

1. / 
root 
in order to know whether api started successfully or not

2. /message
- takes user input
- generate session id (so that llm uniquely identify new user and keep only that particular user's chat_history)
- if session id is new only then pass 'System_prompt'
- call the agent and give it user's query
- show bot_reply
