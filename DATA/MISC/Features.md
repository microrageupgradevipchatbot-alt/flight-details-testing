🚀 Features

AI-based Conversational Booking Assistant
Chatbot built with LangGraph + Google Gemini.
Handles natural conversation with memory for each user session.
Automatically decides which tool or API to call based on system prompt.

# RAG (Retrieval-Augmented Generation) Support

Uses Chroma vector DB + Google Generative AI embeddings.
Answers user FAQs about UpgradeVIP services.
Rejects out-of-scope queries politely (e.g., “capital of France”).

# Multi-Service Handling

User can book both VIP and Transfer in the same conversation.
Combines both into a single invoice when applicable.

# Backend & API

FastAPI backend with /message endpoint for chatbot communication.
Supports CORS for frontend integration (React).

# Logs

Logging system records all API calls, user actions, and errors.

---

# User Session Management

Each user gets a unique session ID.
Chat history is remembered during the entire conversation.

# Flight details Flow

User provides flight number and date.
System fetches flight details using real API.
Bot formats and presents flight options clearly.

# Airport VIP & Transfer Services

Two main booking options: Airport VIP and Transfer.
Fetches VIP or Transport services from API dynamically.
Formats results into card-style choices for user.
Lets users select by number or service title.

# Invoice Generation

Automatically generates a structured HTML invoice.
Includes flight info, service details, passenger data, and total price.
Supports both single and combined invoices if user books multiple services.

# Email Integration

Sends the generated invoice to user’s email using SMTP.
Logs confirmation of email sent.

# Airport List Retrieval

Fetches and formats available airports for users on request.
