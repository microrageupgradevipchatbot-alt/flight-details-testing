# Code Flow

->First msg by bot "Welcome " etc on frontend react.
-> user enters message
-> user_msg goes to langgraph agent
-> backend generates a unique session id of that user
-> now that session id chat_history will be remembered by Agent
-> Agent will decide based on 'system prompt' which tool to be called
-> if its related to booking

- whole booking info is in system prompt so no tool will be called
- bot start taking booking info from user directly
- firstly ask his interest
  - airport vip/transfer
- secondly will ask flight no. and date in MM/DD/YYYY format
- then call the flight tools and take info in json from flight details endpoint
- then call format function of flight details to show response in better way
- then Agent will reply the exact response came format function
- Now Agent ask departure/arrival
- flight class (first,economy/plus,business)
- no.of adults and no. of luggage
- preferred currency
- if user interest was airport vip
  - Agent will call get_airport_vip_services cards tools
    & then its format tool and show the user
- if transfer
  - Agent call transfer services cards tool
    & then its format tool and the response goes to the agent and it will show that to the user
- after user selects card by 'number' or by 'title'
- Agent will ask for preferred time
- msg for steward
- address only if his interest was 'transfer service'
- ask for email
- ask if user wants to book other service too
  - if yes it will ask the booking info of it and generate a combine invoice
  - else:
    - will call the invoice tool - present the invoice - when user confirms the invoice - send the invoice on the email he provided

-> if user ask questions about upgradevip

- bot will call rag_tool
  And answer the user questions i.e about us,policy,servicesetc
  Note: if he ask any question which is out of the scope
  i.e salary of elon,capitol of france
  Bot will simply refuse to answer by saying he don't know anything about that he can help you about giving upgradevip only
  -> if user asks for the list of airports where you provide services
- Agent will call the get_airportList tool
  which will hit the endpoint and take json response
  then tool for format_airport_list called to format it
  and response given to the agent
  and agent will present that to the user
