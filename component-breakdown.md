# Component Breakdown with Props

## 1. ChatWindow
Container component for the entire chat interface.

Props:
- `sessionId` (string): Unique identifier for the chat session
- `onSessionChange` (function): Callback when session changes
- `className` (string): Additional CSS classes

## 2. MessageList
Displays the list of chat messages.

Props:
- `messages` (array): Array of message objects
- `isLoading` (boolean): Whether to show loading indicator
- `className` (string): Additional CSS classes

## 3. MessageBubble
Represents a single message in the chat.

Props:
- `role` (string): 'user' or 'assistant'
- `content` (string | JSX.Element): Message content
- `timestamp` (Date): When the message was sent
- `className` (string): Additional CSS classes

## 4. RestaurantCard
Displays restaurant information in a card format.

Props:
- `restaurant` (object): Restaurant data object
  - `doc_id` (string): Unique document ID
  - `restaurant_name` (string): Name of the restaurant
  - `city` (string): City location
  - `state` (string): State location
  - `item_name` (string): Name of the menu item
  - `item_description` (string): Description of the item
  - `display_price` (number): Price to display
  - `price_per_person` (number): Price per person
  - `serves_min` (number): Minimum servings
  - `serves_max` (number): Maximum servings
  - `dietary_labels` (array): Dietary labels (e.g., vegetarian, vegan)
  - `tags` (array): Tags for the item
  - `rrf_score` (number): Relevance score
- `onClick` (function): Handler when card is clicked
- `className` (string): Additional CSS classes

## 5. InputArea
Handles user input for the chat.

Props:
- `onSend` (function): Handler when user submits a message
- `disabled` (boolean): Whether input is disabled
- `placeholder` (string): Placeholder text for input
- `className` (string): Additional CSS classes

## 6. PromptSuggestions
Displays suggested prompts to help users start conversations.

Props:
- `prompts` (array): Array of suggested prompt strings
- `onSelect` (function): Handler when a prompt is selected
- `className` (string): Additional CSS classes

## 7. LoadingIndicator
Shows loading state when waiting for API response.

Props:
- `message` (string): Message to display while loading
- `className` (string): Additional CSS classes

## 8. ErrorMessage
Displays error messages to the user.

Props:
- `message` (string): Error message to display
- `onRetry` (function): Handler when user wants to retry
- `className` (string): Additional CSS classes

## 9. EmptyResultsMessage
Displays when search returns no results.

Props:
- `query` (string): The query that returned no results
- `onTryAgain` (function): Handler when user wants to try a different query
- `className` (string): Additional CSS classes

## 10. ContextProvider
Provides context state for the application.

Props:
- `initialContext` (object): Initial context values
  - `location` (string): Current location context
  - `partySize` (number): Current party size context
  - `dietaryPreferences` (array): Dietary preferences
- `children` (ReactNode): Child components