# Conversation UX Patterns

## 1. Clarifying Questions

The chatbot should ask clarifying questions when user input is ambiguous or incomplete. These patterns help refine the search query:

### Location Clarification
**Trigger:** User doesn't specify location
**Pattern:** "I see you're looking for [cuisine/type of food], but I don't have your location. Could you let me know which city or neighborhood you're searching in?"

### Party Size Clarification
**Trigger:** User mentions event but doesn't specify size
**Pattern:** "For a [type of event], knowing how many people you're feeding helps me find the right options. How many guests are you planning for?"

### Dietary Restrictions Clarification
**Trigger:** User mentions dietary needs but they're unclear
**Pattern:** "I heard you mention [dietary need], but could you clarify if you mean [options]? This will help me find the most suitable options."

### Price Range Clarification
**Trigger:** User mentions budget concerns but doesn't specify amount
**Pattern:** "I understand you're looking for affordable options. Could you share your approximate budget per person? This helps me narrow down the best matches."

## 2. Disambiguation Strategies

When the system detects multiple possible interpretations of a query:

### Option Presentation
**Pattern:** "I found a few possibilities for your request. Are you looking for:
- [Option 1]: [Description]
- [Option 2]: [Description]
- [Option 3]: [Description]"

### Confirmation Requests
**Pattern:** "Just to confirm, you're looking for [interpreted meaning] for [event/occasion] in [location], correct?"

### Follow-up Clarification
**Pattern:** "I found several [category] options. Would you like me to focus on [specific attribute] or would you prefer to explore [alternative attribute]?"

## 3. Context Carryover

### Explicit Context Setting
**System Response:** "I'll remember that you're looking for [cuisine] options in [location] for [party size] people."

### Implicit Context Recognition
**Pattern:** When a user follows up with "cheaper ones" or "more options," the system should remember the previous context (location, party size, dietary restrictions).

### Context Correction
**Pattern:** "I noticed you mentioned [new info] which differs from what I had noted earlier ([old info]). Should I update your preferences?"

## 4. Error Recovery

### Unclear Queries
**Pattern:** "I'm not quite sure what you're looking for. Could you try rephrasing? Perhaps you could specify the type of cuisine, location, or occasion you have in mind?"

### No Results Found
**Pattern:** "I couldn't find exact matches for [query]. Would you like me to broaden the search to include [related options] or adjust any of your criteria?"

### System Limitations
**Pattern:** "Currently, I can help you find restaurants and catering options. For [out-of-scope request], you might need to contact the restaurant directly. Can I help with anything else related to finding the right restaurant?"

## 5. Progressive Disclosure

### Initial Response
**Pattern:** Start with a concise summary and top 3-5 results, then offer "Would you like to see more options?" or "Tell me more about what you're looking for."

### Detailed Information
**Pattern:** When user shows interest in a specific result, provide more details: "Great choice! Here's more about [restaurant]: [details]. Would you like to see their full menu or compare with similar options?"

## 6. Conversational Flow Management

### Acknowledgment
**Pattern:** "Got it!" or "I'll look for [summary of request]" before processing the query.

### Status Updates
**Pattern:** "Looking for [cuisine] options in [location]..." while processing.

### Transition Handling
**Pattern:** "I found [number] great options for you. Let me show you the top matches:" before displaying results.

## 7. Personalization Cues

### Preference Learning
**Pattern:** "I notice you often look for [pattern in user's requests]. Would you like me to prioritize [similar options] in future searches?"

### History Reference
**Pattern:** "Based on your previous search for [previous query], I thought you might also like [related suggestion]."

## 8. Validation and Confirmation

### Important Detail Verification
**Pattern:** "Just to confirm, you're looking for [expensive item] for [large party]? That sounds like a wonderful event!"

### Action Confirmation
**Pattern:** "Before I search for [query], is this what you're looking for, or would you like to adjust anything?"