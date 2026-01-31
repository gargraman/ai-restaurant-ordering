# Session & Feedback Endpoints Implementation Plan

**Date:** February 1, 2026  
**Status:** Ready for Implementation  
**Priority:** High Value Features for Production

---

## 📋 Executive Summary

This document outlines the implementation plan for connecting three unused backend endpoints to the UI:
- **Session DELETE** - Clear/reset current conversation
- **Feedback POST** - Rate restaurant recommendations  
- **Session GET** - Retrieve and restore session context

All three endpoints are already implemented in the API client but not connected to UI components.

---

## 🎯 Current State Analysis

### ✅ What We Have
- API client functions implemented in [ui/src/lib/api-client.js](ui/src/lib/api-client.js):
  - `getSession(sessionId)`
  - `deleteSession(sessionId)`
  - `submitFeedback(sessionId, docId, rating, comment)`
- Backend endpoints are functional
- Session management infrastructure exists

### ❌ What's Missing
- UI components to trigger these endpoints
- State management for session info and feedback
- User-facing controls (buttons, forms)
- Analytics tracking for these interactions

---

## 🚀 Implementation Phases

### **Phase 1: Session DELETE (New Chat)** 🎯
**Priority:** 🟢 High  
**Effort:** Low  
**Impact:** High  
**Timeline:** Week 1

#### Purpose
Allow users to start a fresh conversation and clear current context.

#### User Stories
- As a user, I want to start a new chat so I can search for different restaurants
- As a user, I want to clear my search history for privacy
- As a user, I want to reset the conversation context when it's no longer relevant

#### Implementation

**Files to Modify:**
1. `ui/src/contexts/ChatContext.js` - Add session reset action
2. `ui/src/components/ChatWindow.js` - Add reset handler
3. `ui/src/app/page.js` - Add "New Chat" button to header
4. `ui/src/lib/analytics/tracker.js` - Add session reset tracking

**Code Changes:**

**1. ChatContext.js - Add Reset Session Action**
```javascript
// Add to reducer actions
case 'RESET_SESSION':
  return {
    ...initialState,
    // Keep only the essential initial state
  };

// Add to provider actions
const resetSession = () => {
  dispatch({ type: 'RESET_SESSION' });
};
```

**2. page.js - Add New Chat Button**
```javascript
<header className="py-3 px-4 sm:py-4 sm:px-6 border-b border-gray-200 bg-white shadow-sm">
  <div className="max-w-4xl mx-auto flex items-center justify-between">
    <h1 className="text-lg sm:text-xl font-bold text-gray-800">Restaurant Discovery Chat</h1>
    <div className="flex items-center space-x-3">
      <button
        onClick={handleNewChat}
        className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1 rounded-lg hover:bg-gray-100 transition-colors"
        aria-label="Start new chat"
      >
        + New Chat
      </button>
      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">Beta</span>
    </div>
  </div>
</header>
```

**3. ChatWindow.js - Add Handler**
```javascript
const handleNewChat = async () => {
  if (messages.length <= 1) {
    // Only welcome message, no need to confirm
    return;
  }

  const confirmed = window.confirm(
    'Start a new chat? This will clear your current conversation.'
  );
  
  if (!confirmed) return;

  try {
    setIsLoading(true);
    
    // Delete session on backend
    await deleteSession(sessionId);
    
    // Track analytics
    analyticsTracker.trackSessionReset(sessionId, messages.length);
    
    // Reset local state
    setMessages([]);
    setError(null);
    resetContext();
    
    // Generate new session ID
    const newSessionId = uuidv4();
    localStorage.setItem('chatSessionId', newSessionId);
    
    // Reload page with new session
    window.location.reload();
    
  } catch (error) {
    setError('Unable to reset session. Please refresh the page.');
  } finally {
    setIsLoading(false);
  }
};
```

**4. analytics/tracker.js - Add Tracking**
```javascript
trackSessionReset(sessionId, messageCount) {
  console.log('[Analytics] Session reset:', { sessionId, messageCount });
  this.logEvent('session_reset', { 
    sessionId, 
    messageCount,
    timestamp: new Date().toISOString()
  });
}
```

#### UI/UX Design

**New Chat Button Location:**
```
┌─────────────────────────────────────────────────┐
│ Restaurant Discovery Chat     + New Chat   Beta │
└─────────────────────────────────────────────────┘
```

**Confirmation Dialog:**
```
╔═════════════════════════════════════════════════╗
║  Start a new conversation?                      ║
║                                                 ║
║  Your current chat will be cleared.             ║
║                                                 ║
║  [Cancel]           [Start New Chat]            ║
╚═════════════════════════════════════════════════╝
```

#### Acceptance Criteria
- [ ] "New Chat" button visible in header on all screen sizes
- [ ] Button disabled when already on empty chat
- [ ] Clicking shows confirmation dialog (if chat has messages)
- [ ] Confirming clears messages and resets context
- [ ] New session ID generated and stored
- [ ] Backend session deleted successfully
- [ ] Error handling shows user-friendly message
- [ ] Analytics tracks reset events
- [ ] Mobile responsive design works

#### Edge Cases
- User clicks during API call → Disable button while resetting
- Backend error → Show error message, allow retry
- No internet → Graceful error with refresh suggestion
- Multiple rapid clicks → Debounce button

---

### **Phase 2: Feedback POST (Rate Results)** ⭐⭐
**Priority:** 🟢 High  
**Effort:** Medium  
**Impact:** High (Critical for search quality improvement)  
**Timeline:** Week 1

#### Purpose
Collect user feedback on restaurant recommendations to improve search quality.

#### User Stories
- As a user, I want to rate restaurant results so the system learns my preferences
- As a user, I want to give quick feedback without disrupting my search flow
- As a product owner, I want to collect feedback data to improve the search algorithm

#### Implementation

**Files to Modify:**
1. `ui/src/components/RestaurantCard.js` - Add feedback UI
2. `ui/src/lib/analytics/tracker.js` - Add feedback tracking
3. `ui/src/lib/api-client.js` - (Already implemented)

**Code Changes:**

**1. RestaurantCard.js - Add Feedback Component**
```javascript
import { useState } from 'react';
import { ThumbsUpIcon, ThumbsDownIcon, CheckCircleIcon } from 'lucide-react';

const RestaurantCard = ({ restaurant, sessionId }) => {
  const [feedbackStatus, setFeedbackStatus] = useState(null); // null | 'submitting' | 'submitted' | 'error'
  const [feedbackType, setFeedbackType] = useState(null); // 'positive' | 'negative'

  const handleFeedback = async (rating) => {
    if (feedbackStatus === 'submitted') return; // Already submitted
    
    setFeedbackStatus('submitting');
    setFeedbackType(rating === 5 ? 'positive' : 'negative');
    
    try {
      await submitFeedback(sessionId, doc_id, rating);
      setFeedbackStatus('submitted');
      analyticsTracker.trackFeedbackSubmitted(doc_id, restaurant_name, rating, sessionId);
    } catch (error) {
      setFeedbackStatus('error');
      analyticsTracker.trackFeedbackFailed(doc_id, error.message, sessionId);
    }
  };

  const renderFeedbackUI = () => {
    if (feedbackStatus === 'submitted') {
      return (
        <div className="flex items-center justify-center gap-2 text-green-700 text-sm">
          <CheckCircleIcon className="w-4 h-4" />
          <span>Thanks for your feedback!</span>
        </div>
      );
    }

    if (feedbackStatus === 'error') {
      return (
        <div className="flex items-center justify-between text-sm">
          <span className="text-red-600">Couldn't submit feedback</span>
          <button
            onClick={() => setFeedbackStatus(null)}
            className="text-red-600 underline hover:text-red-800"
          >
            Retry
          </button>
        </div>
      );
    }

    return (
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">Was this helpful?</span>
        <div className="flex gap-2">
          <button
            onClick={() => handleFeedback(5)}
            disabled={feedbackStatus === 'submitting'}
            className="p-2 rounded-lg hover:bg-green-100 transition-colors disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-green-500"
            aria-label="This was helpful"
          >
            <ThumbsUpIcon className="w-4 h-4 text-green-600" />
          </button>
          <button
            onClick={() => handleFeedback(1)}
            disabled={feedbackStatus === 'submitting'}
            className="p-2 rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-red-500"
            aria-label="This was not helpful"
          >
            <ThumbsDownIcon className="w-4 h-4 text-red-600" />
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-50 rounded-xl p-4 border border-gray-200 hover:border-orange-300 transition-colors w-full">
      {/* Existing card content */}
      
      {/* Add feedback section at bottom */}
      <div className="mt-3 pt-3 border-t border-gray-200">
        {renderFeedbackUI()}
      </div>
    </div>
  );
};
```

**2. analytics/tracker.js - Add Tracking**
```javascript
trackFeedbackSubmitted(docId, restaurantName, rating, sessionId) {
  console.log('[Analytics] Feedback submitted:', { 
    docId, 
    restaurantName, 
    rating, 
    sessionId 
  });
  
  this.logEvent('feedback_submitted', { 
    docId, 
    restaurantName, 
    rating,
    ratingType: rating >= 3 ? 'positive' : 'negative',
    sessionId,
    timestamp: new Date().toISOString()
  });
}

trackFeedbackFailed(docId, errorMessage, sessionId) {
  console.error('[Analytics] Feedback failed:', { 
    docId, 
    errorMessage, 
    sessionId 
  });
  
  this.logEvent('feedback_failed', { 
    docId, 
    errorMessage, 
    sessionId,
    timestamp: new Date().toISOString()
  });
}
```

#### UI/UX Design

**Initial State:**
```
┌───────────────────────────────────────────────┐
│ QDOBA Mexican Eats                      4.5⭐ │
│ Boston, MA                                    │
│                                               │
│ Chicken Burrito Boxed Lunch                   │
│ $12.25                      [View Menu]       │
├───────────────────────────────────────────────┤
│ Was this helpful?              👍  👎         │
└───────────────────────────────────────────────┘
```

**After Positive Feedback:**
```
├───────────────────────────────────────────────┤
│        ✓ Thanks for your feedback!            │
└───────────────────────────────────────────────┘
```

**Error State:**
```
├───────────────────────────────────────────────┤
│ Couldn't submit feedback            Retry     │
└───────────────────────────────────────────────┘
```

#### Acceptance Criteria
- [ ] Thumbs up/down buttons visible on each restaurant card
- [ ] Buttons have proper hover and focus states
- [ ] Clicking submits feedback to backend (rating: 5 for 👍, 1 for 👎)
- [ ] UI shows loading state while submitting
- [ ] Success message displayed after submission
- [ ] Error state with retry option
- [ ] Can only submit feedback once per card per session
- [ ] Analytics tracks all feedback events (success and failure)
- [ ] Accessible via keyboard navigation
- [ ] Mobile-friendly tap targets (min 44x44px)

#### Rating Scale Mapping
- 👍 Thumbs Up → Rating: 5 (Very helpful)
- 👎 Thumbs Down → Rating: 1 (Not helpful)

#### Edge Cases
- Network error during submission → Show retry option
- Double-click prevention → Disable after first click
- Session expired → Handle gracefully with error message
- Rapid feedback on multiple cards → Each tracked independently

---

### **Phase 3: Session GET (Context Restoration)** 
**Priority:** 🟡 Medium  
**Effort:** Low  
**Impact:** Medium (Enhancement, not critical)  
**Timeline:** Week 2

#### Purpose
Restore session context when user returns to the app or refreshes the page.

#### User Stories
- As a user, I want my search context to persist when I refresh the page
- As a user, I want to see my previous conversation history (future enhancement)
- As a user, I want the app to remember my location and preferences

#### Implementation

**Files to Modify:**
1. `ui/src/contexts/ChatContext.js` - Add session info state
2. `ui/src/components/ChatWindow.js` - Fetch session on mount
3. *(Optional)* `ui/src/components/SessionInfo.js` - Display session details

**Code Changes:**

**1. ChatContext.js - Add Session Info State**
```javascript
const initialState = {
  messages: [],
  isLoading: false,
  error: null,
  sessionInfo: null, // NEW: Store session metadata
  context: {
    location: '',
    partySize: null,
    dietaryPreferences: []
  }
};

// Add reducer case
case 'SET_SESSION_INFO':
  return { ...state, sessionInfo: action.payload };

// Add action
const setSessionInfo = (info) => {
  dispatch({ type: 'SET_SESSION_INFO', payload: info });
};
```

**2. ChatWindow.js - Fetch Session on Mount**
```javascript
useEffect(() => {
  const fetchSessionInfo = async () => {
    try {
      const sessionData = await getSession(sessionId);
      
      // Update context with session entities
      if (sessionData.entities) {
        const updates = {};
        
        if (sessionData.entities.location) {
          updates.location = sessionData.entities.location;
        }
        if (sessionData.entities.party_size) {
          updates.partySize = sessionData.entities.party_size;
        }
        if (sessionData.entities.dietary_restrictions) {
          updates.dietaryPreferences = sessionData.entities.dietary_restrictions;
        }
        
        if (Object.keys(updates).length > 0) {
          updateContext(updates);
        }
      }
      
      setSessionInfo(sessionData);
      
    } catch (error) {
      // Fail silently - session may not exist yet
      console.warn('Could not fetch session info:', error);
    }
  };
  
  if (sessionId) {
    fetchSessionInfo();
  }
}, [sessionId]);
```

**3. (Optional) SessionInfo.js - Display Component**
```javascript
const SessionInfo = ({ sessionInfo, context }) => {
  if (!context.location && !context.partySize && context.dietaryPreferences.length === 0) {
    return null; // Nothing to show
  }

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
      <p className="text-xs text-blue-600 font-medium mb-2">Your Search Context:</p>
      <div className="flex flex-wrap gap-2">
        {context.location && (
          <span className="text-xs bg-white text-blue-800 px-2 py-1 rounded-full">
            📍 {context.location}
          </span>
        )}
        {context.partySize && (
          <span className="text-xs bg-white text-blue-800 px-2 py-1 rounded-full">
            👥 {context.partySize} people
          </span>
        )}
        {context.dietaryPreferences.map((pref, idx) => (
          <span key={idx} className="text-xs bg-white text-blue-800 px-2 py-1 rounded-full">
            🥗 {pref}
          </span>
        ))}
      </div>
    </div>
  );
};
```

#### UI/UX Design

**Session Context Display (Optional):**
```
┌───────────────────────────────────────────────┐
│  Your Search Context:                         │
│  📍 Boston   👥 10 people   🥗 Vegetarian     │
└───────────────────────────────────────────────┘
```

#### Acceptance Criteria
- [ ] Session info fetched on component mount
- [ ] Context restored from session entities if available
- [ ] Errors fail silently without breaking UI
- [ ] No loading spinner shown (background operation)
- [ ] Optional: Session info displayed to user
- [ ] Session info updates when context changes
- [ ] Works with new and existing sessions

#### Edge Cases
- Session doesn't exist (new user) → Fail silently
- Session expired → Handle gracefully
- Network error → Don't block main UI
- Partial session data → Use what's available

---

## 📊 Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Dependencies | Timeline |
|---------|----------|--------|--------|--------------|----------|
| **Session DELETE** | 🟢 High | Low | High | None | Week 1, Day 1-2 |
| **Feedback POST** | 🟢 High | Medium | High | None | Week 1, Day 3-5 |
| **Session GET** | 🟡 Medium | Low | Medium | None | Week 2, Day 1-2 |
| **Session Info UI** | 🟡 Low | Medium | Low | Session GET | Future |

---

## 🎨 Design System Integration

### Colors
- **Success (Feedback):** `green-600`, `green-100`
- **Error:** `red-600`, `red-50`
- **Info (Session):** `blue-600`, `blue-50`
- **Neutral:** Existing gray scale

### Icons (lucide-react)
- ✅ `ThumbsUpIcon` - Positive feedback
- ✅ `ThumbsDownIcon` - Negative feedback
- ✅ `CheckCircleIcon` - Success state
- ✅ `XCircleIcon` - Error state
- ✅ `RefreshCwIcon` - Reset/Retry

### Typography
- Button text: `text-sm`
- Feedback text: `text-xs`
- Session context: `text-xs`

---

## 🔧 Technical Considerations

### Error Handling

**Session DELETE:**
```javascript
try {
  await deleteSession(sessionId);
} catch (error) {
  setError('Unable to reset session. Please refresh the page.');
}
```

**Feedback POST:**
```javascript
try {
  await submitFeedback(sessionId, docId, rating);
} catch (error) {
  setFeedbackStatus('error');
  // Allow retry
}
```

**Session GET:**
```javascript
try {
  const sessionData = await getSession(sessionId);
} catch (error) {
  // Fail silently - don't block UI
  console.warn('Session fetch failed:', error);
}
```

### Race Conditions

**Prevent Multiple Session Resets:**
```javascript
const [isResetting, setIsResetting] = useState(false);

const handleReset = async () => {
  if (isResetting) return;
  setIsResetting(true);
  try {
    await deleteSession(sessionId);
  } finally {
    setIsResetting(false);
  }
};
```

**Prevent Multiple Feedback Submissions:**
```javascript
if (feedbackStatus === 'submitted' || feedbackStatus === 'submitting') {
  return; // Already handled
}
```

### Performance Optimization

**Debouncing:**
- New Chat button → Debounce 300ms
- Feedback buttons → Immediate (optimistic UI)

**Lazy Loading:**
- Session info fetched in background (non-blocking)

**Caching:**
- Consider caching session info for 5 minutes
- Invalidate on context changes

---

## 📈 Analytics Events

### New Events to Track

```javascript
// Session Management
trackSessionReset(sessionId, messageCount)
trackSessionRestored(sessionId, conversationLength)

// Feedback
trackFeedbackSubmitted(docId, restaurantName, rating, sessionId)
trackFeedbackFailed(docId, errorMessage, sessionId)

// Engagement
trackNewChatClicked(sessionId)
trackSessionInfoViewed(sessionId)
```

### Analytics Dashboard Metrics
- Session reset rate
- Feedback submission rate (% of results rated)
- Positive vs negative feedback ratio
- Average session length before reset
- Error rates for each endpoint

---

## 🧪 Testing Checklist

### Manual Testing

**Session DELETE:**
- [ ] Button appears in header
- [ ] Confirmation dialog works
- [ ] Session cleared on backend
- [ ] Local state reset
- [ ] New session ID generated
- [ ] Error handling works
- [ ] Mobile responsive

**Feedback POST:**
- [ ] Thumbs up/down buttons appear
- [ ] Clicking submits to backend
- [ ] Success message displays
- [ ] Error state with retry
- [ ] Cannot submit twice
- [ ] Analytics tracked
- [ ] Keyboard accessible

**Session GET:**
- [ ] Fetches on mount
- [ ] Context restored
- [ ] Fails silently on error
- [ ] Works with new sessions

### Edge Case Testing
- [ ] Offline mode behavior
- [ ] Rapid button clicking
- [ ] Session expiration
- [ ] Network timeout
- [ ] Concurrent operations

---

## 🚀 Deployment Strategy

### Phase 1 Deployment (Week 1)
1. Deploy Session DELETE feature
2. Monitor error rates and user adoption
3. A/B test confirmation dialog (optional)

### Phase 2 Deployment (Week 1)
1. Deploy Feedback POST feature
2. Monitor submission rates
3. Collect initial feedback data

### Phase 3 Deployment (Week 2)
1. Deploy Session GET feature
2. Monitor context restoration success rate
3. Evaluate need for Session Info UI

### Rollback Plan
- All features are additive (no breaking changes)
- Can disable via feature flags if needed
- Backend endpoints remain functional

---

## 📝 Success Metrics

### Session DELETE
- **Target:** 30% of users use "New Chat" feature
- **Metric:** Session reset rate
- **Tracking:** Analytics event `session_reset`

### Feedback POST
- **Target:** 40% of results receive feedback
- **Metric:** Feedback submission rate
- **Tracking:** Ratio of `feedback_submitted` to `result_clicked`

### Session GET
- **Target:** 80% successful context restoration
- **Metric:** Session fetch success rate
- **Tracking:** Backend logs + client-side errors

---

## 🔄 Future Enhancements

### Post-MVP Ideas
1. **Rich Feedback** - Allow users to add comments with ratings
2. **Feedback History** - Show user their previous feedback
3. **Session History** - Browse previous conversations
4. **Context Suggestions** - Suggest context based on time/location
5. **Export Chat** - Allow users to save/share conversation
6. **Voice Feedback** - Quick voice note with rating
7. **Smart Reset** - Preserve relevant context when resetting
8. **Session Analytics** - Show user their search patterns

---

## 👥 Stakeholders

### Development Team
- Frontend: UI implementation
- Backend: Ensure endpoints are production-ready
- Design: Review UI/UX mockups
- QA: Test all scenarios

### Product Team
- Define success metrics
- Prioritize features
- User feedback collection

### Analytics Team
- Set up event tracking
- Create dashboard
- Monitor adoption

---

## 📚 References

### Documentation
- [API Contracts](api-contracts.md)
- [Component Breakdown](component-breakdown.md)
- [Conversation UX Patterns](conversation-ux-patterns.md)

### Code Files
- [ui/src/lib/api-client.js](ui/src/lib/api-client.js)
- [ui/src/contexts/ChatContext.js](ui/src/contexts/ChatContext.js)
- [ui/src/components/RestaurantCard.js](ui/src/components/RestaurantCard.js)
- [ui/src/lib/analytics/tracker.js](ui/src/lib/analytics/tracker.js)

### External Resources
- [Lucide React Icons](https://lucide.dev/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)

---

## ✅ Next Steps

1. **Review this plan** with team
2. **Get design approval** for UI mockups
3. **Start with Phase 1** (Session DELETE)
4. **Run usability tests** after each phase
5. **Collect user feedback** and iterate

---

**Document Version:** 1.0  
**Last Updated:** February 1, 2026  
**Owner:** UI Development Team  
**Status:** ✅ Ready for Implementation
