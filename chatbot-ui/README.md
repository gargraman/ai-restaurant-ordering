# Restaurant Catering Chatbot UI

A production-ready, AI-powered chatbot interface for restaurant catering discovery and ordering.

## 🚀 Features

### Core Functionality
- **Conversational Search**: Natural language interface for finding catering options
- **Hybrid Search Integration**: Connects to Hybrid Search v2 backend (BM25 + Vector search)
- **Real-time Cart Management**: Add, update, and remove items with optimistic UI updates
- **Session Persistence**: Conversations persist across page reloads via localStorage

### UI Components
- **Rich Message Bubbles**: User/bot avatars, timestamps, markdown support
- **Menu Cards**: Beautiful food item cards with images, prices, dietary labels
- **Offer Banners**: Promotional cards with discount codes
- **Order Summary Panel**: Collapsible side panel with cart management
- **Typing Indicator**: Animated bot typing feedback
- **Quick Replies**: Suggested questions for easy interaction

### UX Features
- **Smooth Animations**: Framer Motion transitions and micro-interactions
- **Toast Notifications**: Success/error feedback with Sonner
- **Loading States**: Skeleton screens and typing indicators
- **Error Handling**: Graceful error fallbacks with retry options
- **Responsive Design**: Mobile-first, collapses order panel on small screens
- **Accessibility**: ARIA labels, keyboard navigation, focus management

### Technical Stack
- **Next.js 14**: App Router with React Server Components
- **TypeScript**: Full type safety
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: High-quality component library
- **Framer Motion**: Animation library
- **React Markdown**: Formatted message rendering

## 📁 Project Structure

```
chatbot-ui/
├── src/
│   ├── app/
│   │   ├── chat/
│   │   │   └── page.tsx          # Main chat page
│   │   ├── globals.css           # Global styles
│   │   ├── layout.tsx            # Root layout
│   │   └── page.tsx              # Home (redirects to /chat)
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatWindow.tsx    # Main chat container
│   │   │   ├── MessageBubble.tsx # Individual message
│   │   │   ├── MenuCard.tsx      # Food item card
│   │   │   ├── OfferCard.tsx     # Promotional banner
│   │   │   ├── OrderPanel.tsx    # Cart sidebar
│   │   │   └── TypingIndicator.tsx
│   │   └── ui/                   # shadcn/ui components
│   ├── lib/
│   │   ├── api.ts                # API integration layer
│   │   └── utils.ts              # Utility functions
│   └── types/
│       └── api.ts                # TypeScript types
├── .env.example
├── next.config.js
├── tailwind.config.ts
└── package.json
```

## 🛠️ Setup & Installation

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Hybrid Search v2 backend running (see main project README)

### Installation

```bash
# Navigate to chatbot-ui directory
cd chatbot-ui

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Start development server
npm run dev
```

### Environment Variables

Create a `.env.local` file:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🚀 Running the Application

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Production Build

```bash
npm run build
npm start
```

## 🔌 API Integration

The chatbot connects to the Hybrid Search v2 backend:

### Endpoints Used

- `POST /chat/search` - Execute conversational search
- `GET /session/{session_id}` - Retrieve conversation history
- `DELETE /session/{session_id}` - Clear session
- `POST /session/{session_id}/feedback` - Submit result feedback
- `GET /health` - Health check

### Session Management

Sessions are automatically created and managed:
- New session ID generated on first visit
- Stored in localStorage for persistence
- 24-hour TTL on backend (Redis)

## 🎨 Customization

### Theme Colors

Edit `tailwind.config.ts` to customize colors:

```typescript
theme: {
  extend: {
    colors: {
      primary: {
        DEFAULT: "#FF6B35", // Orange brand color
        foreground: "#FFFFFF",
      },
    },
  },
}
```

### Bot Avatar

Update avatar images in components:
- Bot: `TypingIndicator.tsx`, `MessageBubble.tsx`
- User: `MessageBubble.tsx`

### Food Images

Currently using Unsplash placeholder images. Replace `getFoodImage()` in `lib/utils.ts` with your image service.

## 📱 Responsive Design

### Desktop (> 768px)
- Full chat window with side order panel
- All features visible

### Mobile (< 768px)
- Order panel collapses by default
- Toggle button to expand/collapse
- Optimized touch targets

## ♿ Accessibility

- **ARIA Labels**: All interactive elements labeled
- **Keyboard Navigation**: Tab through all controls
- **Focus Management**: Visible focus indicators
- **Screen Reader**: Semantic HTML and live regions
- **Color Contrast**: WCAG AA compliant

## 🧪 Testing

```bash
# Run linting
npm run lint

# Type checking
npm run type-check

# Build verification
npm run build
```

## 🚧 Missing Backend Features

The following features would enhance the chatbot but are not currently in the backend:

### 1. **Cart Management API**
```
POST /cart/items - Add item to cart
GET /cart - Get cart contents
PUT /cart/items/{id} - Update quantity
DELETE /cart/items/{id} - Remove item
```

**Impact**: Currently cart is client-side only. Backend integration would enable:
- Persistent carts across devices
- Order resumption
- Multi-session support

### 2. **Order Creation API**
```
POST /orders - Create order from cart
GET /orders/{id} - Get order details
POST /orders/{id}/cancel - Cancel order
```

**Note**: Partially exists in backend but not fully integrated with chat flow.

### 3. **Payment Integration**
```
POST /payments/intent - Create payment intent
POST /payments/webhook - Handle payment callbacks
```

**Status**: Stripe integration exists but needs chat UI integration.

### 4. **Real-time Order Status**
```
WS /orders/{id}/status - WebSocket for order updates
```

**Impact**: Enable live order tracking in chat.

### 5. **Restaurant Menu API**
```
GET /restaurants/{id}/menu - Get full menu
GET /menu/categories - Get menu categories
```

**Impact**: Better menu browsing beyond search results.

### 6. **User Authentication**
```
POST /auth/login - User login
POST /auth/register - User registration
GET /auth/profile - User profile
```

**Status**: JWT auth exists but not enforced on chat endpoint.

### 7. **Image Upload for Reviews**
```
POST /reviews/{id}/images - Upload review images
```

**Impact**: Enable users to share event photos.

## 📊 Performance Optimizations

- **Lazy Loading**: Images load on demand
- **Code Splitting**: Automatic with Next.js
- **Memoization**: React.memo for expensive components
- **Debouncing**: Search input debouncing
- **Optimistic Updates**: Cart updates before API confirmation

## 🔐 Security Considerations

- **Input Sanitization**: All user input sanitized
- **XSS Protection**: React Markdown sanitizes content
- **CORS**: Configured for allowed origins only
- **Rate Limiting**: Recommended on backend
- **Authentication**: Recommended for production

## 📈 Analytics & Monitoring

Recommended integrations:

```typescript
// Google Analytics
- Page views
- Chat interactions
- Cart additions

// Error Tracking (Sentry)
- API errors
- Client-side errors
- Performance metrics

// Product Analytics (Mixpanel/Amplitude)
- Conversation flows
- Popular menu items
- Conversion rates
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Hybrid Search v2**: Backend search infrastructure
- **shadcn/ui**: Beautiful UI components
- **Vercel**: Next.js creators
- **Unsplash**: Food photography

## 📞 Support

For issues or questions:
- Create an issue on GitHub
- Check existing documentation
- Contact the development team

---

**Built with ❤️ for restaurant catering discovery**
