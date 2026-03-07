# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                             │
│                     http://localhost:3000                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP/WS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Next.js 14 Frontend                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   App Router (Chat Page)                  │  │
│  │  ┌──────────────────┐         ┌──────────────────────┐   │  │
│  │  │   ChatWindow     │◄───────►│    OrderPanel        │   │  │
│  │  │   Component      │         │   (Collapsible)      │   │  │
│  │  │                  │         │                      │   │  │
│  │  │  - MessageBubble │         │  - Cart Items        │   │  │
│  │  │  - MenuCard      │         │  - Quantity Ctrl     │   │  │
│  │  │  - OfferCard     │         │  - Order Summary     │   │  │
│  │  │  - TypingInd.    │         │  - Checkout          │   │  │
│  │  └──────────────────┘         └──────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Component Library                       │  │
│  │  shadcn/ui: Button, Card, Input, Badge, Avatar, etc.     │  │
│  │  Framer Motion: Animations & Transitions                 │  │
│  │  React Markdown: Message Formatting                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ REST API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Hybrid Search v2 Backend                        │
│                   FastAPI (Port 8000)                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    API Endpoints                          │  │
│  │  POST /chat/search        - Conversational search         │  │
│  │  GET  /session/{id}       - Get session state             │  │
│  │  DELETE /session/{id}     - Clear session                 │  │
│  │  POST /session/{id}/fb    - Submit feedback               │  │
│  │  GET  /health             - Health check                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  LangGraph Pipeline                       │  │
│  │  Context → Intent → Query → Search → Fusion → RAG → Answer│  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Search Engines                          │  │
│  │  ┌──────────────┐      ┌──────────────┐                  │  │
│  │  │   BM25       │      │    Vector    │                  │  │
│  │  │ (OpenSearch) │      │  (pgvector)  │                  │  │
│  │  │   Port 9200  │      │   Port 5433  │                  │  │
│  │  └──────────────┘      └──────────────┘                  │  │
│  │         │                      │                          │  │
│  │         └──────────┬───────────┘                          │  │
│  │                    ▼                                      │  │
│  │           ┌─────────────────┐                             │  │
│  │           │  RRF Fusion     │                             │  │
│  │           │  (k=60, w=1.0)  │                             │  │
│  │           └─────────────────┘                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│     Redis     │   │   OpenAI      │   │   Neo4j       │
│   Session     │   │  Embeddings   │   │   Graph       │
│   Storage     │   │   & LLM       │   │  (Optional)   │
│   Port 6379   │   │   API         │   │  Port 7687    │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Chat Page                               │
│                   (app/chat/page.tsx)                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   ChatWindow                           │ │
│  │  ┌────────────────┐        ┌──────────────────────┐   │ │
│  │  │   Header       │        │   Messages Area      │   │ │
│  │  │  - Logo        │        │   ┌──────────────┐   │   │ │
│  │  │  - Title       │        │   │MessageBubble │   │   │ │
│  │  │  - Badge       │        │   │ - Avatar     │   │   │ │
│  │  └────────────────┘        │   │ - Content    │   │   │ │
│  │                            │   │ - MenuCards  │   │   │ │
│  │  ┌────────────────────────┐│   │ - QuickRepl. │   │   │ │
│  │  │   Input Area           ││   │ ┌──────────┐ │   │   │ │
│  │  │  - Text Input          ││   │ │MenuCard  │ │   │   │ │
│  │  │  - Voice Button        ││   │ │- Image   │ │   │   │ │
│  │  │  - Send Button         ││   │ │- Price   │ │   │   │ │
│  │  └────────────────────────┘│   │ └──────────┘ │   │   │ │
│  │                            │   └──────────────┘   │   │ │
│  │                            │   ┌──────────────┐   │   │ │
│  │                            │   │TypingIndicator│  │   │ │
│  │                            │   └──────────────┘   │   │ │
│  │                            └──────────────────────┘   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   OrderPanel                           │ │
│  │  ┌────────────────┐        ┌──────────────────────┐   │ │
│  │  │   Header       │        │   Cart Items         │   │ │
│  │  │  - Title       │        │   ┌──────────────┐   │   │ │
│  │  │  - Count       │        │   │ Item Card    │   │   │ │
│  │  │  - Collapse    │        │   │ - Image      │   │   │ │
│  │  └────────────────┘        │   │ - Qty Ctrl   │   │   │ │
│  │                            │   │ - Remove     │   │   │ │
│  │  ┌────────────────────────┐│   └──────────────┘   │   │ │
│  │  │   Footer               ││                      │   │ │
│  │  │  - Subtotal            ││   ┌──────────────┐   │   │ │
│  │  │  - Tax                 ││   │ Empty State  │   │   │ │
│  │  │  - Total               ││   └──────────────┘   │   │ │
│  │  │  - Checkout Button     ││                      │   │ │
│  │  └────────────────────────┘└──────────────────────┘   │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User Input
    │
    ▼
┌─────────────────┐
│  ChatWindow     │
│  handleSubmit() │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  executeSearch()│◄───────┐
│  (lib/api.ts)   │        │
└────────┬────────┘        │
         │                 │ Session ID
         ▼                 │ (localStorage)
┌─────────────────┐        │
│  POST /chat/    │        │
│  search         │────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  LangGraph Pipeline             │
│  1. Context Resolution          │
│  2. Intent Detection            │
│  3. Query Rewriting             │
│  4. Parallel Search (BM25+Vec)  │
│  5. RRF Fusion                  │
│  6. RAG Generation              │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  SearchResponse │
│  - answer       │
│  - results[]    │
│  - filters      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChatWindow     │
│  Update State   │
└────────┬────────┘
         │
         ├──────────────┬─────────────┐
         ▼              ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌──────────┐
│MessageBubble│ │ MenuCard[]  │ │  Toast   │
│ - answer    │ │ - results   │ │  Notify  │
└─────────────┘ └─────────────┘ └──────────┘
```

## State Management

```
┌──────────────────────────────────────────────────────┐
│                   Application State                  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────┐      ┌──────────────────┐     │
│  │  Component State │      │   Local Storage  │     │
│  │  (React useState)│      │                  │     │
│  │                  │      │  - session_id    │     │
│  │  - messages[]    │      │  - cart items    │     │
│  │  - cartItems[]   │      │                  │     │
│  │  - isLoading     │      └──────────────────┘     │
│  │  - isCollapsed   │                                │
│  └──────────────────┘      ┌──────────────────┐     │
│                            │   Backend Redis  │     │
│                            │                  │     │
│                            │  - Session data  │     │
│                            │  - Conversation  │     │
│                            │  - Entities      │     │
│                            │  - TTL: 24h      │     │
│                            └──────────────────┘     │
└──────────────────────────────────────────────────────┘
```

## Security Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Security Layers                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Frontend (Next.js)                         │   │
│  │  - Input sanitization (React Markdown)      │   │
│  │  - XSS protection (React auto-escape)       │   │
│  │  - CORS configured                          │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Backend (FastAPI) - REQUIRES FIXES         │   │
│  │  ⚠️  No auth on /chat/search                │   │
│  │  ⚠️  Prompt injection vulnerability          │   │
│  │  ⚠️  SQL injection risk in vector search    │   │
│  │  ⚠️  No rate limiting                       │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Recommended Fixes                          │   │
│  │  ✅ Add JWT middleware to /chat/search      │   │
│  │  ✅ Separate system/user messages           │   │
│  │  ✅ Parameterized queries                   │   │
│  │  ✅ Rate limiting (Redis-based)             │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌──────────────────────────────────────────────────────┐
│                 Production Deployment                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────┐         ┌────────────────┐      │
│  │   Vercel /     │         │   Docker       │      │
│  │   Next.js      │         │   Backend      │      │
│  │   Frontend     │         │   (FastAPI)    │      │
│  │   (CDN)        │         │                │      │
│  └────────────────┘         └────────────────┘      │
│         │                          │                 │
│         │                          │                 │
│         └──────────┬───────────────┘                 │
│                    │                                 │
│         ┌──────────▼───────────┐                     │
│         │   Infrastructure     │                     │
│         │   (Docker Compose)   │                     │
│         │                      │                     │
│         │  - OpenSearch        │                     │
│         │  - PostgreSQL        │                     │
│         │  - Redis             │                     │
│         │  - Neo4j (optional)  │                     │
│         └──────────────────────┘                     │
└──────────────────────────────────────────────────────┘
```

## File Organization

```
chatbot-ui/
│
├── src/
│   ├── app/
│   │   ├── chat/
│   │   │   └── page.tsx              # Main chat interface
│   │   ├── globals.css               # Global styles
│   │   ├── layout.tsx                # Root layout
│   │   └── page.tsx                  # Home redirect
│   │
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatWindow.tsx        # Chat container (236 lines)
│   │   │   ├── MessageBubble.tsx     # Message display (148 lines)
│   │   │   ├── MenuCard.tsx          # Food item card (131 lines)
│   │   │   ├── OfferCard.tsx         # Promo banner (92 lines)
│   │   │   ├── OrderPanel.tsx        # Cart sidebar (238 lines)
│   │   │   └── TypingIndicator.tsx   # Typing animation (60 lines)
│   │   │
│   │   └── ui/                       # shadcn/ui components
│   │       ├── avatar.tsx
│   │       ├── badge.tsx
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── input.tsx
│   │       ├── scroll-area.tsx
│   │       ├── skeleton.tsx
│   │       └── sonner.tsx
│   │
│   ├── lib/
│   │   ├── api.ts                    # API integration (150 lines)
│   │   └── utils.ts                  # Utilities (84 lines)
│   │
│   └── types/
│       └── api.ts                    # TypeScript types (70 lines)
│
├── public/                           # Static assets
│
├── .env.example                      # Environment template
├── components.json                   # shadcn config
├── next.config.mjs                   # Next.js config
├── package.json                      # Dependencies
├── tailwind.config.ts                # Tailwind config
├── tsconfig.json                     # TypeScript config
│
└── Documentation/
    ├── README.md                     # User documentation
    ├── IMPLEMENTATION_GUIDE.md       # Developer guide
    ├── PROJECT_SUMMARY.md            # Project summary
    └── ARCHITECTURE.md               # This file
```

## Technology Stack

```
┌─────────────────────────────────────────────────────┐
│                  Technology Stack                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Frontend                                           │
│  ├─ Next.js 14 (App Router)                        │
│  ├─ React 18                                       │
│  ├─ TypeScript 5                                   │
│  ├─ Tailwind CSS 3                                 │
│  ├─ Framer Motion                                  │
│  ├─ React Markdown                                 │
│  └─ shadcn/ui                                      │
│                                                     │
│  Backend (Hybrid Search v2)                         │
│  ├─ FastAPI                                        │
│  ├─ LangGraph                                      │
│  ├─ OpenSearch (BM25)                              │
│  ├─ PostgreSQL + pgvector (Vector)                 │
│  ├─ Redis (Session/Cache)                          │
│  ├─ Neo4j (Graph - optional)                       │
│  └─ OpenAI (Embeddings + LLM)                      │
│                                                     │
│  Development                                        │
│  ├─ ESLint                                         │
│  ├─ Prettier                                       │
│  ├─ Jest (testing - recommended)                   │
│  └─ Playwright (E2E - recommended)                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```
