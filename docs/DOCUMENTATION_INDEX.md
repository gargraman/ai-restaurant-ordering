# Documentation Index

**Complete codebase documentation for the Hybrid Search & RAG system**

---

## 📚 Main Documentation Files

### [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - System Architecture & Execution Flows
**120 sections | 852 lines | 25 KB**

Comprehensive guide to how the system works end-to-end.

**Start here for:**
- Understanding the complete request flow
- Learning how the 12-node LangGraph pipeline works
- Session persistence and conversation handling
- RRF ranking algorithm
- Error handling patterns
- Performance optimization opportunities
- Monitoring and metrics collection

**Key sections:**
1. System Architecture Overview
2. Key Files & Responsibilities
3. Main Request Flow: `/chat/search`
4. 12 Pipeline Nodes (detailed)
5. Follow-Up Handling Flows
6. Session Persistence
7. Search Scoring (RRF algorithm)
8. Ingestion Pipeline
9. Configuration Reference
10. Extension Points
11. Performance Considerations
12. Debugging Guide
13. Architectural Rules
14. Monitoring & Metrics Integration

---

### [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Quick Reference & How-To
**35 sections | 409 lines | 9.5 KB**

Practical, task-focused guide for daily development work.

**Use for:**
- Adding a new filter type (complete step-by-step example)
- Adding a new graph query type
- Debugging a specific request
- Running tests (commands provided)
- Modifying node logic
- Session management operations
- Search integration
- LLM interactions
- Configuration changes
- Performance optimization checklist
- Troubleshooting problems

**Quick lookup:**
- File Map by Task (what to modify for each feature)
- State Shape Reference (GraphState TypedDict)
- Critical Code Paths (4 main flows)
- Troubleshooting Matrix (symptom → cause → fix)

---

### [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) - Complete Dependency & Function Map
**45 sections | 601 lines | 18 KB**

Function-level navigation and dependency tracking.

**Navigate to:**
- Find where functions are defined
- Understand function call chains
- Discover function signatures and inputs/outputs
- See all dependencies for each node
- Understand data model structures
- Review import structure and avoid circular dependencies
- Explore monitoring and metrics functions

**Structure:**
1. Entry Points (HTTP API)
2. Core Pipeline Nodes (12 nodes with full dependency trees)
3. Search Implementations (BM25, Vector, Graph, Hybrid)
4. Session Management
5. LLM Integration
6. Utility Functions
7. Monitoring & Metrics
8. Middleware
9. Configuration
10. Data Models
11. Import Structure
12. Testing Dependencies

---

## 🎯 Quick Navigation by Task

### "I'm new to the codebase"
1. Read: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - System Architecture Overview
2. Read: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Main Request Flow
3. Skim: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) - Entry Points
4. Bookmark: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - File Map by Task

### "I need to add a new filter"
1. Check: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - "Adding Support for New Filter Type"
2. Reference: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) - Core Pipeline Nodes
3. Verify: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Query Rewriter Node

### "I'm debugging a problem"
1. Use: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - "Debugging a Specific Request"
2. Check: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - "Troubleshooting Matrix"
3. Trace: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) - dependency trees
4. Review: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Error Handling Patterns

### "I need to optimize performance"
1. Review: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - "Performance Considerations"
2. Check: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - "Performance Checklist"
3. Measure: latency breakdown against actual system
4. Reference: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) - call chains

### "I'm implementing a new feature"
1. Check: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - "Extension Points for Developers"
2. Find files: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - "File Map by Task"
3. Copy template: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - "Modifying Node Logic"
4. Understand flow: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) - dependency trees

### "I need to understand session handling"
1. Read: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - "Session Persistence"
2. Review: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - "Session Management"
3. Check: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) - Session Management section

### "I'm adding tests"
1. Check: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - "Testing Strategy"
2. Review: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - "Running Tests"
3. Reference: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) - "Testing Dependencies"

---

## 📊 Documentation Coverage

| Component | SYSTEM_ARCHITECTURE | DEVELOPMENT_GUIDE | CODEBASE_REFERENCE |
|-----------|-----------|-----------------|------------|
| Architecture | ✅ (detailed) | ✅ (overview) | ✅ (dependencies) |
| All 10 Nodes | ✅ (all) | ✅ (summary) | ✅ (dependencies) |
| Session Mgmt | ✅ (design) | ✅ (usage) | ✅ (signatures) |
| Search | ✅ (algorithm) | ✅ (integration) | ✅ (classes) |
| LLM | ✅ (prompts) | ✅ (examples) | ✅ (functions) |
| Configuration | ✅ (all) | ✅ (changes) | ✅ (structure) |
| Follow-ups | ✅ (flows) | ✅ (overview) | ⚠️ (implicit) |
| Testing | ✅ (strategy) | ✅ (commands) | ✅ (mocks) |
| Debugging | ✅ (detailed) | ✅ (procedures) | ✅ (traces) |
| Performance | ✅ (analysis) | ✅ (checklist) | ✅ (calls) |

---

## 🔍 Finding Information Quickly

### By Topic

**Request Handling**
- Start: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Main Request Flow section
- Code: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) → Entry Points section

**Pipeline Nodes**
- Overview: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Pipeline Nodes section
- Detailed: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) → Core Pipeline Nodes section

**Search & Ranking**
- Algorithm: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Search Scoring & Ranking
- Code: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) → Search Implementations

**Conversation/Sessions**
- Design: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Session Persistence
- Usage: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) → Session Management
- Code: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) → Session Management

**Configuration**
- All settings: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Configuration & Settings
- How to change: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) → Configuration Changes

**Features & Extensions**
- How to add: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Extension Points
- File map: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) → File Map by Task
- Example: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) → Adding New Filter Type

**Testing**
- Strategy: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Testing Strategy
- Commands: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) → Running Tests
- Setup: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) → Testing Dependencies

**Debugging**
- Guide: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Debugging Guide
- Procedures: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) → Debugging a Specific Request
- Traces: [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) → all dependency trees

**Performance**
- Analysis: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Performance Considerations
- Checklist: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) → Performance Checklist
- Optimization: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → Optimization Opportunities

---

## 📋 Document Features

### SYSTEM_ARCHITECTURE.md
- System diagrams
- Step-by-step flows
- Complete node explanations
- Algorithm details
- Configuration reference
- Best for: Understanding the big picture

### DEVELOPMENT_GUIDE.md
- Task-focused sections
- Copy-paste code examples
- Bash commands
- Checklists and matrices
- Best for: Day-to-day development

### CODEBASE_REFERENCE.md
- Dependency trees
- Function signatures
- File locations
- Class structures
- Best for: Code navigation and refactoring

---

## 🚀 Key Information Summary

### System Architecture
- **Entry:** FastAPI `/chat/search` endpoint
- **Session:** Redis (24h TTL, conversation history)
- **Pipeline:** 10-node LangGraph with conditional routing
- **Search:** 3 backends (BM25, Vector, Graph)
- **Ranking:** RRF (Reciprocal Rank Fusion)
- **Generation:** GPT-4 Turbo with grounded context
- **Persistence:** Session updates after each request

### Main Flows
1. **New Search** → Query Rewriter → BM25 + Vector → RRF → Context Selector → RAG
2. **Follow-up Filter** → Filter Previous → Context Selector → RAG
3. **Clarification** → Ask User → END
4. **Graph Search** → Graph Query → Context Selector → RAG

### 10 Core Nodes
1. **context_resolver_node** - Load session + previous results
2. **intent_detector_node** - Classify intent (search/filter/clarify)
3. *(Router)* - Route based on intent
4. **query_rewriter_node** - Extract entities, expand query
5. **bm25_search_node** - OpenSearch lexical search
6. **vector_search_node** - pgvector semantic search
7. **rrf_merge_node** - Combine results with RRF
8. **context_selector_node** - Select diverse context
9. **rag_generator_node** - Generate LLM response
10. **clarification_node** - Ask for clarification

### 2 Optional Nodes (Phase 5)
- **graph_search_node** - Neo4j relationship queries
- **rrf_merge_3way_node** - Include graph in ranking

### Key Concepts
- **Entity Extraction:** Cuisine, price, serving, dietary, location
- **Follow-ups:** Price, serving, dietary, scope (same/other restaurant), location
- **Scope Detection:** "same restaurant" / "other restaurants"
- **RRF Scoring:** Docs in multiple result sets rank higher
- **Token Budget:** Configuration in settings.py (not fully enforced)
- **Graceful Degradation:** Missing dependencies don't crash pipeline

---

## ✅ What's Documented

- ✅ All 10 pipeline nodes with full explanations
- ✅ 3 search backends (BM25, Vector, Graph)
- ✅ Session persistence and conversation handling
- ✅ Follow-up conversation flows with examples
- ✅ RRF algorithm with weighted scoring
- ✅ Entity extraction patterns
- ✅ Filter types and logic
- ✅ LLM integration and prompting
- ✅ Configuration and settings
- ✅ Error handling patterns
- ✅ Testing strategy and commands
- ✅ Debugging procedures
- ✅ Performance analysis
- ✅ Extension points for new features
- ✅ Data model structures
- ✅ Function dependencies

---

## 📞 Document Maintenance

When code changes:
1. **New Node Added?** → Update [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md) Core Pipeline Nodes
2. **Changed Flow?** → Update [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) Pipeline Nodes section
3. **New Feature?** → Add to [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) File Map by Task
4. **Changed Config?** → Update [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) Configuration section

---

## 🎓 Learning Path

**Day 1: Fundamentals**
- Read: SYSTEM_ARCHITECTURE.md - System Architecture Overview
- Read: SYSTEM_ARCHITECTURE.md - Main Request Flow
- Read: SYSTEM_ARCHITECTURE.md - Pipeline Nodes (skim)

**Day 2: Deep Dive**
- Read: SYSTEM_ARCHITECTURE.md - All remaining sections
- Reference: CODEBASE_REFERENCE.md as needed

**Day 3: Practical**
- Bookmark: DEVELOPMENT_GUIDE.md
- Reference: CODEBASE_REFERENCE.md for code navigation
- Run: Tests to see system in action

**Ongoing**
- Use DEVELOPMENT_GUIDE.md for daily tasks
- Reference CODEBASE_REFERENCE.md when navigating code
- Return to SYSTEM_ARCHITECTURE.md for architecture questions

---

**Created:** January 29, 2026
**Total Documentation:** 1,425 lines across 3 files
**Format:** Markdown (GitHub-compatible)
**Scope:** Complete system documentation for developers

