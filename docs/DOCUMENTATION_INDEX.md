# Documentation Index

**Complete codebase documentation for the Hybrid Search & RAG system**

---

## 📚 Main Documentation Files

### [CODE_FLOW.md](CODE_FLOW.md) - System Architecture & Execution Flows
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

### [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Quick Reference & How-To
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

### [FUNCTION_MAP.md](FUNCTION_MAP.md) - Complete Dependency & Function Map
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
1. Read: [CODE_FLOW.md](CODE_FLOW.md) - System Architecture Overview
2. Read: [CODE_FLOW.md](CODE_FLOW.md) - Main Request Flow
3. Skim: [FUNCTION_MAP.md](FUNCTION_MAP.md) - Entry Points
4. Bookmark: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - File Map by Task

### "I need to add a new filter"
1. Check: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "Adding Support for New Filter Type"
2. Reference: [FUNCTION_MAP.md](FUNCTION_MAP.md) - Core Pipeline Nodes
3. Verify: [CODE_FLOW.md](CODE_FLOW.md) - Query Rewriter Node

### "I'm debugging a problem"
1. Use: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "Debugging a Specific Request"
2. Check: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "Troubleshooting Matrix"
3. Trace: [FUNCTION_MAP.md](FUNCTION_MAP.md) - dependency trees
4. Review: [CODE_FLOW.md](CODE_FLOW.md) - Error Handling Patterns

### "I need to optimize performance"
1. Review: [CODE_FLOW.md](CODE_FLOW.md) - "Performance Considerations"
2. Check: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "Performance Checklist"
3. Measure: latency breakdown against actual system
4. Reference: [FUNCTION_MAP.md](FUNCTION_MAP.md) - call chains

### "I'm implementing a new feature"
1. Check: [CODE_FLOW.md](CODE_FLOW.md) - "Extension Points for Developers"
2. Find files: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "File Map by Task"
3. Copy template: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "Modifying Node Logic"
4. Understand flow: [FUNCTION_MAP.md](FUNCTION_MAP.md) - dependency trees

### "I need to understand session handling"
1. Read: [CODE_FLOW.md](CODE_FLOW.md) - "Session Persistence"
2. Review: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "Session Management"
3. Check: [FUNCTION_MAP.md](FUNCTION_MAP.md) - Session Management section

### "I'm adding tests"
1. Check: [CODE_FLOW.md](CODE_FLOW.md) - "Testing Strategy"
2. Review: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "Running Tests"
3. Reference: [FUNCTION_MAP.md](FUNCTION_MAP.md) - "Testing Dependencies"

---

## 📊 Documentation Coverage

| Component | CODE_FLOW | DEVELOPER_GUIDE | FUNCTION_MAP |
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
- Start: [CODE_FLOW.md](CODE_FLOW.md) → Main Request Flow section
- Code: [FUNCTION_MAP.md](FUNCTION_MAP.md) → Entry Points section

**Pipeline Nodes**
- Overview: [CODE_FLOW.md](CODE_FLOW.md) → Pipeline Nodes section
- Detailed: [FUNCTION_MAP.md](FUNCTION_MAP.md) → Core Pipeline Nodes section

**Search & Ranking**
- Algorithm: [CODE_FLOW.md](CODE_FLOW.md) → Search Scoring & Ranking
- Code: [FUNCTION_MAP.md](FUNCTION_MAP.md) → Search Implementations

**Conversation/Sessions**
- Design: [CODE_FLOW.md](CODE_FLOW.md) → Session Persistence
- Usage: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) → Session Management
- Code: [FUNCTION_MAP.md](FUNCTION_MAP.md) → Session Management

**Configuration**
- All settings: [CODE_FLOW.md](CODE_FLOW.md) → Configuration & Settings
- How to change: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) → Configuration Changes

**Features & Extensions**
- How to add: [CODE_FLOW.md](CODE_FLOW.md) → Extension Points
- File map: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) → File Map by Task
- Example: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) → Adding New Filter Type

**Testing**
- Strategy: [CODE_FLOW.md](CODE_FLOW.md) → Testing Strategy
- Commands: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) → Running Tests
- Setup: [FUNCTION_MAP.md](FUNCTION_MAP.md) → Testing Dependencies

**Debugging**
- Guide: [CODE_FLOW.md](CODE_FLOW.md) → Debugging Guide
- Procedures: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) → Debugging a Specific Request
- Traces: [FUNCTION_MAP.md](FUNCTION_MAP.md) → all dependency trees

**Performance**
- Analysis: [CODE_FLOW.md](CODE_FLOW.md) → Performance Considerations
- Checklist: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) → Performance Checklist
- Optimization: [CODE_FLOW.md](CODE_FLOW.md) → Optimization Opportunities

---

## 📋 Document Features

### CODE_FLOW.md
- System diagrams
- Step-by-step flows
- Complete node explanations
- Algorithm details
- Configuration reference
- Best for: Understanding the big picture

### DEVELOPER_GUIDE.md
- Task-focused sections
- Copy-paste code examples
- Bash commands
- Checklists and matrices
- Best for: Day-to-day development

### FUNCTION_MAP.md
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
1. **New Node Added?** → Update [FUNCTION_MAP.md](FUNCTION_MAP.md) Core Pipeline Nodes
2. **Changed Flow?** → Update [CODE_FLOW.md](CODE_FLOW.md) Pipeline Nodes section
3. **New Feature?** → Add to [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) File Map by Task
4. **Changed Config?** → Update [CODE_FLOW.md](CODE_FLOW.md) Configuration section

---

## 🎓 Learning Path

**Day 1: Fundamentals**
- Read: CODE_FLOW.md - System Architecture Overview
- Read: CODE_FLOW.md - Main Request Flow
- Read: CODE_FLOW.md - Pipeline Nodes (skim)

**Day 2: Deep Dive**
- Read: CODE_FLOW.md - All remaining sections
- Reference: FUNCTION_MAP.md as needed

**Day 3: Practical**
- Bookmark: DEVELOPER_GUIDE.md
- Reference: FUNCTION_MAP.md for code navigation
- Run: Tests to see system in action

**Ongoing**
- Use DEVELOPER_GUIDE.md for daily tasks
- Reference FUNCTION_MAP.md when navigating code
- Return to CODE_FLOW.md for architecture questions

---

**Created:** January 29, 2026  
**Total Documentation:** 1,425 lines across 3 files  
**Format:** Markdown (GitHub-compatible)  
**Scope:** Complete system documentation for developers

