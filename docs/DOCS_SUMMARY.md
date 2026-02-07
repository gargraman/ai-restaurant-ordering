# Documentation Summary

**Updated:** 2026-02-01
**Goal:** Reduce duplication and keep a small, authoritative set of docs.

---

## Canonical Documents (use these)

### Search Platform (3)
1. SYSTEM_ARCHITECTURE.md — Architecture and request flow overview.
2. DEVELOPMENT_GUIDE.md — Task‑oriented developer workflows.
3. CODEBASE_REFERENCE.md — Function signatures and dependency map.

### Order Management (3)
1. ORDER_MANAGEMENT_PLAN.md — Single source of truth for requirements and phases.
2. IMPLEMENTATION_STATUS.md — Canonical gap tracker mapped to files/endpoints.
3. QUICK_START_GUIDE.md — Phase 1 execution guide (after Phase 0 blockers are closed).

---

## Supporting Summary (1)

PROJECT_STATUS_SUMMARY.md — Status snapshot derived from the checklist.

---

## De‑duplication Rules

1. If a fact appears in multiple docs, keep it only in:
	- ORDER_MANAGEMENT_PLAN.md (requirements)
	- IMPLEMENTATION_STATUS.md (status/gaps)

2. PROJECT_STATUS_SUMMARY.md must mirror the checklist and contain no new requirements.

3. QUICK_START_GUIDE.md must only contain execution steps, not requirements or status.

4. If a change is made, update in this order:
	status → completion summary → quick start (if needed).

