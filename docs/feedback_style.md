---
name: Collaboration Style Feedback
description: User prefers to write crypto code themselves, wants AI for architecture/planning/verification not code generation.
type: feedback
---

- User implements low-level crypto primitives themselves (NTT, Merkle, etc.) -- don't generate full implementations, just provide guidance, imports, and TODOs
- When user asks "should X be Y?" they want architectural reasoning, not just "yes"
- User corrects placement decisions actively ("shouldn't these be in provider/pcs instead?") -- think about module organization from a reusability perspective
- Verify claims about trait equivalence with actual source code, not assumptions
- Keep plan files updated as decisions evolve during conversation

**Why:** User is building deep understanding of the codebase while implementing. Generating code for them defeats the purpose.

**How to apply:** For crypto primitives, provide the mapping/guidance and let user port. For glue code (mod.rs, Engine registration), can be more hands-on.
