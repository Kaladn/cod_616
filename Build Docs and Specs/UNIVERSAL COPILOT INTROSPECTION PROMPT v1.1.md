You are operating inside ONE workspace.

Workspace Root:
- Identify the workspace root directory explicitly.
- Analyze ONLY files inside this workspace root.
- Do NOT infer behavior from similarly named folders or external deployments.
- If a referenced file is outside this workspace, mark it as NOT PRESENT.

Your task is NOT to build, refactor, optimize, critique, or invent.
Your task is to EXTRACT FACTS about THIS workspace only.

STRICT RULES:
- Do NOT create or modify files
- Do NOT suggest improvements or fixes
- Do NOT evaluate quality, performance, or design tradeoffs
- Do NOT assume future architecture
- Do NOT reference other modules unless directly imported here
- If behavior is not explicitly implemented, say NOT IMPLEMENTED
- If behavior is implied but not enforced, say IMPLIED ONLY

Produce a structured report with the following sections ONLY:

1. MODULE IDENTITY
   - Module name
   - Intended role (as implemented or explicitly documented)
   - What this module DOES
   - What this module explicitly DOES NOT do (based on code or comments only)

2. DATA OWNERSHIP
   - Data types this module creates
   - Data types this module consumes
   - Data types this module persists (if any)
   - Data types this module does not persist

3. PUBLIC INTERFACES
   - Public classes
   - Public functions / methods
   - Expected inputs and outputs (types + meaning)
   - Side effects (disk, memory, stdout, network, none)

4. DEPENDENCIES
   - Internal imports (within this workspace)
   - External imports (stdlib / third-party)
   - Runtime import behavior (if any)

5. STATE & MEMORY
   - State held in memory
   - Whether state is bounded or unbounded
   - How state is reset, evicted, or never cleared

6. INTEGRATION POINTS
   - Explicit upstream inputs expected
   - Explicit downstream outputs produced
   - Any required external conditions (schemas, files, environment)

7. OPEN FACTUAL QUESTIONS
   - Declared but unused code paths
   - Defined but unraised exceptions
   - Referenced but missing components
   - Implicit assumptions required for correct operation

Output MUST be factual and grounded in code.
Do NOT include opinions, risks, or recommendations.
End the report. Do not propose changes.
