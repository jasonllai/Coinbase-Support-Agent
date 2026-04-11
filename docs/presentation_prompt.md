# Presentation Deck Prompt — Coinbase Support Agent (Balanced)

> **For use with Gamma, Manus, or any AI presentation generator.**
> Paste everything under "## PROMPT" into the generator. This prompt is self-contained.

---

## PROMPT (copy from here)

---

Create a **6-slide professional presentation deck** for a project called **"Coinbase Support Agent"**.

**Balance rule — strictly enforced for every slide:**
- Each slide must have **one dominant visual** (diagram, flowchart, chart, or icon grid) that occupies at least 50% of the content area and carries the main message.
- Text is **supporting only**: short labels, 3–5 word bullets, or single-sentence callouts. No paragraphs anywhere.
- Think editorial magazine layout, not a lecture slide.

---

### DESIGN SYSTEM

**Colors:**
- Primary accent: Coinbase Blue `#0052FF`
- Background: White `#FFFFFF`
- Card / panel fill: Light grey `#F5F7FA`
- Body text: Charcoal `#1A1A2E`
- Warning / safety accent: Amber `#F5A623`
- Success / positive accent: Teal `#00C2A8`

**Typography:** Clean sans-serif. Titles bold at ~36pt. Labels bold at ~14pt. Callout numbers at ~52pt. No paragraph prose on any slide.

**Style:** Modern fintech product deck — think Stripe, Linear, or Vercel's visual language. Generous whitespace. Minimal borders. No clip art. Icons should be simple, outlined, and monochrome unless colour carries meaning.

**Every slide:** Slide number `01/06` – `06/06` bottom-right corner. Small label "Coinbase Support Agent" bottom-left in `#0052FF`.

---

### SLIDE 1 — TITLE

**Layout:** Full-bleed dark background. Deep navy `#0A1628` to Coinbase blue `#0052FF` vertical gradient. All text white. Centred composition.

**Visual element:** Subtle abstract background pattern — faint connected-node network or dot grid with soft glow in `#0052FF`, low opacity, behind the text.

**Text (centred, vertical stack):**

1. **Large bold title:**
   `Coinbase Support Agent`

2. **Subtitle, lighter weight:**
   `An AI-Powered Help Center Assistant — Grounded, Scoped, and Safe`

3. **Four pill badges in a horizontal row** (rounded rectangle, semi-transparent white border, white text, small font):
   `RAG · FAISS + BM25` &nbsp;&nbsp; `LangGraph` &nbsp;&nbsp; `Guardrails` &nbsp;&nbsp; `Multi-Turn Actions`

4. **Bottom label, small muted text:**
   `LLM Applications — Course Final Project`

---

### SLIDE 2 — PROBLEM, DOMAIN & DATASET

**Layout:** Three horizontal sections stacked vertically. Top third: problem contrast diagram (primary visual). Middle third: domain & dataset rationale (2-column). Bottom strip: corpus breakdown bar.

**Slide title:** Why Coinbase — and How We Built the Knowledge Base

**Top section — Visual contrast diagram (~35% of slide, primary visual):**

Draw two side-by-side mini-flow diagrams separated by a vertical divider:

*Left side — labelled "Generic Chatbot" (amber `#F5A623` accents, ❌):*
```
[User] → [LLM] → ❌ Hallucinated policy
                → ❌ No citation
                → ❌ No action
```

*Right side — labelled "Coinbase Support Agent" (teal `#00C2A8` accents, ✓):*
```
[User] → [RAG + LangGraph] → ✓ Evidence-grounded answer
                           → ✓ Cited Help Center source
                           → ✓ Action executed
```

**Middle section — 2-column domain & dataset rationale (~35% of slide):**

*Left column — "Why Coinbase?" (icon + 2-line label each, 3 rows):*

| Icon | Label |
|---|---|
| 🎯 | **High-volume support domain** — account access, transactions, security, fees |
| 📖 | **Rich public knowledge base** — help.coinbase.com with structured articles |
| 🔒 | **Safety stakes are real** — 2FA bypass, KYC fraud, account compromise are genuine threats |

*Right column — "How We Built the KB" (icon + 2-line label each, 3 rows):*

| Icon | Label |
|---|---|
| 🏛️ | **Internet Archive (Wayback Machine)** — live site blocks scrapers with Cloudflare; archived `id_` captures are stable and reproducible |
| ✂️ | **Semantic chunking** — heading/paragraph-aware splits preserve article structure |
| 🔗 | **Canonical URLs preserved** — citations always point to the live help.coinbase.com page |

**Bottom strip — Corpus breakdown horizontal bar (full width, ~30% of slide):**

Label above: `Knowledge Base — 118 Articles · 200 Indexed Chunks`

Segmented horizontal bar (proportional, colour-coded):
- Getting Started: 73 articles (62%) — Coinbase blue `#0052FF`
- Managing My Account: 25 (21%) — medium blue `#4D8EFF`
- Privacy & Security: 11 (9%) — teal `#00C2A8`
- Trading & Funding: 8 (7%) — indigo `#7B5EA7`
- Other: 1 (1%) — grey

Small legend labels beneath each segment.

---

### SLIDE 3 — SYSTEM ARCHITECTURE

**Layout:** The pipeline diagram is the hero and takes up ~65% of the slide. Tech-stack chips sit below it. A compact node legend sits to the right.

**Slide title:** How It Works

**Primary visual — LangGraph pipeline (centre, large):**

Draw a horizontal left-to-right flow with styled rounded boxes and labelled arrows:

```
[Streamlit UI]
      |
      ▼
 ┌─────────────────────────────────────────────────┐
 │           LangGraph Agent Pipeline              │
 │                                                 │
 │  [LOAD] ──► [GUARD] ──► [INTENT] ──► [DISPATCH] ──► [PERSIST] │
 └─────────────────────────────────────────────────┘
      |              |            |           |
      ▼              ▼            ▼           ▼
 [SQLite DB]  [Qwen3-30B]  [9 Intent     [FAISS +
              [LLM]         Labels]       BM25]
```

Node colour coding:
- **LOAD** — light grey fill
- **GUARD** — amber `#F5A623` fill (safety stage)
- **INTENT** — Coinbase blue `#0052FF` fill, white text
- **DISPATCH** — darker blue `#003CC5` fill, white text
- **PERSIST** — light grey fill

**Right sidebar — 5-row node legend (compact, 10pt text):**

| Node | One-line job |
|---|---|
| LOAD | Restore session from SQLite |
| GUARD | Regex screen + LLM safety check |
| INTENT | Route to 1 of 9 intent labels |
| DISPATCH | Run KB search, action, or recall |
| PERSIST | Save messages + trace |

**Bottom row — 3 tech-stack chips (pill style, grey background, `#0052FF` text):**

`Qwen3-30B-A3B-FP8` &nbsp;·&nbsp; `FAISS IndexFlatIP (768-dim) + BM25` &nbsp;·&nbsp; `SQLite — 4 tables`

---

### SLIDE 4 — CORE CAPABILITIES

**Layout:** Top half: a user-journey swimlane diagram showing a question flowing through the system to different outputs. Bottom half: 3-column icon grid (one column per action type).

**Slide title:** What the Agent Can Do

**Primary visual — Intent routing swimlane (top half, ~50% of slide):**

Draw a horizontal swimlane with 3 lanes:

```
User Input ──────────────────────────────────────────────────────►
                    │
                [Intent Router]
               /       |       \         \
              ▼        ▼        ▼          ▼
          [KB Q&A] [Action] [Guardrail] [Memory
                    Flow]   → Refusal    Shortcut]
              │        │
              ▼        ▼
         [Answer +  [Ticket /
          Citations] Recovery /
                    Transaction]
```

Use colour-coded output boxes:
- KB Q&A box: blue `#0052FF`
- Action box: teal `#00C2A8`
- Guardrail/Refusal box: amber `#F5A623`
- Memory shortcut box: purple `#7B5EA7`

**Bottom half — 3-column icon grid (one card per column):**

**Column 1 — Knowledge Q&A** (blue accent)
- Icon: book/document
- Title: Help Center Q&A
- 2 lines: `118 articles · 200 chunks` / `Cites source URL per answer`

**Column 2 — Actions** (teal accent)
- Icon: lightning bolt / gears
- Title: 4 Operational Actions
- 2 lines: `Transaction · Ticket · Onboarding` / `Account Recovery (multi-turn)`

**Column 3 — Safety** (amber accent)
- Icon: shield
- Title: Guardrails + Memory
- 2 lines: `Blocks injection, bypass, off-scope` / `Recalls prior questions & case IDs`

---

### SLIDE 5 — EVALUATION RESULTS

**Layout:** Top row: 5 KPI stat callouts. Middle: horizontal bar chart (test coverage). Bottom-right: donut chart (corpus category split). Bottom-left: 2-line caption.

**Slide title:** Evaluation — 50 Scenarios, 0 Failures

**Top row — 5 KPI cards (equal width, horizontal):**

Each card: white background, thin `#0052FF` top border, large metric number in `#0052FF`, small label below in charcoal.

| Metric | Number | Sub-label |
|---|---|---|
| Overall Pass Rate | **100%** | 50 / 50 scenarios |
| Intent Accuracy | **100%** | 24 / 24 routed correctly |
| Guardrail Rate | **100%** | 9 / 9 blocked correctly |
| Citation Rate | **92.3%** | 12 / 13 KB answers cited |
| Action Success | **100%** | 26 / 26 actions completed |

**Middle — horizontal bar chart:**

Title (small, above chart): `Test Coverage by Category`

Bars (horizontal, `#0052FF` fill, count label on right end of each bar):

```
KB Q&A           ████████████  12
Actions          ████████████████████████  24
Guardrails       █████████  9
Memory & Recall  █████  5
Edge Cases       ██  2
Infra Smoke      █  1
```

**Bottom-right — donut chart:**

Title: `Knowledge Base — 118 Articles`
Segments (use shades of blue and teal):
- Getting Started: 73 (62%)
- Managing My Account: 25 (21%)
- Privacy & Security: 11 (9%)
- Trading & Funding: 8 (7%)
- Other: 1 (1%)

**Bottom-left — 2-line caption (small, muted charcoal):**
`End-to-end scenarios via live agent pipeline`
`118 articles · 200 indexed chunks · 5 topic categories`

---

### SLIDE 6 — LEARNINGS & LIMITATIONS

**Layout:** 2×2 grid of quadrants. Top two quadrants = "What Worked Well" (teal accent). Bottom two quadrants = "Challenges & Limitations" (amber accent). Each quadrant has a small icon, a bold 3–5 word title, and one mini-diagram or icon-supported visual — no prose paragraphs.

**Slide title:** What We Learned

**Top-left quadrant — "Hybrid Retrieval Works" (teal `#00C2A8` accent border)**

Mini-diagram:
```
Dense FAISS  ──60%──┐
                    ├──► Fused Score ──► Top-4 results ✓
BM25 Lexical ──40%──┘
```

Two short bullets:
- Catches semantic paraphrases *and* exact product strings
- Removing the cross-encoder reranker had negligible quality impact

**Top-right quadrant — "Explicit Routing Is Explainable" (teal accent border)**

Mini-diagram (LangGraph node strip, simplified):
```
[GUARD] → [INTENT] → [DISPATCH]
  amber      blue      dark blue
```

Two short bullets:
- Each node maps to one demo talking point
- Router few-shot examples reduced misclassification significantly

**Bottom-left quadrant — "Dependency Conflicts Are Costly" (amber `#F5A623` accent border)**

Mini-diagram (problem → fix):
```
sentence-transformers + FAISS
          │
    ❌ OMP crash on macOS
          │
    ✓ Switch to remote embed API
    ✓ Remove cross-encoder reranker
```

Two short bullets:
- `libomp.dylib` conflict between PyTorch and FAISS: hours of debugging
- Solution: remote embedding API eliminates shared-library conflicts

**Bottom-right quadrant — "Known Limitations" (amber accent border)**

Visual: 3-row icon + label list (no diagram needed — use clear warning icons)

| Icon | Limitation |
|---|---|
| 🕐 | **Historical corpus** — Wayback snapshots, not live; policies may have changed |
| 🎲 | **LLM variance** — router and safety classifier outputs can shift between runs |
| 🧪 | **Mock data only** — transactions, tickets, and recovery cases are simulated |
| 📏 | **Scope is narrow** — agent cannot help with trading, portfolio, or price queries |

---

### FINAL DESIGN NOTES

1. **No prose paragraphs anywhere** — if a sentence is longer than 15 words, split or cut it.
2. **Diagrams:** Use clean box-and-arrow style. Rounded rectangles. 1–3 word labels inside boxes.
3. **Charts:** Flat, 2D only. No 3D. No pie charts (use segmented bar or donut). `#0052FF` as primary colour.
4. **Whitespace:** Generous padding inside every card and quadrant. Nothing feels crammed.
5. **Slide numbers:** `01/06` through `06/06` in small grey text, bottom-right, every slide.
6. **Persistent footer label:** `Coinbase Support Agent` in small `#0052FF`, bottom-left, every slide.
7. **Export target:** Static PDF and live screen. No animation needed.
