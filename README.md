# Bedtime Story Generator (Safety + Planning + Judge/Rewrite Loops)

A CLI tool that generates **safe, cozy bedtime stories for kids (ages 5–10)** using an LLM, then iteratively improves the story using a **judge → rewrite loop**. After the story is printed, the user can provide feedback (e.g., “shorter”, “funnier”, “sad ending”), and the tool revises the *same* story while preserving characters/setting/plot unless asked otherwise.

This project demonstrates an “agentic” workflow:
- **Planning** (structured plan JSON)
- **Generation** (story from plan)
- **Evaluation** (judge JSON + heuristics)
- **Revision** (editor rewrite)
- **Interactive iteration** (feedback parsing + feedback judge)

---

## Requirements

- Python 3.9+
- `openai` Python SDK
- An OpenAI API key set as an environment variable

> Model is intentionally pinned:  
> `MODEL_NAME = "gpt-3.5-turbo"  # DO NOT CHANGE`

---

## Run

Run the program and follow the prompts:
- enter an initial story request
- optionally enter feedback prompts until you press Enter on an empty line

Example prompts:
- `funny and long`
- `a gently spooky bedtime story about a kitten hearing sounds at night`
- `much shorter, quick bedtime version`
- `longer and funnier and cuter`
- `change the ending to be sad`

---

## High-Level Architecture (Block Diagram)

The system has two main loops.

### A) Initial Story Generation Loop

1. **User request**
2. **Safety check**
   - If unsafe → rewrite request into a kid-safe version + show note
   - If safe → proceed
3. **Build story spec** (length/tone/humor inferred from request)
4. **LLM Plan step** → outputs structured **plan JSON**
5. **LLM Story step** → story written following the plan
6. **Judge step** (JSON scores + issues) + **length check**
7. If not passing → **Rewrite step** and repeat judge (up to max rounds)
8. **Clean discouraged words** + print story
9. Ask for feedback (optional)

### B) Feedback Revision Loop

1. **User feedback**
2. **Safety check** on feedback
   - If unsafe → replace with safe alternative + show note
3. **Parse feedback constraints** into JSON via LLM
4. **Deterministic overrides** (keyword-based) to prevent misclassification
5. **Update spec** + compute target word count bounds
6. **LLM revise** story using explicit constraints
7. **Feedback judge** checks:
   - follows feedback
   - preserves core story
   - safety
   - simplicity + bedtime vibe
   - length bounds
8. If not passing → revise again using must-fix list (up to max rounds)
9. **Clean discouraged words** + print revised story

---

## Key Components

### 1) LLM Client + Disk Cache

- `get_client()` lazily initializes `OpenAI(api_key=...)`
- `call_model()` caches responses to `.llm_cache.json`
- Cache key = SHA256 of `{model, messages, max_tokens, temperature}`

Cache behavior notes:
- If `ENABLE_CACHE = True`, repeated identical prompts reuse cached responses.
- Deleting `.llm_cache.json` effectively “resets” cached outputs.

### 2) Safety Filtering

- `check_text_safety(text)` detects unsafe keywords/patterns
- `sanitize_user_text_for_kids(text, kind="request"|"feedback")`
  - request: rewrite into a safe bedtime prompt + display note
  - feedback: replace with a safe alternative + display note

Goal: keep stories **kid-safe** (no violence, cruelty, explicit sexual content, self-harm, intense horror).

### 3) Planning + Generate + Judge + Rewrite (Initial Story)

- `build_story_spec(user_request)`  
  Extracts settings like `length`, `tone`, `humor` from the user request.
- `create_story_plan(spec)`  
  Produces structured plan JSON (title, setting, characters, attempts, resolution).
- `generate_story(spec, plan)`  
  Writes a story following the plan closely.
- `judge_story(spec, story)`  
  Produces judge JSON with scores, `must_fix`, and word/sentence simplicity indicators.
- `revise_story(spec, story, judge_json)`  
  Edits the story to fix judge issues (simpler words, better arc, better bedtime ending).

### 4) Feedback Revision Loop (Preserve Story)

- `parse_user_feedback_constraints(feedback)`  
  Converts messy feedback into a structured constraint JSON.
- Keyword override sets (`SPOOKY_WORDS`, `LONGER_WORDS`, etc.)  
  Deterministically corrects cases where the LLM might misread intent.
- `update_spec_with_feedback(base_spec, feedback)`  
  Adjusts tone/humor/length targets.
- `feedback_rewriter_messages(...)`  
  Revises the story while preserving characters/setting/plot unless requested otherwise.
- `feedback_judge_messages(...)`  
  Verifies:
  - feedback is followed
  - core story is preserved
  - story is safe and bedtime-appropriate
  - vocabulary/sentences remain simple
  - length matches target bounds

---

## Files

- `main.py` — all logic (LLM calls, caching, safety, planning, judge/rewrite loops, CLI)

---

## Example Run (What You’ll See)

- Prompt for an initial request
- A story printed
- Repeated prompts for feedback until you press Enter on an empty line

```text
What kind of story do you want to hear? funny and long
... (story prints) ...
Want any changes? longer and funnier and cuter
... (revised story prints) ...
Want any changes?
... (and so on) ...
