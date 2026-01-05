import os
import json
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI

"""
Next 2 hours ideas (already partially implemented here):
- Make “short vs medium vs long” match outputs more reliably by adding a strict word-count gate.
- Add a stronger reading-level estimator (Flesch-Kincaid) and enforce range with judge.
- Add themed recipe templates chosen by classifier (space, detective, ocean, etc.).
- Add regression tests for safety + bedtime tone + vocabulary simplicity.
- Session memory for preferences (favorite character, length, humor style).
"""

MODEL_NAME = "gpt-3.5-turbo"  # DO NOT CHANGE

# ----------------------------
# Client (lazy init)
# ----------------------------
_CLIENT: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _CLIENT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


# ----------------------------
# Simple disk cache (optional but recommended)
# ----------------------------
ENABLE_CACHE = True
CACHE_PATH = ".llm_cache.json"
_cache_mem: Dict[str, str] = {}


def _load_cache() -> None:
    global _cache_mem
    if not ENABLE_CACHE:
        return
    if _cache_mem:
        return
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            _cache_mem = json.load(f)
    except Exception:
        _cache_mem = {}


def _save_cache() -> None:
    if not ENABLE_CACHE:
        return
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache_mem, f)
    except Exception:
        pass


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def call_model(messages: List[Dict[str, str]], max_tokens=1200, temperature=0.7) -> str:
    """
    Cached chat completion call.
    """
    client = get_client()
    _load_cache()

    cache_key = _stable_hash(
        {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    )

    if ENABLE_CACHE and cache_key in _cache_mem:
        return _cache_mem[cache_key]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""

    if ENABLE_CACHE:
        _cache_mem[cache_key] = text
        _save_cache()

    return text


# ----------------------------
# Helpers
# ----------------------------
def extract_json_loose(text: str) -> Dict[str, Any]:
    """
    Extract the first {...} JSON object found.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in output: {text[:200]}...")
    return json.loads(match.group(0))


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter; good enough for heuristics
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


# ----------------------------
# Safety filtering for requests + feedback
# ----------------------------
# Note: Not listing hate slurs here. In a production setting, use a vetted internal list
# or a moderation endpoint. This list focuses on violence/sexual/self-harm/drugs/abuse.
UNSAFE_KEYWORDS = {
    # Violence / harm
    "kill", "murder", "stab", "shoot", "gun", "rifle", "pistol", "knife", "blood", "gore",
    "decapitate", "dismember", "torture", "execute", "strangle", "choke", "poison",
    "kidnap", "abduct", "hostage", "bomb", "explosion", "grenade", "terror", "massacre",
    "beat", "punch", "kick", "assault", "attack", "hurt", "harm", "maim", "brutal",
    "cruel", "sadistic", "abuse", "abusive",
    # Self-harm
    "suicide", "self-harm", "self harm", "cut myself", "cutting", "hang myself", "overdose",
    # Sexual content / adult themes
    "sex", "sexy", "nude", "naked", "porn", "pornography", "erotic", "orgasm", "fetish",
    "rape", "molest", "incest",
    # Drugs / intoxication
    "cocaine", "heroin", "meth", "fentanyl", "opioid", "weed", "marijuana", "lsd", "ecstasy",
    "crack", "high", "stoned", "overdose",
    # Extreme horror
    "horror", "nightmare fuel", "graphic", "gruesome",
}

UNSAFE_PATTERNS = [
    r"\bkill (him|her|them|someone|anyone)\b",
    r"\bmake it (sadistic|cruel|violent|gory|graphic)\b",
    r"\bchild abuse\b",
    r"\bsexual\b",
    r"\brape\b",
    r"\bsuicid(e|al)\b",
    r"\bself[- ]?harm\b",
    r"\boverdose\b",
]


def check_text_safety(text: str) -> Tuple[bool, List[str]]:
    """
    Returns (is_safe, reasons).
    """
    t = text.lower()
    reasons: List[str] = []

    for kw in UNSAFE_KEYWORDS:
        if kw in t:
            reasons.append(f"keyword:{kw}")

    for pat in UNSAFE_PATTERNS:
        if re.search(pat, t):
            reasons.append(f"pattern:{pat}")

    # Also block requests explicitly asking for cruelty/meanness to characters
    if "bully" in t or "bullying" in t or "humiliate" in t:
        reasons.append("keyword:bully/humiliate")

    is_safe = len(reasons) == 0
    return is_safe, reasons


def sanitize_user_text_for_kids(text: str, kind: str) -> Tuple[str, Optional[str]]:
    """
    kind: "request" or "feedback"
    If unsafe, returns a safe rewritten instruction and a note to display.
    """
    safe, reasons = check_text_safety(text)
    if safe:
        return text, None

    if kind == "request":
        note = (
            "Note: I can’t include violent/cruel/explicit themes in a kids (5–10) bedtime story. "
            "I’ll make a safe, gentle version instead."
        )
        # Keep their broad intent but steer to kid-safe
        rewritten = (
            "Create a kids bedtime story prompt (ages 5–10) that keeps the same general idea, "
            "but removes violence/cruelty/adult content and becomes gentle and kind. "
            "If there was conflict, turn it into a misunderstanding that ends in friendship."
        )

        return rewritten, note

    # feedback
    note = (
        "Note: I can’t apply parts of that feedback that add cruelty/violence/explicit content for a kids bedtime "
        "story."
        "I’ll apply a safe alternative (gentle emotion + kindness + cozy ending)."
    )
    safe_alt = (
        "Make it gently emotional in a kid-safe way (a small disappointment), "
        "but ensure kindness, comfort, and a warm/cozy resolution."
    )
    return safe_alt, note


def parse_user_feedback_constraints(feedback: str) -> Dict[str, Any]:
    """
    Use the model to convert messy user feedback into structured constraints.
    Keeps it robust for multiple changes like: 'scary and longer and cuter'.
    Returns JSON dict.
    """
    system = (
        "You convert user feedback about a children's bedtime story into JSON constraints.\n"
        "Bedtime story target age is 5-10. Safety is required.\n"
        "Return VALID JSON ONLY."
    )
    user = (
        f"User feedback: {feedback}\n\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "wants_longer": true/false,\n'
        '  "wants_shorter": true/false,\n'
        '  "wants_funnier": true/false,\n'
        '  "wants_cuter": true/false,\n'
        '  "wants_spooky": true/false,\n'
        '  "wants_sad_ending": true/false,\n'
        '  "wants_harder_words": true/false,\n'
        '  "other_requests": ["any other change requests, short strings"]\n'
        "}\n"
        "Guidance:\n"
        "- If user says sad ending / bittersweet ending: wants_sad_ending=true.\n"
        "- If user says scary/spooky: set wants_spooky=true.\n"
        "- If user says harder words: set wants_harder_words=true.\n"
    )
    raw = call_model([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     max_tokens=350, temperature=0.0)
    return extract_json_loose(raw)


def normalize_constraints_for_kids(constraints: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Enforce bedtime+age safety constraints, and produce notes (for transparency).
    """
    notes = []
    c = dict(constraints)

    if c.get("wants_sad_ending"):
        notes.append("I’ll make the ending gently sad (a small goodbye), but still comforting and safe for bedtime.")

    # "Harder words" conflicts with 5-10 + bedtime simplicity.
    if c.get("wants_harder_words"):
        notes.append("I can’t make the vocabulary harder for a kids bedtime story; I’ll keep words simple and clear.")
        c["wants_harder_words"] = False
        # optional: allow *slightly* richer but still kid-friendly language
        c.setdefault("other_requests", [])
        c["other_requests"].append("Use a tiny bit of variety in words, but still simple and kid-friendly.")

    # Spooky allowed only as "gentle spooky"
    if c.get("wants_spooky"):
        notes.append("I’ll make it gently spooky (spooky-cute), not truly scary, and keep a cozy safe ending.")

    # Longer + shorter conflicts: resolve by most recent mention? Here: longer wins if both true.
    if c.get("wants_longer") and c.get("wants_shorter"):
        notes.append("You asked for longer and shorter; I’ll choose a medium length.")
        c["wants_longer"] = False
        c["wants_shorter"] = False

    return c, notes


# ----------------------------
# Readability / vocabulary heuristics (no external deps)
# ----------------------------
COMMON_WORDS = {
    # Small common-word list to reduce false positives
    "a", "an", "and", "are", "as", "at", "away", "be", "because", "been", "before", "big", "but",
    "by", "can", "came", "come", "could", "day", "did", "do", "down", "each", "even", "every",
    "feel", "find", "for", "found", "from", "fun", "game", "get", "go", "good", "got", "great",
    "had", "has", "have", "he", "her", "here", "him", "his", "home", "how", "i", "in", "is",
    "it", "just", "kind", "know", "laughed", "like", "little", "look", "love", "made", "make",
    "many", "may", "me", "more", "most", "my", "near", "need", "new", "night", "no", "not",
    "now", "of", "off", "oh", "on", "one", "or", "our", "out", "over", "play", "pretty",
    "run", "said", "saw", "see", "she", "sleep", "small", "smile", "so", "some", "soon",
    "story", "sweet", "take", "than", "that", "the", "their", "them", "then", "there",
    "they", "this", "time", "to", "together", "too", "tree", "try", "two", "up", "us",
    "very", "was", "we", "went", "were", "what", "when", "where", "who", "will", "with",
    "you", "your"
}


def analyze_simplicity(story: str) -> Dict[str, Any]:
    """
    Heuristic analysis:
    - flags long sentences (>= 18 words)
    - flags 'advanced' words: length >= 10 or not common and length >= 8
    """
    sents = split_sentences(story)
    sent_lengths = [len(words(s)) for s in sents] if sents else []
    avg_sent = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0.0
    long_sents = [s for s in sents if len(words(s)) >= 18]

    w = words(story)
    advanced = set()
    for token in w:
        if token in COMMON_WORDS:
            continue
        if len(token) >= 10:
            advanced.add(token)
        elif len(token) >= 8 and token.isalpha():
            advanced.add(token)

    # Return sorted lists for stable prompting
    return {
        "avg_sentence_words_est": round(avg_sent, 2),
        "long_sentences_count": len(long_sents),
        "too_advanced_words": sorted(list(advanced))[:40],  # cap to avoid huge prompts
    }


# ----------------------------
# Story spec parsing
# ----------------------------
def build_story_spec(user_request: str) -> Dict[str, Any]:
    spec = {
        "age_min": 5,
        "age_max": 10,
        "length": "medium",  # short | medium | long
        "tone": "cozy, warm, gentle, bedtime",
        "humor": "light",  # light | high
        "must_include": user_request.strip(),
        "theme": "bedtime story",
    }

    req_lower = user_request.lower()
    if any(w in req_lower for w in ["short", "quick", "tiny", "brief"]):
        spec["length"] = "short"
    if any(w in req_lower for w in ["long", "longer", "detailed", "chapter"]):
        spec["length"] = "long"
    if any(w in req_lower for w in ["funny", "silly", "goofy", "hilarious", "jokes"]):
        spec["tone"] = "cozy, warm, gentle, bedtime, funny"
        spec["humor"] = "high"

    return spec


def target_word_count(length: str) -> int:
    return {"short": 350, "medium": 650, "long": 1000}.get(length, 650)


def interpret_feedback_target_words(feedback: str, current_words: int, fallback_length: str) -> int:
    fb = feedback.lower()
    m = re.search(r"(\d{3,4})\s*words?", fb)
    if m:
        return int(m.group(1))
    base = target_word_count(fallback_length)

    if any(x in fb for x in ["much longer", "way longer", "a lot longer", "whole extra section"]):
        return max(base, current_words + 350)

    if any(x in fb for x in ["longer", "more detail", "add detail", "add more"]):
        return max(base, current_words + 200)

    if any(x in fb for x in ["much shorter", "way shorter", "quick", "tiny"]):
        return max(180, int(current_words * 0.55))

    if any(x in fb for x in ["shorter", "trim", "condense"]):
        return max(220, int(current_words * 0.75))

    return target_word_count(fallback_length)


def update_spec_with_feedback(spec: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    f = feedback.lower()
    new_spec = dict(spec)

    if any(x in f for x in ["less scary", "not scary", "gentler", "calmer", "softer", "more cozy", "more bedtime"]):
        new_spec["tone"] = "extra cozy, warm, gentle, comforting, bedtime"

    if any(x in f for x in ["funnier", "more funny", "add jokes", "sillier", "goofier"]):
        new_spec["tone"] = "cozy, warm, gentle, bedtime, funny"
        new_spec["humor"] = "high"

    if any(x in f for x in ["less funny", "more serious", "quiet", "peaceful"]):
        new_spec["humor"] = "light"
        new_spec["tone"] = "cozy, warm, gentle, bedtime"

    if "sad" in f or "bittersweet" in f or "emotional" in f:
        new_spec["tone"] = "cozy, warm, gentle, bedtime, softly emotional, comforting"

    if any(x in f for x in ["short version", "keep it short"]):
        new_spec["length"] = "short"
    if any(x in f for x in ["long version", "chapter", "more detailed"]):
        new_spec["length"] = "long"

    return new_spec


def remove_discouraged_words(text: str) -> str:
    # Not unsafe per se, but avoid inserting these into bedtime stories
    discouraged = ["cruel", "kill", "blood", "gore"]
    out = text
    for w in discouraged:
        out = re.sub(rf"\b{re.escape(w)}\b", "mean", out, flags=re.IGNORECASE)
    return out


# ----------------------------
# NEW: Story planning step (agent design improvement)
# ----------------------------
def story_plan_messages(spec: Dict[str, Any]) -> List[Dict[str, str]]:
    wc = target_word_count(spec["length"])
    humor = spec.get("humor", "light")

    system = (
        "You are a planning assistant for children's bedtime stories (ages 5–10).\n"
        "Return VALID JSON ONLY.\n"
        "Your plan must be kid-safe, gentle, and end cozy.\n"
        "Use simple concepts suitable for kids.\n"
    )

    user = (
        f"Create a story plan for this request:\n"
        f"- Must include: {spec['must_include']}\n"
        f"- Tone: {spec['tone']}\n"
        f"- Humor level: {humor}\n"
        f"- Target length: ~{wc} words\n\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "title": "string",\n'
        '  "setting": "string",\n'
        '  "main_characters": ["list of names + 1 short trait each"],\n'
        '  "problem": "string",\n'
        '  "attempts": ["3 short attempt beats"],\n'
        '  "running_gag": "string (kid-safe, optional but recommended if funny)",\n'
        '  "resolution": "string",\n'
        '  "bedtime_ending_image": "1 comforting final image",\n'
        '  "vocab_style": "simple, common words; short sentences"\n'
        "}\n"
        "Make sure the plan is clearly structured and has a full arc."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def create_story_plan(spec: Dict[str, Any]) -> Dict[str, Any]:
    raw = call_model(story_plan_messages(spec), max_tokens=600, temperature=0.2)
    return extract_json_loose(raw)


# ----------------------------
# Prompt builders
# ----------------------------
def storyteller_messages(spec: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, str]]:
    wc = target_word_count(spec["length"])

    humor_instructions = ""
    if spec.get("humor") == "high":
        humor_instructions = (
            "- Make it genuinely funny for kids with 3+ laugh moments.\n"
            "- Include: (1) a silly misunderstanding, (2) a gentle physical gag, (3) a callback joke near the end.\n"
            "- Use the running gag from the plan.\n"
        )

    system = (
        "You are a world-class children's bedtime storyteller.\n"
        "Write a story appropriate for ages 5 to 10.\n"
        "Hard rules:\n"
        "- Keep content safe and gentle (no violence, cruelty, explicit romance, adult themes, intense horror).\n"
        "- Use simple vocabulary suitable for ages 5–10.\n"
        "- Prefer short, common words; avoid rare/advanced words.\n"
        "- Keep sentences short (about 8–14 words on average).\n"
        "- Use short paragraphs.\n"
        "- Include a clear arc: setup -> problem -> attempts -> resolution -> cozy bedtime ending.\n"
        "- End with a soothing final paragraph that helps the child feel calm and sleepy.\n"
        f"{humor_instructions}"
        "Do NOT mention that you used a plan.\n"
    )

    user = (
        f"Story requirements:\n"
        f"- Target age: {spec['age_min']}-{spec['age_max']}\n"
        f"- Tone: {spec['tone']}\n"
        f"- Approx length: ~{wc} words\n"
        f"- Must include: {spec['must_include']}\n\n"
        f"STORY PLAN (JSON):\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
        f"Now write the story following the plan closely."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- Judge (initial story) with vocab + simplicity fields ---
def judge_messages(spec: Dict[str, Any], story: str, heuristic: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are a strict judge evaluating a bedtime story for children ages 5–10.\n"
        "You must return VALID JSON ONLY. No markdown, no commentary.\n"
        "Evaluate:\n"
        "- Safety and age appropriateness (5–10)\n"
        "- Clarity and simplicity\n"
        "- Vocabulary level (avoid rare/advanced words)\n"
        "- Sentence simplicity (short sentences)\n"
        "- Story arc (setup/problem/attempts/resolution)\n"
        "- Bedtime vibe (cozy, calming ending)\n"
        "- Creativity and charm\n"
        "If there are safety concerns, mark is_safe=false.\n"
    )
    user = (
        f"Target age: {spec['age_min']}-{spec['age_max']}\n"
        f"Story requirements: {spec['must_include']}\n\n"
        f"Heuristic simplicity signals (not perfect, use as hints):\n"
        f"{json.dumps(heuristic, indent=2)}\n\n"
        f"STORY:\n{story}\n\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "overall_score": 1-10,\n'
        '  "age_fit_score": 1-10,\n'
        '  "clarity_score": 1-10,\n'
        '  "vocabulary_score": 1-10,\n'
        '  "sentence_simplicity_score": 1-10,\n'
        '  "arc_score": 1-10,\n'
        '  "bedtime_score": 1-10,\n'
        '  "creativity_score": 1-10,\n'
        '  "is_safe": true/false,\n'
        '  "too_advanced_words": ["list of words to simplify"],\n'
        '  "long_sentences_count": number,\n'
        '  "avg_sentence_words_est": number,\n'
        '  "must_fix": [list of concrete issues],\n'
        '  "suggested_edits": [list of specific improvements],\n'
        '  "pass": true/false\n'
        "}\n"
        "Set pass=true only if overall_score>=8 AND is_safe=true AND bedtime_score>=8 "
        "AND vocabulary_score>=8 AND sentence_simplicity_score>=8."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rewriter_messages(spec: Dict[str, Any], story: str, judge_json: Dict[str, Any], heuristic: Dict[str, Any])\
        -> List[Dict[str, str]]:
    wc = target_word_count(spec["length"])
    advanced = judge_json.get("too_advanced_words") or heuristic.get("too_advanced_words") or []

    system = (
        "You are a children's bedtime story editor.\n"
        "Revise the story to address all issues in the judge feedback.\n"
        "Keep it appropriate for ages 5–10 and end with a cozy bedtime tone.\n"
        "Use simple, common words and short sentences.\n"
        "Return ONLY the revised story text.\n"
    )
    user = (
        f"Target age: {spec['age_min']}-{spec['age_max']}\n"
        f"Tone: {spec['tone']}\n"
        f"Approx length: ~{wc} words\n"
        f"Must include: {spec['must_include']}\n\n"
        f"Words to simplify (replace with easier words): {advanced}\n"
        f"Heuristic simplicity signals: {json.dumps(heuristic)}\n\n"
        f"JUDGE FEEDBACK (JSON):\n{json.dumps(judge_json, indent=2)}\n\n"
        f"ORIGINAL STORY:\n{story}\n\n"
        "Now write the revised story, fixing every item in must_fix and applying suggested_edits where helpful."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------
# Feedback revision prompts + judge
# ----------------------------
def feedback_rewriter_messages(
    spec: Dict[str, Any],
    original_story: str,
    feedback: str,
    target_words: int,
    heuristic: Dict[str, Any],
) -> List[Dict[str, str]]:
    humor_instructions = ""
    if spec.get("humor") == "high":
        humor_instructions = (
            "Humor requirement: include 3+ kid-safe funny beats (silly misunderstanding, gentle gag, callback joke).\n"
        )

    system = (
        "You are a children's bedtime story editor.\n"
        "Revise the story based on the user's feedback.\n"
        "Rules:\n"
        "- Keep it appropriate for ages 5–10.\n"
        "- Preserve the SAME characters, setting, and core plot as the original story,\n"
        "  UNLESS the user's feedback explicitly asks to change them.\n"
        "- Use simple, common words and short sentences.\n"
        "- End with a cozy bedtime paragraph.\n"
        "- Return ONLY the revised story text.\n"
    )

    user = (
        f"Target age: {spec['age_min']}-{spec['age_max']}\n"
        f"Tone: {spec['tone']}\n"
        f"{humor_instructions}"
        f"Target length: ~{target_words} words (within ~10–15% is fine)\n"
        f"Heuristic simplicity signals: {json.dumps(heuristic)}\n\n"
        f"USER FEEDBACK:\n{feedback}\n\n"
        f"ORIGINAL STORY:\n{original_story}\n\n"
        "Now produce the revised story."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def feedback_judge_messages(
    spec: Dict[str, Any],
    original_story: str,
    revised_story: str,
    feedback: str,
    target_words: int,
    heuristic: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = (
        "You are a strict judge evaluating a revised children's bedtime story.\n"
        "Return VALID JSON ONLY.\n"
        "Check:\n"
        "- Age-appropriateness (5–10) and safety\n"
        "- Clear arc and cozy bedtime ending\n"
        "- Follows the user's feedback\n"
        "- Preserves the original story unless feedback asked to change it\n"
        "- Vocabulary simplicity and short sentences\n"
        "- Roughly matches requested length\n"
    )

    user = (
        f"Target age: {spec['age_min']}-{spec['age_max']}\n"
        f"Target length: ~{target_words} words\n"
        f"User feedback: {feedback}\n"
        f"Heuristic simplicity signals: {json.dumps(heuristic)}\n\n"
        f"ORIGINAL STORY:\n{original_story}\n\n"
        f"REVISED STORY:\n{revised_story}\n\n"
        f"Constraints: {feedback}\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "overall_score": 1-10,\n'
        '  "follows_feedback_score": 1-10,\n'
        '  "preserves_core_story_score": 1-10,\n'
        '  "bedtime_score": 1-10,\n'
        '  "clarity_score": 1-10,\n'
        '  "vocabulary_score": 1-10,\n'
        '  "spooky_score": 1-10,\n'
        '  "constraints_met": true/false,\n'
        '  "missing_constraints": [list of missing items],\n'
        '  "sentence_simplicity_score": 1-10,\n'
        '  "is_safe": true/false,\n'
        '  "too_advanced_words": ["list of words to simplify"],\n'
        '  "long_sentences_count": number,\n'
        '  "avg_sentence_words_est": number,\n'
        '  "must_fix": [list of concrete issues],\n'
        '  "suggested_edits": [list of specific improvements],\n'
        '  "pass": true/false\n'
        "}\n"
        "Set pass=true only if overall_score>=8 AND is_safe=true AND bedtime_score>=8 "
        "AND follows_feedback_score>=8 AND vocabulary_score>=8 AND sentence_simplicity_score>=8 "
        "AND constraints_met=true."

    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------
# Core pipeline (initial generation + judge loop)
# ----------------------------
def generate_story(spec: Dict[str, Any], plan: Dict[str, Any]) -> str:
    return call_model(storyteller_messages(spec, plan), max_tokens=1400, temperature=0.85)


def judge_story(spec: Dict[str, Any], story: str) -> Dict[str, Any]:
    heuristic = analyze_simplicity(story)
    raw = call_model(judge_messages(spec, story, heuristic), max_tokens=800, temperature=0.0)
    return extract_json_loose(raw)


def revise_story(spec: Dict[str, Any], story: str, judge_json: Dict[str, Any]) -> str:
    heuristic = analyze_simplicity(story)
    return call_model(rewriter_messages(spec, story, judge_json, heuristic), max_tokens=1400, temperature=0.7)


def story_with_judge_loop(user_request: str, max_rounds: int = 3) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (best_story, spec_used).
    """
    # Safety sanitize initial request
    safe_request, note = sanitize_user_text_for_kids(user_request, kind="request")
    if note:
        print("\n" + note + "\n")

    spec = build_story_spec(safe_request)

    # Plan first (reduces generic outputs)
    plan = create_story_plan(spec)

    best_story = ""
    best_score = -1

    story = generate_story(spec, plan)

    for _ in range(max_rounds):
        judgment = judge_story(spec, story)
        score = int(judgment.get("overall_score", 0))
        is_pass = bool(judgment.get("pass", False))

        target = target_word_count(spec["length"])
        lower = int(target * 0.90)
        upper = int(target * 1.15)
        wc = count_words(story)
        length_ok = (lower <= wc <= upper)

        if is_pass and length_ok:
            return story, spec

        length_fix = ""
        if wc < lower:
            length_fix = f"Add more detail and extra beats to reach about {target} words."
        elif wc > upper:
            length_fix = f"Trim to about {target} words while keeping the arc."

        if length_fix:
            # inject a length-only must-fix so the editor reliably expands
            judgment.setdefault("must_fix", [])
            judgment["must_fix"].append(length_fix)

        if score > best_score:
            best_score = score
            best_story = story

        if not judgment.get("is_safe", True):
            spec["tone"] = "extra cozy, warm, gentle, comforting, bedtime"

        story = revise_story(spec, story, judgment)

    return best_story, spec


# ----------------------------
# Feedback-aware revision loop (preserve story; judge verifies feedback + simplicity)
# ----------------------------

SPOOKY_WORDS = {"scary", "spooky", "creepy", "haunted", "ghost", "boo", "monster", "skeleton", "witch"}
LONGER_WORDS = {"longer", "long", "more detail", "add detail", "add more", "extra section", "chapter"}
SHORTER_WORDS = {"shorter", "short", "quick", "tiny", "brief", "condense", "trim"}
FUNNY_WORDS = {"funnier", "funny", "silly", "goofy", "jokes", "more funny"}
CUTE_WORDS = {"cute", "cuter", "adorable", "sweet", "cuddly"}
SAD_WORDS = {"sad", "sadder", "tear", "teary", "bittersweet", "emotional", "melancholy", "sad ending"}


def contains_any(text: str, phrases: set) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)


def revise_story_with_feedback_loop(
    base_spec: Dict[str, Any],
    original_story: str,
    feedback: str,
    max_rounds: int = 3,
) -> str:
    # Safety sanitize feedback (blocks cruelty/violence/explicit)
    safe_feedback, safety_note = sanitize_user_text_for_kids(feedback, kind="feedback")
    if safety_note:
        print("\n" + safety_note + "\n")
    feedback = safe_feedback

    # Parse multiple requests into constraints
    constraints = parse_user_feedback_constraints(feedback)

    # --- deterministic overrides to prevent LLM misclassification ---
    if not contains_any(feedback, SPOOKY_WORDS):
        constraints["wants_spooky"] = False

    # also reinforce obvious length signals
    if contains_any(feedback, LONGER_WORDS):
        constraints["wants_longer"] = True
    if contains_any(feedback, SHORTER_WORDS):
        constraints["wants_shorter"] = True

    if contains_any(feedback, FUNNY_WORDS):
        constraints["wants_funnier"] = True
    if contains_any(feedback, CUTE_WORDS):
        constraints["wants_cuter"] = True

    if contains_any(feedback, SAD_WORDS):
        constraints["wants_sad_ending"] = True

    constraints, constraint_notes = normalize_constraints_for_kids(constraints)
    for n in constraint_notes:
        print("\n" + n + "\n")

    spec = update_spec_with_feedback(base_spec, feedback)

    # Apply constraint-driven tone tweaks
    if constraints.get("wants_cuter"):
        spec["tone"] = "extra cute, cozy, warm, gentle, bedtime"
    if constraints.get("wants_funnier"):
        spec["humor"] = "high"
        spec["tone"] = "cozy, warm, gentle, bedtime, funny"
    if constraints.get("wants_spooky"):
        # safe spooky mode
        spec["tone"] = "cozy, warm, gentle, bedtime, softly spooky, playful, reassuring"


    current_words = count_words(original_story)
    target_words = interpret_feedback_target_words(feedback, current_words, spec["length"])

    # Make length bounds stricter so "longer" actually happens
    lower = int(target_words * 0.90)
    upper = int(target_words * 1.15)

    heuristic0 = analyze_simplicity(original_story)

    # Build explicit instruction block so multiple constraints stick
    constraint_text = []
    if constraints.get("wants_longer"):
        constraint_text.append(f"Make it longer to roughly {target_words} words.")
    if constraints.get("wants_shorter"):
        constraint_text.append(f"Make it shorter to roughly {target_words} words.")
    if constraints.get("wants_cuter"):
        constraint_text.append("Make it cuter (more adorable details, sweet interactions).")
    if constraints.get("wants_funnier"):
        constraint_text.append("Make it funnier with 3+ kid-safe funny moments.")
    if constraints.get("wants_spooky"):
        constraint_text.append("Make it gently spooky (spooky-cute): friendly shadows, silly 'boo', never intense.")
    if constraints.get("other_requests"):
        constraint_text.extend([f"- {x}" for x in constraints["other_requests"]])
    if constraints.get("wants_sad_ending"):
        constraint_text.append(
            "Make the ending gently sad (a small goodbye or missed moment), "
            "but include comfort, kindness, and a cozy final bedtime paragraph."
        )

    feedback_with_constraints = feedback + "\n\nConstraints:\n" + "\n".join(constraint_text)

    revised = call_model(
        feedback_rewriter_messages(spec, original_story, feedback_with_constraints, target_words, heuristic0),
        max_tokens=2000,
        temperature=0.75,
    )

    best_story = revised
    best_score = -1

    for _ in range(max_rounds):
        heuristic = analyze_simplicity(revised)

        raw_j = call_model(
            feedback_judge_messages(spec, original_story, revised, feedback_with_constraints, target_words, heuristic),
            max_tokens=900,
            temperature=0.0,
        )
        judgment = extract_json_loose(raw_j)

        score = int(judgment.get("overall_score", 0))
        is_pass = bool(judgment.get("pass", False))

        # Extra length check (hard gate)
        wc = count_words(revised)
        length_ok = (lower <= wc <= upper)

        if score > best_score:
            best_score = score
            best_story = revised

        if is_pass and length_ok:
            return revised

        # If unsafe, tighten
        if not judgment.get("is_safe", True):
            spec["tone"] = "extra cozy, warm, gentle, comforting, bedtime"

        # If length off, add explicit edit instruction
        length_fix = ""
        if wc < lower:
            length_fix = f"Add more detail and a few extra beats to reach ~{target_words} words."
        elif wc > upper:
            length_fix = f"Trim and condense to reach ~{target_words} words without losing the arc."

        advanced = judgment.get("too_advanced_words") or heuristic.get("too_advanced_words") or []
        combined_feedback = (
            f"{feedback_with_constraints}\n\n"
            f"{length_fix}\n"
            f"Fix these must-fix items: {judgment.get('must_fix', [])}\n"
            f"Apply suggested edits: {judgment.get('suggested_edits', [])}\n"
            f"Simplify these words: {advanced}\n"
            f"Keep sentences short."
        )

        revised = call_model(
            feedback_rewriter_messages(spec, original_story, combined_feedback, target_words, heuristic),
            max_tokens=2000,
            temperature=0.7,
        )

    return best_story


# ----------------------------
# CLI
# ----------------------------
def main():
    user_input = input("What kind of story do you want to hear? ").strip()
    if not user_input:
        print("Please enter a story request.")
        return

    story, spec_used = story_with_judge_loop(user_input, max_rounds=3)
    story = remove_discouraged_words(story)

    print("\n" + "=" * 60 + "\n")
    print(story)
    print("\n" + "=" * 60 + "\n")

    # Keep revising same story
    while True:
        feedback = input(
            "Want any changes? (e.g., shorter, funnier, add dragons, calmer, different ending) Press Enter to finish: "
        ).strip()
        if not feedback:
            break

        story = revise_story_with_feedback_loop(spec_used, story, feedback, max_rounds=2)
        story = remove_discouraged_words(story)
        print("\n" + "=" * 60 + "\n")
        print(story)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
