"""
wordle.py  —  run after precompute.py
──────────────────────────────────────
Scoring formula (3Blue1Brown's architecture):

  E[score] = p × attempt  +  (1−p) × (attempt + f(H_rem − H))
           = attempt  +  (1−p) × f(H_rem − H)

  p       = probability this specific word IS the answer
            = its weight in the current Bayesian distribution
            = 0 for words not in the remaining answer pool
  attempt = current guess number (1–6) — urgency grows as guesses burn
  H       = weighted entropy of this guess vs current remaining pool
  H_rem   = Shannon entropy of the current weighted distribution
  f(·)    = empirical map: residual entropy → avg guesses still needed after
            (built by simulation in precompute.py to match this formula)

We MINIMISE E[score].  The formula is self-adjusting:
  — Early game: p ≈ 0 for all words, so E[score] ≈ attempt + f(H_rem − H).
    Minimising this means maximising H — pure information gathering.
  — Late game: remaining answers have grown p values, so they get a natural
    discount of p × f(H_rem − H) over non-answer words with the same H.
    No hard switch needed — the formula handles it continuously.
"""

import pickle, math, random
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PRECOMPUTED_PATH = "precomputed.pkl"
TOP_N            = 5
WORD_LEN         = 5
MAX_GUESSES      = 6
ALL_GREEN        = 242

# ══════════════════════════════════════════════════════════════════════════════
#  ANSI
# ══════════════════════════════════════════════════════════════════════════════

GREEN   = "\033[42m\033[30m"
YELLOW  = "\033[43m\033[30m"
GRAY    = "\033[100m\033[97m"
CYAN    = "\033[46m\033[30m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"
FILL, EMPTY = "█", "░"

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_precomputed(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"\n  {YELLOW} ! {RESET}  '{path}' not found.")
        print(f"  Run  {BOLD}python precompute.py{RESET}  first.\n")
        raise SystemExit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  PATTERN HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_pattern_int(guess: str, answer: str) -> int:
    p = [0] * WORD_LEN; pool = list(answer)
    for i in range(WORD_LEN):
        if guess[i] == answer[i]: p[i] = 2; pool[i] = None
    for i in range(WORD_LEN):
        if p[i] == 2: continue
        if guess[i] in pool: p[i] = 1; pool[pool.index(guess[i])] = None
    return p[0] + p[1]*3 + p[2]*9 + p[3]*27 + p[4]*81

def decode_pattern_int(code: int) -> tuple:
    return tuple((code // (3**i)) % 3 for i in range(WORD_LEN))

# ══════════════════════════════════════════════════════════════════════════════
#  BAYESIAN DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def make_weights(remaining_idx: list, freq_sigmoid: dict, answers: list) -> dict:
    """
    p(answer=w) ∝ freq_sigmoid[w], renormalised over surviving words.
    Filtering remaining_idx by pattern consistency IS the Bayesian update.
    Returns {answer_idx: probability}.
    """
    raw   = {ai: freq_sigmoid.get(answers[ai], 0.5) for ai in remaining_idx}
    total = sum(raw.values()) or 1.0
    return {ai: w / total for ai, w in raw.items()}

# ══════════════════════════════════════════════════════════════════════════════
#  ENTROPY
# ══════════════════════════════════════════════════════════════════════════════

def weighted_entropy(vi: int, weights: dict, matrix: list) -> float:
    """
    H(guess) = −Σ p(pattern)·log₂p(pattern)
    p(pattern) = Σ p(w) for all w that produce that pattern.
    Uses the Bayesian-weighted distribution, not uniform 1/N.
    """
    row = matrix[vi]; pp: dict = defaultdict(float)
    for ai, p in weights.items():
        pp[row[ai]] += p
    return max(0.0, -sum(p * math.log2(p) for p in pp.values() if p > 0))

def h_of_weights(weights: dict) -> float:
    """Current uncertainty = Shannon entropy of the weighted distribution."""
    return max(0.0, -sum(p * math.log2(p) for p in weights.values() if p > 0))

# ══════════════════════════════════════════════════════════════════════════════
#  f-TABLE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def lookup_f(residual: float, f_table: dict, bucket_size: float) -> float:
    """Residual entropy → expected guesses after this step."""
    b = int(max(0.0, residual) / bucket_size)
    if b in f_table:
        return f_table[b]
    if not f_table:
        return residual / 2.0
    return f_table[min(f_table, key=lambda k: abs(k - b))]

# ══════════════════════════════════════════════════════════════════════════════
#  SCORING  —  minimise E[score]
#
#  E[score] = attempt + (1−p) × f(H_rem − H)
#
#  p  = weights[answer_idx_of_this_word]  if word is a remaining answer
#     = 0                                  otherwise
#
#  Self-adjusting behaviour:
#    p=0 (non-answer or early game): score = attempt + f(H_rem−H) → maximise H
#    p>0 (remaining answer):         score reduced by p×f(H_rem−H) → prefers answers
#      as remaining pool shrinks, each answer's p grows → discount grows → they
#      naturally bubble to the top without any hard switch
# ══════════════════════════════════════════════════════════════════════════════

def score_word(vi: int, word: str,
               weights: dict, H_rem: float, attempt: int,
               matrix: list, f_table: dict, bucket_size: float,
               vi_to_ai: dict) -> tuple:
    """
    Returns (E_score, H, p).
    Lower E_score = better guess.
    """
    H        = weighted_entropy(vi, weights, matrix)
    ai       = vi_to_ai.get(vi)                        # None if not an answer word
    p        = weights.get(ai, 0.0) if ai is not None else 0.0
    residual = max(0.0, H_rem - H)
    f_val    = lookup_f(residual, f_table, bucket_size)

    e_score  = float(attempt) + (1.0 - p) * f_val
    return e_score, H, p

def top_suggestions(vocabulary: list, vocab_indices: list,
                    weights: dict, H_rem: float, attempt: int,
                    matrix: list, f_table: dict, bucket_size: float,
                    vi_to_ai: dict, n: int = TOP_N) -> list:
    """
    Score all vocab words, return top-n sorted by E[score] ascending.
    (Lower expected score = fewer guesses = better.)
    """
    scored = []
    for vi in vocab_indices:
        e, H, p = score_word(vi, vocabulary[vi], weights, H_rem, attempt,
                             matrix, f_table, bucket_size, vi_to_ai)
        scored.append((vocabulary[vi], e, H, p))

    scored.sort(key=lambda x: x[1])   # ascending — minimise expected guesses
    return scored[:n]

# ══════════════════════════════════════════════════════════════════════════════
#  RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def mini_bar(val: float, lo: float, hi: float, w: int = 10) -> str:
    """Bar where lo=full, hi=empty (lower E[score] is better → fuller bar)."""
    span = hi - lo
    frac = 1.0 - ((val - lo) / span) if span > 0 else 1.0
    frac = max(0.0, min(1.0, frac))
    f    = round(frac * w)
    return f"{CYAN}{FILL*f}{EMPTY*(w-f)}{RESET}"

def render_tiles(guess: str, code: int) -> str:
    pat    = decode_pattern_int(code)
    colors = [GREEN if p==2 else YELLOW if p==1 else GRAY for p in pat]
    return "".join(f"{c} {l.upper()} {RESET}" for l, c in zip(guess, colors))

def render_empty_row() -> str:
    return f"{DIM} ·   ·   ·   ·   · {RESET}"

def render_keyboard(used: dict) -> str:
    rows  = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    lines = []
    for row in rows:
        line = "  "
        for ch in row:
            line += f"{used.get(ch, DIM)} {ch.upper()} {RESET}"
        lines.append(line)
    return "\n".join(lines)

def render_suggestions(suggestions: list, remaining_words: set,
                       H_rem: float, attempt: int) -> str:
    if not suggestions:
        return ""

    # Bar: invert so best (lowest E) gets full bar
    scores = [s[1] for s in suggestions]
    lo, hi = min(scores), max(scores)

    lines = [
        f"  {BOLD}Top {len(suggestions)} suggestions{RESET}  "
        f"{DIM}H_rem={H_rem:.3f} bits  guess {attempt}/{MAX_GUESSES}"
        f"  ↓ lower E[score] = better{RESET}",
        f"\n  {DIM}{'word':<8}  {'E[score]':>9}  {'H bits':>7}  {'p(ans)':>7}"
        f"  {'bar':<12}  type{RESET}",
        f"  {'─'*60}",
    ]
    for word, e, H, p in suggestions:
        b     = mini_bar(e, lo, hi)
        valid = (f"{GREEN} answer {RESET}" if word in remaining_words
                 else f"{DIM}  guess  {RESET}")
        lines.append(
            f"  {BOLD}{word.upper():<8}{RESET}"
            f"  {e:9.4f}"
            f"  {H:7.4f}b"
            f"  {p:7.4f}"
            f"  {b}"
            f"  {valid}"
        )
    lines.append(
        f"\n  {DIM}p(ans) = Bayesian probability this word is the answer  |  "
        f"{len(remaining_words):,} word{'s' if len(remaining_words)!=1 else ''} remaining{RESET}"
    )
    return "\n".join(lines)

def header():
    print(f"\n{BOLD}{'═'*56}")
    print("   W O R D L E   ·   E N T R O P Y   ·   B A Y E S")
    print(f"{'═'*56}{RESET}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  GAME LOOP
# ══════════════════════════════════════════════════════════════════════════════

COLOR_RANK = {GRAY: 0, YELLOW: 1, GREEN: 2}

def play(data: dict):
    vocabulary   = data["vocabulary"]
    answers      = data["answers"]
    vocab_index  = data["vocab_index"]
    answer_index = data["answer_index"]
    matrix       = data["pattern_matrix"]
    freq_sigmoid = data["freq_sigmoid"]
    f_table      = data["f_table"]
    bucket_size  = data["bucket_size"]

    # Map vocab_index → answer_index (only for words that are answers)
    # Used to look up p(word is answer) = weights[ai]
    vi_to_ai = {vocab_index[w]: ai
                for ai, w in enumerate(answers) if w in vocab_index}

    answer      = random.choice(answers)
    answer_ai   = answer_index[answer]
    total       = len(answers)

    remaining_idx = list(range(total))
    vocab_indices = list(range(len(vocabulary)))
    history       = []
    keyboard      = {}

    header()
    p_ans = freq_sigmoid.get(answer, 0.5)
    print(f"  {DIM}[DEBUG]  Answer → {RESET}{BOLD}{answer.upper()}{RESET}  "
          f"{DIM}(freq_sigmoid = {p_ans:.3f}){RESET}\n")
    print(f"  Answer pool : {total:,} words")
    print(f"  Vocabulary  : {len(vocabulary):,} words\n")

    for attempt in range(1, MAX_GUESSES + 1):

        # ── Dynamic state ─────────────────────────────────────────────────────
        weights = make_weights(remaining_idx, freq_sigmoid, answers)
        H_rem   = h_of_weights(weights)

        # ── Board ─────────────────────────────────────────────────────────────
        print(f"  {'─'*31}")
        for g, code in history:
            print("    ", render_tiles(g, code))
        for _ in range(MAX_GUESSES - len(history)):
            print("    ", render_empty_row())
        print(f"  {'─'*31}\n")

        # ── Keyboard ──────────────────────────────────────────────────────────
        print(render_keyboard(keyboard))
        print()

        # ── Suggestions ───────────────────────────────────────────────────────
        remaining_words = {answers[ai] for ai in remaining_idx}
        suggestions     = top_suggestions(
            vocabulary, vocab_indices, weights, H_rem, attempt,
            matrix, f_table, bucket_size, vi_to_ai, TOP_N
        )
        print(render_suggestions(suggestions, remaining_words, H_rem, attempt))
        print()

        # ── Input ─────────────────────────────────────────────────────────────
        while True:
            raw = input(f"  Guess {attempt}/{MAX_GUESSES}: ").strip().lower()
            if len(raw) == WORD_LEN and raw.isalpha():
                break
            print(f"  {YELLOW} ! {RESET}  Enter a {WORD_LEN}-letter word.\n")

        # ── Pattern ───────────────────────────────────────────────────────────
        if raw in vocab_index:
            vi           = vocab_index[raw]
            pattern_code = matrix[vi][answer_ai]
        else:
            vi           = None
            pattern_code = compute_pattern_int(raw, answer)

        history.append((raw, pattern_code))

        # ── Bayesian update: filter remaining, renormalise next turn ──────────
        old_size = len(remaining_idx)
        if vi is not None:
            row           = matrix[vi]
            remaining_idx = [ai for ai in remaining_idx if row[ai] == pattern_code]
        else:
            remaining_idx = [ai for ai in remaining_idx
                             if compute_pattern_int(raw, answers[ai]) == pattern_code]

        new_size    = len(remaining_idx)
        new_weights = make_weights(remaining_idx, freq_sigmoid, answers) if remaining_idx else {}
        H_new       = h_of_weights(new_weights)
        bits_gained = max(0.0, H_rem - H_new)

        # ── Keyboard ──────────────────────────────────────────────────────────
        pat_vals  = decode_pattern_int(pattern_code)
        color_map = [GREEN if p==2 else YELLOW if p==1 else GRAY for p in pat_vals]
        for letter, color in zip(raw, color_map):
            prev = keyboard.get(letter, GRAY)
            if COLOR_RANK.get(color, 0) > COLOR_RANK.get(prev, 0):
                keyboard[letter] = color

        print(f"\n  {CYAN}Δ{bits_gained:.3f} bits{RESET}  "
              f"H_rem  {H_rem:.3f} → {H_new:.3f}  |  "
              f"{new_size:,} word{'s' if new_size != 1 else ''} remaining\n")

        # ── Win ───────────────────────────────────────────────────────────────
        if pattern_code == ALL_GREEN:
            print(f"  {'─'*31}")
            for g, code in history:
                print("    ", render_tiles(g, code))
            print(f"  {'─'*31}\n")
            msgs = ["Genius!", "Magnificent!", "Impressive!", "Splendid!", "Great!", "Phew!"]
            print(f"  {GREEN} {msgs[attempt-1]} {RESET}  "
                  f"Solved in {attempt} guess{'es' if attempt > 1 else ''}!\n")
            return

    # ── Out of guesses ────────────────────────────────────────────────────────
    print(f"  {'─'*31}")
    for g, code in history:
        print("    ", render_tiles(g, code))
    print(f"  {'─'*31}\n")
    print(f"  {GRAY} Out of guesses! {RESET}  The word was {BOLD}{answer.upper()}{RESET}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{BOLD}  Loading precomputed data…{RESET}", end=" ", flush=True)
    data = load_precomputed(PRECOMPUTED_PATH)
    print(f"{GREEN}done{RESET}")
    print(f"  {DIM}{len(data['vocabulary']):,} vocab  ·  "
          f"{len(data['answers']):,} answers  ·  "
          f"{len(data['f_table'])} f-table buckets{RESET}\n")

    while True:
        play(data)
        again = input("  Play again? (y/n): ").strip().lower()
        if again != "y":
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            break
        print()