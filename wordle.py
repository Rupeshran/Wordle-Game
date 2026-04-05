import random
import math
import time
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURE YOUR WORD LIST PATHS HERE
# ══════════════════════════════════════════════════════════════════════════════

ANSWERS_PATH = "data/allowed_words.txt"      # words the game will pick from
GUESSES_PATH = "data/possible_words.txt"      # full vocabulary used for entropy scoring
                                   # (should be a superset of ANSWERS_PATH)

TOP_N        = 5                   # how many entropy suggestions to show
WORD_LEN     = 5
MAX_GUESSES  = 6

# ══════════════════════════════════════════════════════════════════════════════
#  ANSI colours
# ══════════════════════════════════════════════════════════════════════════════

GREEN  = "\033[42m\033[30m"
YELLOW = "\033[43m\033[30m"
GRAY   = "\033[100m\033[97m"
CYAN   = "\033[46m\033[30m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

BAR_FILL  = "█"
BAR_EMPTY = "░"

# ══════════════════════════════════════════════════════════════════════════════
#  Word loading
# ══════════════════════════════════════════════════════════════════════════════

def load_words(path: str) -> list:
    try:
        with open(path) as f:
            words = [w.strip().lower() for w in f
                     if len(w.strip()) == WORD_LEN and w.strip().isalpha()]
        if not words:
            raise ValueError("No valid 5-letter words found.")
        return words
    except FileNotFoundError:
        print(f"\n  {YELLOW} ! {RESET}  Could not find '{path}'.")
        print(f"  Check the path constants at the top of the script.\n")
        raise SystemExit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  Pattern encoding
#
#  Each guess produces a colour pattern: 5 tiles, each Green/Yellow/Gray.
#  We encode this as a base-3 integer:  Green=2, Yellow=1, Gray=0
#  e.g. (2,1,0,2,1) → 2·3⁴ + 1·3³ + 0·3² + 2·3¹ + 1·3⁰ = 169 + 27 + 6 + 1 = 203
#  Range: 0 – 242.  Fits in one byte (uint8).
#
#  ALL_GREEN = 2+2*3+2*9+2*27+2*81 = 242
# ══════════════════════════════════════════════════════════════════════════════

_B3 = (1, 3, 9, 27, 81)   # powers of 3 for positions 0-4
ALL_GREEN = 242             # 2*(1+3+9+27+81)

def compute_pattern_int(guess: str, answer: str) -> int:
    """
    Two-pass Wordle pattern logic → encodes result as a single int 0-242.
    """
    p0 = p1 = p2 = p3 = p4 = 0
    a0, a1, a2, a3, a4 = answer[0], answer[1], answer[2], answer[3], answer[4]
    pool = [a0, a1, a2, a3, a4]

    # Pass 1 – greens
    if guess[0] == a0: p0 = 2; pool[0] = None
    if guess[1] == a1: p1 = 2; pool[1] = None
    if guess[2] == a2: p2 = 2; pool[2] = None
    if guess[3] == a3: p3 = 2; pool[3] = None
    if guess[4] == a4: p4 = 2; pool[4] = None

    # Pass 2 – yellows
    if p0 != 2 and guess[0] in pool: pool[pool.index(guess[0])] = None; p0 = 1
    if p1 != 2 and guess[1] in pool: pool[pool.index(guess[1])] = None; p1 = 1
    if p2 != 2 and guess[2] in pool: pool[pool.index(guess[2])] = None; p2 = 1
    if p3 != 2 and guess[3] in pool: pool[pool.index(guess[3])] = None; p3 = 1
    if p4 != 2 and guess[4] in pool: pool[pool.index(guess[4])] = None; p4 = 1

    return p0 + p1*3 + p2*9 + p3*27 + p4*81

def decode_pattern_int(code: int) -> tuple:
    """Decode a pattern int back to a tuple of 5 values (0/1/2)."""
    return tuple((code // _B3[i]) % 3 for i in range(5))

# ══════════════════════════════════════════════════════════════════════════════
#  Precomputation
#
#  pattern_matrix[vi][ai] = encoded pattern int when vocabulary[vi] is guessed
#                            and answers[ai] is the answer.
#
#  Stored as a list of bytearrays: 1 byte per cell, very compact.
#  Memory: len(vocab) × len(answers) bytes  ≈ 12 k × 2.3 k ≈ 27 MB
# ══════════════════════════════════════════════════════════════════════════════

def precompute_patterns(vocabulary: list, answers: list) -> list:
    """
    Build the full pattern matrix.
    Returns pattern_matrix: list[bytearray]  (indexed by vocab index, then answer index)
    Shows a live progress bar.
    """
    n_vocab   = len(vocabulary)
    n_answers = len(answers)
    total     = n_vocab * n_answers

    print(f"\n  {BOLD}Precomputing pattern matrix…{RESET}")
    print(f"  {DIM}{n_vocab:,} guesses  ×  {n_answers:,} answers  =  {total:,} patterns{RESET}\n")

    BAR_W    = 35
    start    = time.time()
    matrix   = []

    for vi, guess in enumerate(vocabulary):
        row = bytearray(n_answers)
        for ai, answer in enumerate(answers):
            row[ai] = compute_pattern_int(guess, answer)
        matrix.append(row)

        # ── progress bar ──────────────────────────────────────────────────────
        if vi % 100 == 0 or vi == n_vocab - 1:
            done    = vi + 1
            frac    = done / n_vocab
            filled  = round(frac * BAR_W)
            bar     = BAR_FILL * filled + BAR_EMPTY * (BAR_W - filled)
            elapsed = time.time() - start
            eta     = (elapsed / frac) * (1 - frac) if frac > 0 else 0
            print(f"  {CYAN}{bar}{RESET}  {frac*100:5.1f}%   ETA {eta:4.0f}s", end="\r")

    elapsed = time.time() - start
    print(f"  {CYAN}{BAR_FILL * BAR_W}{RESET}  100.0%   done in {elapsed:.1f}s        ")
    print()
    return matrix

# ══════════════════════════════════════════════════════════════════════════════
#  Entropy (fast — uses precomputed matrix)
# ══════════════════════════════════════════════════════════════════════════════

# Precompute log2 table for counts 1..N to avoid repeated log calls
_LOG2_CACHE: dict = {}

def _log2(n: int) -> float:
    if n not in _LOG2_CACHE:
        _LOG2_CACHE[n] = math.log2(n)
    return _LOG2_CACHE[n]

def expected_entropy_fast(vocab_idx: int,
                          remaining_indices: list,
                          pattern_matrix: list) -> float:
    """
    H = −Σ p·log₂p  over all patterns produced by vocabulary[vocab_idx]
    against the current remaining answer pool.

    Instead of calling get_pattern() at all, we just look up
    pattern_matrix[vocab_idx][answer_idx].
    """
    row    = pattern_matrix[vocab_idx]
    counts: dict = defaultdict(int)
    for ai in remaining_indices:
        counts[row[ai]] += 1

    total   = len(remaining_indices)
    log2tot = _log2(total)
    entropy = 0.0
    for count in counts.values():
        # p·log₂p  =  (count/total)·(log₂count − log₂total)
        entropy -= (count / total) * (_log2(count) - log2tot)
    return entropy

def top_suggestions_fast(vocabulary: list,
                         vocab_indices: list,
                         remaining_indices: list,
                         remaining_set: set,
                         pattern_matrix: list,
                         n: int = TOP_N) -> list:
    """
    Score every word in vocab_indices and return top-n by expected entropy.
    Valid answers get a tiny tiebreak bonus (same logic as before).
    """
    scored = []
    for vi in vocab_indices:
        h     = expected_entropy_fast(vi, remaining_indices, pattern_matrix)
        bonus = 1e-6 if vi in remaining_set else 0.0
        scored.append((vocabulary[vi], h + bonus))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]

# ══════════════════════════════════════════════════════════════════════════════
#  Filtering (uses precomputed matrix)
# ══════════════════════════════════════════════════════════════════════════════

def filter_remaining(remaining_indices: list,
                     vocab_idx: int,
                     observed_pattern: int,
                     pattern_matrix: list) -> list:
    """Keep only answer indices whose precomputed pattern matches observed."""
    row = pattern_matrix[vocab_idx]
    return [ai for ai in remaining_indices if row[ai] == observed_pattern]

# ══════════════════════════════════════════════════════════════════════════════
#  Rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_tiles(guess: str, pattern_code: int) -> str:
    pattern = decode_pattern_int(pattern_code)
    colors  = [GREEN if p == 2 else YELLOW if p == 1 else GRAY for p in pattern]
    return "".join(f"{c} {l.upper()} {RESET}" for l, c in zip(guess, colors))

def render_empty_row() -> str:
    return f"{DIM} ·   ·   ·   ·   · {RESET}"

def render_keyboard(used: dict) -> str:
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    lines = []
    for row in rows:
        line = "  "
        for ch in row:
            color = used.get(ch, DIM)
            line += f"{color} {ch.upper()} {RESET}"
        lines.append(line)
    return "\n".join(lines)

def render_entropy_bar(h: float, max_h: float, width: int = 12) -> str:
    filled = round((h / max_h) * width) if max_h > 0 else 0
    bar    = BAR_FILL * filled + BAR_EMPTY * (width - filled)
    return f"{CYAN}{bar}{RESET}"

def render_suggestions(suggestions: list, remaining_words: set) -> str:
    if not suggestions:
        return ""
    max_h  = suggestions[0][1] if suggestions else 1.0
    lines  = [f"  {BOLD}Top {len(suggestions)} suggestions  (expected bits gained):{RESET}"]
    for rank, (word, h) in enumerate(suggestions, 1):
        bar   = render_entropy_bar(h, max_h)
        valid = f"{GREEN} ✓ answer {RESET}" if word in remaining_words else f"{DIM}  guess   {RESET}"
        lines.append(f"  {rank}. {BOLD}{word.upper()}{RESET}  {bar}  {h:.3f} bits  {valid}")
    lines.append(
        f"\n  {DIM}✓ answer = still a valid solution  |  "
        f"{len(remaining_words):,} word{'s' if len(remaining_words) != 1 else ''} remaining{RESET}"
    )
    return "\n".join(lines)

def header():
    print(f"\n{BOLD}{'─'*50}")
    print("       W O R D L E   +   E N T R O P Y")
    print(f"{'─'*50}{RESET}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  Main game loop
# ══════════════════════════════════════════════════════════════════════════════

COLOR_RANK = {GRAY: 0, YELLOW: 1, GREEN: 2}

def play(answers: list, vocabulary: list,
         answer_index: dict, vocab_index: dict,
         pattern_matrix: list):

    answer         = random.choice(answers)
    answer_ai      = answer_index[answer]          # int index into answers[]
    remaining_idx  = list(range(len(answers)))     # shrinks each turn
    remaining_set  = set(range(len(answers)))      # for O(1) membership
    vocab_indices  = list(range(len(vocabulary)))  # all vocab indices
    history        = []                            # (guess_str, pattern_code)
    keyboard       = {}                            # letter → best colour

    header()

    for attempt in range(1, MAX_GUESSES + 1):
        print(f"  Answer pool : {len(answers):,} words   "
            f"Starting entropy: {math.log2(len(answers)):.2f} bits")
        print(f"  Vocabulary  : {len(vocabulary):,} words\n")

        print(f"  {DIM}[DEBUG]  Answer → {RESET}{BOLD}{answer.upper()}{RESET}\n")

        # ── Board ─────────────────────────────────────────────────────────────
        print(f"  {'─'*29}")
        for g, code in history:
            print("   ", render_tiles(g, code))
        for _ in range(MAX_GUESSES - len(history)):
            print("   ", render_empty_row())
        print(f"  {'─'*29}\n")

        # ── Keyboard ──────────────────────────────────────────────────────────
        print(render_keyboard(keyboard))
        print()

        # ── Entropy suggestions (instant — precomputed) ───────────────────────
        remaining_words = {answers[i] for i in remaining_idx}
        suggestions = top_suggestions_fast(
            vocabulary, vocab_indices, remaining_idx,
            remaining_set, pattern_matrix, TOP_N
        )
        print(render_suggestions(suggestions, remaining_words))
        print()

        # ── Input ─────────────────────────────────────────────────────────────
        while True:
            raw = input(f"  Guess {attempt}/{MAX_GUESSES}: ").strip().lower()
            if len(raw) == WORD_LEN and raw.isalpha():
                break
            print(f"  {YELLOW} ! {RESET}  Please enter a {WORD_LEN}-letter word.\n")

        # ── Compute / look up pattern ─────────────────────────────────────────
        if raw in vocab_index:
            # Fast path: guess is in vocabulary → use precomputed matrix
            vi           = vocab_index[raw]
            pattern_code = pattern_matrix[vi][answer_ai]
        else:
            # Fallback: compute on the fly (word not in vocabulary)
            pattern_code = compute_pattern_int(raw, answer)
            vi           = None

        history.append((raw, pattern_code))

        # ── Narrow remaining pool ─────────────────────────────────────────────
        old_size = len(remaining_idx)
        if vi is not None:
            remaining_idx = filter_remaining(remaining_idx, vi, pattern_code, pattern_matrix)
        else:
            # Fallback filter using live computation
            remaining_idx = [ai for ai in remaining_idx
                             if compute_pattern_int(raw, answers[ai]) == pattern_code]
        remaining_set = set(remaining_idx)

        new_size    = len(remaining_idx)
        bits_gained = (math.log2(old_size / new_size)
                       if new_size > 0 and new_size < old_size else 0.0)

        # ── Update keyboard colours ───────────────────────────────────────────
        pattern_vals = decode_pattern_int(pattern_code)
        color_map    = [GREEN if p == 2 else YELLOW if p == 1 else GRAY for p in pattern_vals]
        for letter, color in zip(raw, color_map):
            prev = keyboard.get(letter, GRAY)
            if COLOR_RANK.get(color, 0) > COLOR_RANK.get(prev, 0):
                keyboard[letter] = color

        print(f"\n  {CYAN}+{bits_gained:.2f} bits{RESET}  →  "
              f"{new_size:,} word{'s' if new_size != 1 else ''} remaining\n")

        # ── Win ───────────────────────────────────────────────────────────────
        if pattern_code == ALL_GREEN:
            print(f"  {'─'*29}")
            for g, code in history:
                print("   ", render_tiles(g, code))
            print(f"  {'─'*29}\n")
            msgs = ["Genius!", "Magnificent!", "Impressive!", "Splendid!", "Great!", "Phew!"]
            print(f"  {GREEN} {msgs[attempt-1]} {RESET}  "
                  f"Solved in {attempt} guess{'es' if attempt > 1 else ''}!\n")
            return

    # ── Out of guesses ────────────────────────────────────────────────────────
    print(f"  {'─'*29}")
    for g, code in history:
        print("   ", render_tiles(g, code))
    print(f"  {'─'*29}\n")
    print(f"  {GRAY} Out of guesses! {RESET}  The word was {BOLD}{answer.upper()}{RESET}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    answers    = load_words(ANSWERS_PATH)
    vocabulary = load_words(GUESSES_PATH)

    # Ensure every answer is also guessable
    vocab_set = set(vocabulary)
    missing   = [w for w in answers if w not in vocab_set]
    if missing:
        print(f"  {YELLOW} ! {RESET}  {len(missing)} answer word(s) not in guesses list — adding automatically.")
        vocabulary = vocabulary + missing

    # Build index maps for O(1) lookup
    answer_index = {w: i for i, w in enumerate(answers)}    # word → answer index
    vocab_index  = {w: i for i, w in enumerate(vocabulary)} # word → vocab index

    # ── One-time precomputation ───────────────────────────────────────────────
    pattern_matrix = precompute_patterns(vocabulary, answers)

    while True:
        play(answers, vocabulary, answer_index, vocab_index, pattern_matrix)
        again = input("  Play again? (y/n): ").strip().lower()
        if again != "y":
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            break
        print()