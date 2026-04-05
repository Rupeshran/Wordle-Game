"""
precompute.py  —  run once, produces precomputed.pkl
─────────────────────────────────────────────────────
Saves:
  pattern_matrix   vocab * answers bytearray  (base-3, 0-242)
  vocabulary       list[str]
  answers          list[str]
  vocab_index      word → int
  answer_index     word → int
  freq_raw         word → float  (raw corpus frequency)
  freq_sigmoid     word → float  (sigmoid-normalised, clamped [0,1])
  f_table          bucket_idx → avg guesses still needed AFTER current guess
  bucket_size      float (bits per bucket)

f_table semantics match wordle.py's formula exactly:
  E[score] = attempt + (1-p) * f(H_rem - H)
  f(residual) = expected additional guesses needed AFTER this guess
              = avg(guesses_after_this_step) bucketed by residual entropy
"""

import pickle, math, random, time
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════

ANSWERS_PATH = "data/possible_words.txt"
GUESSES_PATH = "data/allowed_words.txt"
OUTPUT_PATH  = "precomputed.pkl"

WORD_LEN    = 5
MAX_GUESSES = 6
ALL_GREEN   = 242
N_SIMS      = 2315
BUCKET_SIZE = 0.5

# ══════════════════════════════════════════════════════════════════════════════
#  ANSI
# ══════════════════════════════════════════════════════════════════════════════

GREEN  = "\033[42m\033[30m"
YELLOW = "\033[43m\033[30m"
CYAN   = "\033[46m\033[30m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"
FILL, EMPTY = "█", "░"

def bar(frac: float, w: int = 38) -> str:
    f = max(0, min(w, round(frac * w)))
    return f"{CYAN}{FILL*f}{EMPTY*(w-f)}{RESET}"

def section(title: str):
    pad = max(0, 48 - len(title))
    print(f"\n{BOLD}  ┌─ {title} {'─'*pad}┐{RESET}")

def done_line(t: float):
    print(f"  {GREEN} ✓ {RESET} done in {t:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
#  WORD LOADING
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
        print(f"\n  {YELLOW} ! {RESET}  '{path}' not found.\n")
        raise SystemExit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  PATTERN ENCODING  (base-3: 2=green 1=yellow 0=gray, range 0–242)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pattern_int(guess: str, answer: str) -> int:
    p = [0] * WORD_LEN; pool = list(answer)
    for i in range(WORD_LEN):
        if guess[i] == answer[i]: p[i] = 2; pool[i] = None
    for i in range(WORD_LEN):
        if p[i] == 2: continue
        if guess[i] in pool: p[i] = 1; pool[pool.index(guess[i])] = None
    return p[0] + p[1]*3 + p[2]*9 + p[3]*27 + p[4]*81

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PATTERN MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_pattern_matrix(vocabulary: list, answers: list) -> list:
    section("1 / 3   Pattern matrix")
    nv, na = len(vocabulary), len(answers)
    print(f"  {DIM}{nv:,} vocab  ×  {na:,} answers  =  {nv*na:,} cells{RESET}\n")
    matrix = []; t0 = time.time()
    for vi, guess in enumerate(vocabulary):
        row = bytearray(na)
        for ai, answer in enumerate(answers): 
            row[ai] = compute_pattern_int(guess, answer)
        matrix.append(row)
        if vi % 50 == 0 or vi == nv - 1:
            frac = (vi + 1) / nv
            eta  = ((time.time() - t0) / frac) * (1 - frac)
            print(f"  {bar(frac)}  {frac*100:5.1f}%   ETA {eta:4.0f}s", end="\r")
    print(f"  {bar(1.0)}  100.0%              ")
    done_line(time.time() - t0)
    return matrix

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — WORD FREQUENCIES + SIGMOID
# ══════════════════════════════════════════════════════════════════════════════

def build_frequencies(vocabulary: list) -> tuple:
    section("2 / 3   Word frequencies + sigmoid")
    try:
        from wordfreq import word_frequency
    except ImportError:
        print("  pip install wordfreq\n"); raise SystemExit(1)

    n = len(vocabulary); t0 = time.time()
    print(f"  {DIM}Fetching {n:,} frequencies…{RESET}\n")

    freq_raw = {}
    for i, word in enumerate(vocabulary):
        freq_raw[word] = word_frequency(word, 'en')
        if i % 200 == 0 or i == n - 1:
            print(f"  {bar((i+1)/n)}  {(i+1)/n*100:5.1f}%", end="\r")
    print(f"  {bar(1.0)}  100.0%")

    log_vals = {w: math.log10(f + 1e-10) for w, f in freq_raw.items()}
    vals     = list(log_vals.values())
    mean     = sum(vals) / len(vals)
    std      = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals)) + 1e-9
    normed   = {w: (v - mean) / std for w, v in log_vals.items()}

    def sigmoid(x: float) -> float:
        try:    s = 1.0 / (1.0 + math.exp(-x))
        except: s = 0.0 if x < 0 else 1.0
        return max(0.0, min(1.0, s))

    freq_sigmoid = {w: sigmoid(x) for w, x in normed.items()}
    sv = sorted(freq_sigmoid.values())
    print(f"\n  {BOLD}Sigmoid spread:{RESET}  "
          f"p10={sv[int(.10*len(sv))]:.3f}  "
          f"p50={sv[int(.50*len(sv))]:.3f}  "
          f"p90={sv[int(.90*len(sv))]:.3f}")
    done_line(time.time() - t0)
    return freq_raw, freq_sigmoid

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DISTRIBUTION HELPERS  (used in simulation and wordle.py)
# ══════════════════════════════════════════════════════════════════════════════

def make_weights(remaining_idx: list, freq_sigmoid: dict, answers: list) -> dict:
    """p(answer=w) ∝ freq_sigmoid[w], renormalised. This IS the Bayesian update."""
    raw   = {ai: freq_sigmoid.get(answers[ai], 0.5) for ai in remaining_idx}
    total = sum(raw.values()) or 1.0
    return {ai: w / total for ai, w in raw.items()}

def weighted_entropy(vi: int, weights: dict, matrix: list) -> float:
    """H(guess) = −Σ p(pattern)·log₂p(pattern) with frequency-weighted p."""
    row = matrix[vi]; pattern_probs: dict = defaultdict(float)
    for ai, p in weights.items():
        pattern_probs[row[ai]] += p
    return max(0.0, -sum(p * math.log2(p) for p in pattern_probs.values() if p > 0))

def h_of_weights(weights: dict) -> float:
    return max(0.0, -sum(p * math.log2(p) for p in weights.values() if p > 0))

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — f-TABLE via simulation
#
#  Policy: greedy weighted-entropy (same as wordle.py, minus the f-table term
#  since we're building it — we use H directly as a proxy for f during sim).
#
#  What we record at each non-winning step:
#    residual = H_rem_before_guess − H_of_guess   (entropy still left after)
#    guesses_after = steps still needed after this one to reach the solution
#
#  Then f_table[bucket] = mean(guesses_after) for residuals in that bucket.
#  This aligns exactly with:  E[score] = attempt + (1−p) × f(H_rem − H)
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_one(answer_ai: int, answers: list,
                  matrix: list, freq_sigmoid: dict) -> list:
    """
    Returns list of (residual_entropy, guesses_after_this_step).
    Only records non-winning steps — winning steps contribute 0 future guesses
    and are already accounted for by the p*attempt term in E[score].
    """
    remaining = list(range(len(answers)))
    step_data = []   # (H_rem_before, H_of_guess) for each non-winning step

    for _ in range(MAX_GUESSES):
        if not remaining:
            break
        weights  = make_weights(remaining, freq_sigmoid, answers)
        H_rem    = h_of_weights(weights)

        # Greedy policy: maximise weighted entropy (proxy — f not available yet)
        best_vi  = max(remaining,
                       key=lambda ai: weighted_entropy(ai, weights, matrix))
        H_guess  = weighted_entropy(best_vi, weights, matrix)
        pattern  = matrix[best_vi][answer_ai]

        if pattern == ALL_GREEN:
            break   # winning step — not recorded

        step_data.append((H_rem, H_guess))
        remaining = [ai for ai in remaining if matrix[best_vi][ai] == pattern]

    # guesses_after step i = total non-winning steps − i
    # (one more winning guess follows all non-winning steps)
    k       = len(step_data)
    records = []
    for i, (H_rem, H_guess) in enumerate(step_data):
        residual     = max(0.0, H_rem - H_guess)
        guesses_after = k - i        # steps still needed after step i+1
        records.append((residual, guesses_after))
    return records

def build_f_table(answers: list, matrix: list, freq_sigmoid: dict) -> dict:
    section("3 / 3   f-table  (residual entropy → guesses after)")
    print(f"  {DIM}Running {N_SIMS} games — recording residual entropy per step…{RESET}\n")

    all_records: list = []; t0 = time.time(); random.seed(42)
    for sim in range(N_SIMS):
        ai = random.randrange(len(answers))
        all_records.extend(_simulate_one(ai, answers, matrix, freq_sigmoid))
        frac = (sim + 1) / N_SIMS
        print(f"  {bar(frac)}  {frac*100:5.1f}%   game {sim+1}/{N_SIMS}", end="\r")
    print(f"  {bar(1.0)}  100.0%              ")

    sums:   dict = defaultdict(float)
    counts: dict = defaultdict(int)
    for residual, g_after in all_records:
        b = int(residual / BUCKET_SIZE)
        sums[b]   += g_after
        counts[b] += 1

    f_table = {b: sums[b] / counts[b] for b in sums}

    print(f"\n  {BOLD}f-table  (residual H bucket → avg guesses after this step):{RESET}")
    for b in sorted(f_table):
        lo, hi = b * BUCKET_SIZE, (b+1) * BUCKET_SIZE
        avg    = f_table[b]
        print(f"  {lo:.1f}–{hi:.1f} bits  {bar(min(avg/MAX_GUESSES,1.0), w=20)}"
              f"  {avg:.2f} guesses  {DIM}n={counts[b]}{RESET}")
    done_line(time.time() - t0)
    return f_table

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save(path: str, payload: dict):
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6
    print(f"  {GREEN} ✓ {RESET} Saved  {BOLD}{path}{RESET}  ({mb:.1f} MB)\n")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*54}")
    print("   P R E C O M P U T E   ·   W O R D L E")
    print(f"{'═'*54}{RESET}")

    answers    = load_words(ANSWERS_PATH)
    vocabulary = load_words(GUESSES_PATH)

    missing = [w for w in answers if w not in set(vocabulary)]
    if missing:
        print(f"\n  {YELLOW} ! {RESET} {len(missing)} answer word(s) not in guesses — adding.")
        vocabulary = vocabulary + missing

    vocab_index  = {w: i for i, w in enumerate(vocabulary)}
    answer_index = {w: i for i, w in enumerate(answers)}

    print(f"\n  Answers    : {len(answers):,} words")
    print(f"  Vocabulary : {len(vocabulary):,} words")

    t0                     = time.time()
    pattern_matrix         = build_pattern_matrix(vocabulary, answers)
    freq_raw, freq_sigmoid = build_frequencies(vocabulary)
    f_table                = build_f_table(answers, pattern_matrix, freq_sigmoid)

    print(f"\n{BOLD}  ┌─ Saving {'─'*42}┐{RESET}")
    save(OUTPUT_PATH, {
        "vocabulary"    : vocabulary,
        "answers"       : answers,
        "vocab_index"   : vocab_index,
        "answer_index"  : answer_index,
        "pattern_matrix": pattern_matrix,
        "freq_raw"      : freq_raw,
        "freq_sigmoid"  : freq_sigmoid,
        "f_table"       : f_table,
        "bucket_size"   : BUCKET_SIZE,
    })
    print(f"  {BOLD}Total time: {time.time()-t0:.1f}s{RESET}\n")
    print(f"  Run  {BOLD}python wordle.py{RESET}  now.\n")