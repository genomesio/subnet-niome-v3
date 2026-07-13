import json
import math

from niome_subnet.genomics.model import (
  Stage1Result,
  Stage2Result,
)
from Bio import SeqIO

from niome_subnet.utils.settings import (
  CONTRACT_PATH,
  CHR11_PATH,
  INVALID_EXPERIMENTS_PATH,
  MINER_SUBMISSION_PATH,
  HBB_REFERENCE_PATH,
  VALID_EXPERIMENTS_PATH,
)


# =========================================================
# LOAD GENOME
# =========================================================
def load_chr11(path):
    for record in SeqIO.parse(path, "fasta"):
        return str(record.seq)
    raise ValueError("chr11 not found")


# =========================================================
# BIOLOGY HELPERS
# =========================================================
def gc_content(seq):
    return sum(b in "GC" for b in seq) / max(1, len(seq))


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


# =========================================================
# PAM CHECK
# =========================================================
def check_pam(seq, start, L, cas):

    if start < 0 or start + L >= len(seq):
        return False, "out_of_bounds"

    if cas == "Cas9":
        pam = seq[start + L:start + L + 3]
        return (len(pam) == 3 and pam[1:] == "GG"), "ok"

    if cas == "Cas12a":
        if start < 4:
            return False, "out_of_bounds"
        pam = seq[start - 4:start]
        return (len(pam) == 4 and pam[:3] == "TTT"), "ok"

    return False, "invalid_cas"


# =========================================================
# STAGE 1 — STRICT GATES ONLY
# =========================================================
def stage1(exp, seq, mutation_map, contract):

    guide = exp["guideRNA"]
    cas = exp["cas_system"]
    start = exp["target_alignment_start"]
    mutation = exp["mutation"]

    if mutation not in contract["active_mutations"]:
        return 0.0, "mutation_not_allowed"

    if len(guide) not in (20, 23):
        return 0.0, "invalid_length"

    if start < 0 or start + len(guide) >= len(seq):
        return 0.0, "out_of_bounds"

    pam_ok, pam_status = check_pam(seq, start, len(guide), cas)
    if not pam_ok:
        return 0.0, f"pam_{pam_status}"

    mm = hamming(guide, seq[start:start + len(guide)])

    if mm > contract["rules"]["max_mismatches"]:
        return 0.0, "too_many_mismatches"

    return 1.0, "ok"


# =========================================================
# STAGE 2 — STRUCTURAL SCORE ONLY
# =========================================================
def stage2(exp, seq, mutation_map):

    guide = exp["guideRNA"]
    start = exp["target_alignment_start"]
    mut = mutation_map[exp["mutation"]]

    gc = gc_content(guide)
    dist = abs(start - mut)

    gc_score = max(0.0, 1.0 - abs(gc - 0.5) * 2)
    dist_score = math.exp(-dist / 1200)
    consistency = 1.0 if dist < 2000 else 0.3

    score = 0.5 * gc_score + 0.3 * dist_score + 0.2 * consistency

    return score, {
        "gc": gc,
        "distance": dist,
        "gc_score": gc_score,
        "dist_score": dist_score,
        "consistency": consistency
    }


def run_stage12() -> tuple[Stage1Result, Stage2Result]:
    reference = json.load(open(HBB_REFERENCE_PATH))
    contract = json.load(open(CONTRACT_PATH))
    submission = json.load(open(MINER_SUBMISSION_PATH))

    seq = load_chr11(CHR11_PATH)
    mutation_map = reference["mutation_map"]

    valid_experiments = []
    invalid_experiments = []

    stage2_logs = []

    stage1_scores = []
    stage2_scores = []

    for exp in submission:
        s1, reason = stage1(exp, seq, mutation_map, contract)
        stage1_scores.append(s1)

        if s1 == 0.0:
            invalid_experiments.append({
                "experiment": exp,
                "stage1_pass": False,
                "reason": reason
            })
            continue

        s2, s2_info = stage2(exp, seq, mutation_map)

        stage2_scores.append(s2)
        stage2_logs.append(s2_info)

        valid_experiments.append({
            "experiment": exp,

            # CLEAN CONTRACT FOR STAGE 3 / 4
            "features": {
                "gc": s2_info["gc"],
                "distance_to_mutation": s2_info["distance"],
                "gc_score": s2_info["gc_score"],
                "dist_score": s2_info["dist_score"],
                "consistency": s2_info["consistency"]
            },

            "stage1": {
                "valid": True
            },

            "stage2": {
                "structural_score": s2
            }
        })

    # =====================================================
    # EXPORT (CRITICAL FOR STAGE 3 + 4)
    # =====================================================
    with open(VALID_EXPERIMENTS_PATH, "w") as f:
        json.dump(valid_experiments, f, indent=2)

    with open(INVALID_EXPERIMENTS_PATH, "w") as f:
        json.dump(invalid_experiments, f, indent=2)

    stage1 = Stage1Result(
        min=min(stage1_scores),
        max=max(stage1_scores),
        mean=sum(stage1_scores) / len(stage1_scores),
        valid=sum(1 for s in stage1_scores if s > 0.0) / len(stage1_scores)
    )
    
    if stage1.valid < 1:
        return stage1, None
    
    return stage1, Stage2Result(
        min=min(stage2_scores) if stage2_scores else 0.0,
        max=max(stage2_scores) if stage2_scores else 0.0,
        mean=sum(stage2_scores) / len(stage2_scores) if stage2_scores else 0.0
    )
