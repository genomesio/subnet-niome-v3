import json
import math
import random

from collections import defaultdict, Counter
from niome_subnet.genomics.model import Stage3Result

from niome_subnet.utils.settings import STAGE3_DATASET, VALID_EXPERIMENTS_PATH


# =========================================================
# LOAD STAGE 12 OUTPUT
# =========================================================
def load_experiments(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    raise ValueError("Expected valid_experiments.json (list format)")


# =========================================================
# FEATURE ENGINE (from Stage 2 output)
# =========================================================
def extract_features(exp):

    feat = exp["features"]

    return {
        "gc": feat["gc"],
        "distance": feat["distance_to_mutation"],
        "gc_score": feat["gc_score"],
        "dist_score": feat["dist_score"],
        "consistency": feat["consistency"]
    }


# =========================================================
# BIOPHYSICAL ENERGY MODEL
# =========================================================
def sequence_energy(features):
    """
    proxy stability function (deterministic)
    """
    gc = features["gc"]
    dist = features["distance"]

    return max(0.0, min(1.0,
        1.8 * gc +
        0.6 * math.exp(-dist / 1500)
    ))


# =========================================================
# MICROHOMOLOGY MODEL (latent stochastic event)
# =========================================================
def microhomology_trigger(features):
    gc = features["gc"]

    repeat_bias = gc * (1 - gc)
    p_mh = min(0.6, repeat_bias * 2.2)

    return random.random() < p_mh


# =========================================================
# CUT PROBABILITY MODEL
# =========================================================
def cut_probability(cas, energy):

    base = 0.86 if cas == "Cas9" else 0.78

    return min(0.99, max(0.4, base + 0.18 * energy))


# =========================================================
# REPAIR MODE MODEL
# =========================================================
def repair_mode(cas, energy, mh):

    if cas == "Cas9":
        hdr_base = 0.32
    else:
        hdr_base = 0.24

    hdr = hdr_base + 0.35 * energy
    mh_nhej = 0.30 if mh else 0.12
    blunt = 0.35

    total = hdr + mh_nhej + blunt

    r = random.random() * total

    if r < hdr:
        return "HDR"
    elif r < hdr + mh_nhej:
        return "MH_NHEJ"
    else:
        return "BLUNT_NHEJ"


# =========================================================
# INDEL MODEL (research-grade mixture)
# =========================================================
def sample_indel_length(mode):

    if mode == "HDR":
        return 0

    if mode == "MH_NHEJ":
        return max(1, int(random.gammavariate(2.2, 2.8)))

    if mode == "BLUNT_NHEJ":
        return max(1, int(random.expovariate(0.6)))

    return 1


# =========================================================
# SIMULATION CORE
# =========================================================
def simulate(exp):

    features = extract_features(exp)
    experiment_id = exp["experiment"]["experiment_id"]

    cas = exp["experiment"]["cas_system"]
    mutation = exp["experiment"]["mutation"]

    energy = sequence_energy(features)
    mh = microhomology_trigger(features)

    cut_p = cut_probability(cas, energy)

    if random.random() > cut_p:
        return {
            "experiment_id": experiment_id,
            "mutation": mutation,
            "cas": cas,
            "outcome": "no_cut",
            "indel_length": 0,
            "features": features,
            "energy": energy,
            "mh": mh
        }

    mode = repair_mode(cas, energy, mh)
    indel = sample_indel_length(mode)

    return {
        "experiment_id": experiment_id,
        "mutation": mutation,
        "cas": cas,
        "outcome": mode,
        "indel_length": indel,
        "features": features,
        "energy": energy,
        "mh": mh
    }


def run_stage3() -> Stage3Result:
    experiments = load_experiments(VALID_EXPERIMENTS_PATH)

    results = []
    mutation_stats = defaultdict(Counter)

    for exp in experiments:
        r = simulate(exp)
        results.append(r)
        mutation_stats[r["mutation"]][r["outcome"]] += 1

    # =====================================================
    # SUMMARY
    # =====================================================
    total = len(results)

    outcome_counts = Counter([r["outcome"] for r in results])
    indels = [r["indel_length"] for r in results]
    energies = [r["energy"] for r in results]

    result = Stage3Result(
        n=total,
        cut_rate=1 - outcome_counts["no_cut"] / total,
        mean_indel_length=sum(indels) / max(1, len(indels)),
        mean_energy=sum(energies) / max(1, len(energies)),
        outcomes=dict(outcome_counts)
    )

    # =====================================================
    # EXPORT (IMPORTANT FOR STAGE 4)
    # =====================================================
    with open(STAGE3_DATASET, "w") as f:
        json.dump(results, f, indent=2)

    return result
