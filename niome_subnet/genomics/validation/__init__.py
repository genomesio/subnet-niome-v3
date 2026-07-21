from niome_subnet.genomics.model import MinerScore
from niome_subnet.genomics.validation.stage12 import run_stage12
from niome_subnet.genomics.validation.stage3 import run_stage3
from niome_subnet.genomics.validation.stage4 import run_stage4

def benchmark_submission(uid: int) -> MinerScore:
    final_score = 0
    
    stage1_result, stage2_result = run_stage12()
    final_score += stage1_result.mean * (stage1_result.mean * stage1_result.valid) * 0.2

    if stage2_result and stage2_result.mean > 0:
        final_score += stage2_result.mean * 0.2
        stage3_result = run_stage3()
    else:
        stage3_result = None

    if stage3_result and stage3_result.mean_energy > 0:
        final_score += stage3_result.mean_energy * 0.2
        stage4_result = run_stage4()
    else:
        stage4_result = None

    if stage4_result and stage4_result.consistency_score > 0:
        final_score += stage4_result.consistency_score * 0.4
    else:
        stage4_result = None

    return MinerScore(
        uid=uid,
        stage1=stage1_result,
        stage2=stage2_result,
        stage3=stage3_result,
        stage4=stage4_result,
        final_score=final_score,
        log=f"Stage 1: {stage1_result}\nStage 2: {stage2_result}\nStage 3: {stage3_result}\nStage 4: {stage4_result}"
    )
