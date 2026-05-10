#!/bin/bash
set -e

echo "Running full pipeline verification test..."

export TRAIN_EXTRA_ARGS="--iters 10"

for stage in 1 2 3 4 5; do
    echo "========================================="
    echo "TEST RUN: Stage $stage"
    echo "========================================="
    
    # Run the stage script. Stage 1 is fresh, 2-5 resume from previous stage's adapter.
    # Stage 4 and 5 require --allow-implicit-resume or --resume-from now per our new checks.
    if [ "$stage" -ge 4 ]; then
        export ALLOW_IMPLICIT_RESUME=1
    fi
    
    if [ "$stage" -eq 1 ]; then
        ./tools/run_stage.sh "$stage"
    else
        ./tools/run_stage.sh "$stage" --resume
    fi
    
    echo "Stage $stage test run passed."
done

echo "========================================="
echo "Pipeline validation complete. All 5 stages successfully ran with new telemetry."
echo "Cleaning up test runs..."
echo "========================================="

rm -rf runs/curriculum_stage1
rm -rf runs/curriculum_stage2
rm -rf runs/curriculum_stage3
rm -rf runs/curriculum_stage4
rm -rf runs/curriculum_stage5
rm -f runs/tikz_stage1_adapter.safetensors
rm -f runs/tikz_stage2_adapter.safetensors
rm -f runs/tikz_stage3_adapter.safetensors
rm -f runs/tikz_stage4_adapter.safetensors
rm -f runs/tikz_stage5_adapter.safetensors

echo "Cleanup finished!"
