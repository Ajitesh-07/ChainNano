# run_full_pipeline.ps1

# Write-Host "=================================================" -ForegroundColor Cyan
# Write-Host "  STARTING PHASE 1: BOOTSTRAP (High Speed)" -ForegroundColor Cyan
# Write-Host "=================================================" -ForegroundColor Cyan

# # Phase 1: Fast cycles, high LR, small memory buffer
# python pipeline.py --iterations 50 --games 500 --timeout 500 --threads 24 --epochs 3 --batch-size 1024 --accum-steps 2 --lr 1e-3 --max-buffer 2500 --max-games 2500

# # Safety check: If Phase 1 crashed, stop the script so we don't ruin Phase 2
# if ($LASTEXITCODE -ne 0) {
#     Write-Host "`n[FATAL] Phase 1 crashed. Aborting Phase 2." -ForegroundColor Red
#     exit $LASTEXITCODE
# }

# Write-Host "`n=================================================" -ForegroundColor Green
# Write-Host "  PHASE 1 COMPLETE. GRADUATING TO PHASE 2" -ForegroundColor Green
# Write-Host "  STARTING PHASE 2: DEEPENING (Tactical Focus)" -ForegroundColor Green
# Write-Host "=================================================" -ForegroundColor Green

# Phase 2: More games, lower LR, massive memory buffer
python pipeline.py --iterations 1000 --games 1000 --timeout 500 --threads 24 --epochs 1 --batch-size 1024 --accum-steps 4 --lr 3e-4 --max-buffer 50000 --max-games 50000

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[FATAL] Phase 2 crashed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[SYSTEM] ALL TRAINING PHASES COMPLETED SUCCESSFULLY." -ForegroundColor Green