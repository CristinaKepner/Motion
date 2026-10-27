#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
python tests/test_adapters.py
python tests/test_genea.py
python scripts/download_genea.py
python scripts/retarget_genea.py
python scripts/build_genea_catalog.py
A3_OUT=results/genea python scripts/build_library.py --dataset-name 'GENEA 2023 · 24 training recordings' --dataset-note '24 recordings; CC BY-NC 4.0; rule-derived semantics, not human labels'
python scripts/run_benchmark.py library
python review_page/build_review.py results/genea/library/review_bundle.json review_page/dist/A3_genea_review.html
python scripts/export_genea_sources.py
A3_OUT=results/genea A3_DATA=results/genea/source_csv python scripts/viz_library.py --source-fps 30 --dataset-name 'GENEA 2023 · 24 training recordings'
python review_page/build_atlas.py results/genea/viz/viz_data.json review_page/dist/A3_genea_atlas.html
python scripts/sonic_validate.py
python scripts/speech_gesture_demo.py
python review_page/build_benchmark.py
python review_page/build_demo.py
python scripts/semantic_style_matrix.py
