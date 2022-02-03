#/bin/bash

# prerequisite:
# pip install tf2onnx

RUNDIR="path/to/run/dir"

python -m tf2onnx.convert --saved-model $RUNDIR/best_savedmodel --output $RUNDIR/best_model.onnx