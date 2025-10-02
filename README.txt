Generate files for training

python scripts/prepare_data.py --config configs/llm_270m.yaml


To Train the model
python scripts/train.py


$env:PYTHONPATH="C:\Projects\cooking_360M_LLM_from_scratch"

!python3 -m scripts.prepare_data --config configs/llm_270m.yaml

python -m scripts.train --config .\configs\small_run.yaml

python -m scripts.generate --config configs/llm_270m.yaml --prompt "Once Upon a Time"

deepspeed --num_gpus=1 --module scripts.train --config ./configs/small_run.yaml