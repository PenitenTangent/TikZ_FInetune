PYTHON ?= python3
PACKAGE = tikz_mlx
CONFIG ?= configs/lora_prod.yaml
DESCRIPTION ?= Draw a simple labeled triangle with black edges and vertex labels A, B, and C.
OUTDIR ?= outputs/example
DATASET_ID ?= nllg/DaTikZ-V4
SPLIT ?= train
VAL_FRACTION ?= 0.10
GOLD_EVAL_FRACTION ?= 0.05
TRAIN_DATASET ?= data/prepared/train_unified.jsonl
VAL_DATASET ?= data/prepared/val_unified.jsonl
RESUME_OUTPUT ?= runs/tikz_lora_adapter_full_resume.safetensors

.PHONY: install prep check-dataset prep-dataset split-dataset infer train train-resume-latest train-stage2 full-finetune test

install:
	$(PYTHON) -m pip install -e ".[dev]"

prep:
	$(PYTHON) -m $(PACKAGE).cli validate-config --config $(CONFIG)

check-dataset:
	$(PYTHON) -m $(PACKAGE).cli check-dataset --dataset-id $(DATASET_ID) --split $(SPLIT)

prep-dataset:
	$(PYTHON) -m $(PACKAGE).cli prepare-dataset --config $(CONFIG) --dataset-id $(DATASET_ID) --split $(SPLIT) --overwrite

split-dataset:
	$(PYTHON) -m $(PACKAGE).cli split-dataset --config $(CONFIG) --val-fraction $(VAL_FRACTION) --gold-eval-fraction $(GOLD_EVAL_FRACTION) --overwrite

infer:
	$(PYTHON) -m $(PACKAGE).cli infer --config $(CONFIG) --description "$(DESCRIPTION)" --output-dir $(OUTDIR)

train:
	$(PYTHON) -m $(PACKAGE).cli train --config $(CONFIG) --dry-run

train-resume-latest:
	$(PYTHON) -m $(PACKAGE).cli train --config $(CONFIG) --dataset $(TRAIN_DATASET) --val-dataset $(VAL_DATASET) --output-path $(RESUME_OUTPUT)

train-stage2:
	$(PYTHON) -m $(PACKAGE).cli train-stage2 --config $(CONFIG) --dry-run

full-finetune:
	bash tools/run_full_finetune.sh

phase1-finetune:
	bash tools/run_phase1_finetune.sh

test:
	$(PYTHON) -m pytest
