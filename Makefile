# survey2agent — release Makefile
#
# Usage:
#   make help       Show this help
#   make install    Install package + api/dev/hf extras
#   make fetch      Download benchmark ZIP from Hugging Face (~36 MB; expands to ~410 MB)
#   make smoke      Fast end-to-end smoke test (~30 s, no API)
#   make reproduce  Reproduce all 23 paper scripts (~40-60 min, no API)
#   make regenerate-appendix-f-results
#                   Rebuild Appendix-F aggregate JSONs (~45-60 min, no API)
#   make all        install + fetch + smoke + reproduce
#   make test       Full pytest suite (~35-45 min)
#   make clean      Remove generated outputs and pytest cache
#
# All targets are POSIX-shell compatible (bash and zsh).

PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip
PYTHONPATH_ := src

export PYTHONPATH := $(PYTHONPATH_)

.PHONY: help install fetch smoke reproduce reproduce-main reproduce-appendix regenerate-appendix-f-results regenerate-appendix-c-results all test clean

help:
	@echo "survey2agent — release targets"
	@echo ""
	@echo "  make install              Install package + api/dev/hf extras   (~30 s)"
	@echo "  make fetch                Download benchmark ZIP from HF       (~1-3 min; ~36 MB -> ~410 MB)"
	@echo "  make smoke                End-to-end smoke test                 (~30 s, no API)"
	@echo "  make reproduce            Reproduce all 23 paper scripts        (~40-60 min, no API)"
	@echo "  make reproduce-main       Reproduce 4 main tables only          (~20-30 min)"
	@echo "  make reproduce-appendix   Reproduce 19 appendix tables only     (~15-25 min)"
	@echo "  make regenerate-appendix-f-results"
	@echo "                            Rebuild Appendix-F aggregate JSONs    (~45-60 min, no API)"
	@echo "  make regenerate-appendix-c-results"
	@echo "                            Alias for regenerate-appendix-f-results"
	@echo "  make seed-<seed>          Re-generate one seed's benchmark      (~5-10 min)"
	@echo "  make all                  install + fetch + smoke + reproduce"
	@echo "  make test                 Full pytest suite                     (~35-45 min)"
	@echo "  make clean                Remove paper_artifacts/output and pytest cache"

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e '.[api,dev,hf]'

fetch:
	$(PYTHON) data/fetch_benchmark.py

smoke:
	$(PYTHON) -m pytest tests/test_smoke_end_to_end.py examples/02_custom_question/test_custom_question.py examples/05_custom_stream/test_custom_stream.py -v

reproduce:
	$(PYTHON) -m paper_artifacts.reproduce_paper

reproduce-main:
	$(PYTHON) -m paper_artifacts.reproduce_paper --tier main

reproduce-appendix:
	$(PYTHON) -m paper_artifacts.reproduce_paper --tier appendix

regenerate-appendix-f-results:
	$(PYTHON) scripts/regenerate_appendix_f_results.py

regenerate-appendix-c-results: regenerate-appendix-f-results

# Re-generate one seed's benchmark from scratch.
# Example: `make seed-s20260321` runs the four data-generation stages
# (personas → events → sources → ground_truth) into
# `data/benchmark/seeds/<seed>/`. NL renders are produced inline by the
# extractor; no separate render step is needed. Existing data in the
# target directory is overwritten.
.PHONY: seed-%
seed-%:
	@echo "Re-generating $* into data/benchmark/seeds/$*/"
	$(PYTHON) -m survey2agent.data_generation.generate_personas \
	    --seed $$(echo $* | sed 's/^s//') \
	    --output-dir data/benchmark/seeds/$*
	$(PYTHON) -m survey2agent.data_generation.generate_events       --dataset-dir data/benchmark/seeds/$*
	$(PYTHON) -m survey2agent.data_generation.generate_sources      --dataset-dir data/benchmark/seeds/$*
	$(PYTHON) -m survey2agent.data_generation.generate_ground_truth --dataset-dir data/benchmark/seeds/$*

all: install fetch smoke reproduce

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf paper_artifacts/output/main/*.csv paper_artifacts/output/main/*.md
	rm -rf paper_artifacts/output/appendix/*.csv paper_artifacts/output/appendix/*.md
	rm -rf .pytest_cache
