.PHONY: all test check-consistency audit clean docker-build docker-verify

PYTHON ?= python3

all: test check-consistency

test:
	$(PYTHON) musespark_harvester/verify_and_push_hf.py --self-test

check-consistency:
	$(PYTHON) musespark_harvester/check_consistency.py

audit:
	$(PYTHON) musespark_harvester/verify_and_push_hf.py --require-executed

clean:
	rm -rf __pycache__ musespark_harvester/__pycache__ *.pyc temp_eval_* solution_bin

docker-build:
	docker build -f Dockerfile.verifier -t nemotron-verifier .

docker-verify: docker-build
	docker run --rm -v $(PWD):/workspace nemotron-verifier make test check-consistency
