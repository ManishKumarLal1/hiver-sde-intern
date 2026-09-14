.PHONY: setup evaluate baselines agent clean

setup:
	pip install -r requirements.txt

agent:
	python src/agent_hybrid.py
	python src/fix_agent_reasons.py

baselines:
	python src/baseline_trivial.py
	python src/baseline_simple.py

evaluate: agent
	python src/eval_harness.py predictions/agent.csv
	python src/eval_harness.py predictions/baseline_simple.csv
	python src/eval_harness.py predictions/baseline_trivial.csv

clean:
	rm -rf predictions/*.csv results/*.json __pycache__