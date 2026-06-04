.PHONY: setup data-check baseline transformer eval fairness all app test clean

# macOS Python.org builds need certifi for HuggingFace downloads
CERT := $(shell python3 -c "import certifi; print(certifi.where())" 2>/dev/null)
ifdef CERT
  export REQUESTS_CA_BUNDLE := $(CERT)
  export SSL_CERT_FILE       := $(CERT)
endif

setup:
	pip install -r requirements.txt

# Confirm the pre-built dataset is present before doing anything else
data-check:
	python3 scripts/01_prepare_data.py

# Full pipeline end-to-end: baseline → transformer → eval → fairness audit
all: baseline transformer eval fairness
	@echo ""
	@echo "✅  Full pipeline complete. Launch the demo with:  make app"

baseline:
	python3 scripts/02_train_baseline.py

transformer:
	python3 scripts/03_train_transformer.py

eval:
	python3 scripts/04_evaluate.py

fairness:
	python3 scripts/05_fairness_audit.py

app:
	streamlit run app/streamlit_app.py

test:
	python3 -m pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
