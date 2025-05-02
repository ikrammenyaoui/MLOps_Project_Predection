VENV = .venv
PYTHON = $(VENV)\Scripts\python
PIP = $(PYTHON) -m pip
UVICORN = $(VENV)\Scripts\uvicorn

.PHONY: install run test clean

install:
	@echo "Creating virtual environment..."
	python -m venv $(VENV)
	@echo "Installing package in development mode..."
	$(PIP) install -e .
	@echo "Installation complete!"

run:
	$(UVICORN) app.main:app --reload

test:
	$(PYTHON) -m app.predict

clean:
	powershell -Command "if (Test-Path '__pycache__') { Remove-Item '__pycache__' -Recurse -Force }"
	powershell -Command "if (Test-Path 'venv') { Remove-Item 'venv' -Recurse -Force }"