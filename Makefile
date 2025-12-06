# Makefile for Face Swap AI system

.PHONY: help install install-dev setup clean test test-cov lint format \
        train train-quick download-dataset export-onnx \
        api api-dev docker-build docker-run \
        check verify graphs

# default python and pip
PYTHON := python3
PIP := pip
VENV := venv
PYTEST := pytest

# colors for terminal output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

#------------------------------------------------------------------------------
# help
#------------------------------------------------------------------------------

help:
	@echo ""
	@echo "$(GREEN)Face Swap AI - Available Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install dev dependencies (includes testing)"
	@echo "  make setup          Full setup (venv + install + download dataset)"
	@echo "  make clean          Remove cache, temp files, and venv"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test           Run all unit tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make test-models    Run model tests only"
	@echo "  make test-api       Run API tests only"
	@echo "  make verify         Run full system verification"
	@echo ""
	@echo "$(YELLOW)Training:$(NC)"
	@echo "  make download       Download LFW dataset from Kaggle"
	@echo "  make train          Train model (100 epochs)"
	@echo "  make train-quick    Quick training run (10 epochs)"
	@echo "  make export         Export trained model to ONNX"
	@echo "  make graphs         Generate loss graphs from training history"
	@echo ""
	@echo "$(YELLOW)API Server:$(NC)"
	@echo "  make api            Start production API server"
	@echo "  make api-dev        Start API server with hot reload"
	@echo "  make api-test       Test API health endpoint"
	@echo ""
	@echo "$(YELLOW)Code Quality:$(NC)"
	@echo "  make lint           Run linter (flake8)"
	@echo "  make format         Format code (black)"
	@echo "  make check          Run lint + tests"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo ""

#------------------------------------------------------------------------------
# setup and installation
#------------------------------------------------------------------------------

venv:
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created. Activate with: source $(VENV)/bin/activate"

install: venv
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(VENV)/bin/$(PIP) install --upgrade pip
	$(VENV)/bin/$(PIP) install -r requirements.txt

install-dev: install
	@echo "$(GREEN)Installing dev dependencies...$(NC)"
	$(VENV)/bin/$(PIP) install pytest pytest-cov httpx black flake8 isort

setup: install-dev download
	@echo "$(GREEN)Setup complete!$(NC)"

clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	rm -rf $(VENV)
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf *.egg-info
	rm -rf .coverage htmlcov
	rm -rf *.pyc */*.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete$(NC)"

clean-data:
	@echo "$(YELLOW)Removing dataset and checkpoints...$(NC)"
	rm -rf data/v1/raw/*
	rm -rf data/v1/aligned/*
	rm -rf checkpoints/*.pt checkpoints/*.onnx
	rm -rf graphs/*.png
	@echo "$(GREEN)Data cleanup complete$(NC)"

#------------------------------------------------------------------------------
# testing
#------------------------------------------------------------------------------

test:
	@echo "$(GREEN)Running tests...$(NC)"
	$(VENV)/bin/$(PYTEST) tests/ -v

test-cov:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(VENV)/bin/$(PYTEST) tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

test-models:
	@echo "$(GREEN)Running model tests...$(NC)"
	$(VENV)/bin/$(PYTEST) tests/test_models.py -v

test-losses:
	@echo "$(GREEN)Running loss function tests...$(NC)"
	$(VENV)/bin/$(PYTEST) tests/test_losses.py -v

test-dataset:
	@echo "$(GREEN)Running dataset tests...$(NC)"
	$(VENV)/bin/$(PYTEST) tests/test_dataset.py -v

test-api:
	@echo "$(GREEN)Running API tests...$(NC)"
	$(VENV)/bin/$(PYTEST) tests/test_api.py -v

test-video:
	@echo "$(GREEN)Running video tests...$(NC)"
	$(VENV)/bin/$(PYTEST) tests/test_video.py -v

verify:
	@echo "$(GREEN)Running full system verification...$(NC)"
	$(VENV)/bin/$(PYTHON) test_setup.py

#------------------------------------------------------------------------------
# training
#------------------------------------------------------------------------------

download:
	@echo "$(GREEN)Downloading LFW dataset...$(NC)"
	$(VENV)/bin/$(PYTHON) train.py --download --epochs 0

train:
	@echo "$(GREEN)Starting training (100 epochs)...$(NC)"
	$(VENV)/bin/$(PYTHON) train.py --epochs 100 --batch-size 8

train-quick:
	@echo "$(GREEN)Quick training run (10 epochs)...$(NC)"
	$(VENV)/bin/$(PYTHON) train.py --epochs 10 --batch-size 4

train-resume:
	@echo "$(GREEN)Resuming training from latest checkpoint...$(NC)"
	$(VENV)/bin/$(PYTHON) train.py --resume checkpoints/checkpoint_final.pt

export:
	@echo "$(GREEN)Exporting model to ONNX...$(NC)"
	$(VENV)/bin/$(PYTHON) scripts/export_onnx.py \
		--checkpoint checkpoints/checkpoint_final.pt \
		--output checkpoints/face_swap.onnx \
		--verify

graphs:
	@echo "$(GREEN)Generating loss graphs...$(NC)"
	$(VENV)/bin/$(PYTHON) -c "from training import TrainingVisualizer; \
		import torch; \
		ckpt = torch.load('checkpoints/checkpoint_final.pt', map_location='cpu'); \
		viz = TrainingVisualizer('graphs'); \
		viz.plot_losses(ckpt.get('loss_history', {}))"
	@echo "Graphs saved to graphs/"

#------------------------------------------------------------------------------
# api server
#------------------------------------------------------------------------------

api:
	@echo "$(GREEN)Starting API server...$(NC)"
	$(VENV)/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000

api-dev:
	@echo "$(GREEN)Starting API server (dev mode with reload)...$(NC)"
	$(VENV)/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api-test:
	@echo "$(GREEN)Testing API health...$(NC)"
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "$(RED)API not running$(NC)"

api-docs:
	@echo "$(GREEN)Opening API docs...$(NC)"
	@echo "Swagger UI: http://localhost:8000/docs"
	@echo "ReDoc: http://localhost:8000/redoc"

#------------------------------------------------------------------------------
# code quality
#------------------------------------------------------------------------------

lint:
	@echo "$(GREEN)Running linter...$(NC)"
	$(VENV)/bin/flake8 --max-line-length=100 --ignore=E501,W503 \
		api/ config/ dataset/ losses/ models/ training/ video/ utils/ \
		--exclude=__pycache__,venv

format:
	@echo "$(GREEN)Formatting code...$(NC)"
	$(VENV)/bin/black --line-length=100 \
		api/ config/ dataset/ losses/ models/ training/ video/ utils/ tests/
	$(VENV)/bin/isort --profile black \
		api/ config/ dataset/ losses/ models/ training/ video/ utils/ tests/

check: lint test
	@echo "$(GREEN)All checks passed!$(NC)"

#------------------------------------------------------------------------------
# docker
#------------------------------------------------------------------------------

docker-build:
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t face-swap-ai:latest .

docker-run:
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run -p 8000:8000 --gpus all face-swap-ai:latest

docker-run-cpu:
	@echo "$(GREEN)Running Docker container (CPU only)...$(NC)"
	docker run -p 8000:8000 face-swap-ai:latest

#------------------------------------------------------------------------------
# utilities
#------------------------------------------------------------------------------

swap-image:
	@echo "Usage: make swap-image SOURCE=path/to/source.jpg TARGET=path/to/target.jpg"
	@if [ -z "$(SOURCE)" ] || [ -z "$(TARGET)" ]; then \
		echo "$(RED)Error: SOURCE and TARGET must be specified$(NC)"; \
	else \
		$(VENV)/bin/$(PYTHON) -c "import requests, base64, json; \
			s = open('$(SOURCE)', 'rb').read(); \
			t = open('$(TARGET)', 'rb').read(); \
			r = requests.post('http://localhost:8000/swap/image', json={ \
				'source_image': 'data:image/jpeg;base64,' + base64.b64encode(s).decode(), \
				'target_image': 'data:image/jpeg;base64,' + base64.b64encode(t).decode() \
			}); \
			print(f'Status: {r.status_code}'); \
			if r.status_code == 200: \
				import io; from PIL import Image; \
				img = Image.open(io.BytesIO(base64.b64decode(r.json()['result'].split(',')[1]))); \
				img.save('swapped_output.jpg'); \
				print('Saved to swapped_output.jpg'); \
			else: print(r.json())"; \
	fi

swap-video:
	@echo "Usage: make swap-video SOURCE=path/to/source.jpg VIDEO=path/to/video.mp4"
	@if [ -z "$(SOURCE)" ] || [ -z "$(VIDEO)" ]; then \
		echo "$(RED)Error: SOURCE and VIDEO must be specified$(NC)"; \
	else \
		$(VENV)/bin/$(PYTHON) scripts/process_video.py \
			--source $(SOURCE) --video $(VIDEO) --output swapped_video.mp4; \
	fi

info:
	@echo "$(GREEN)Project Info$(NC)"
	@echo "Python: $$($(VENV)/bin/$(PYTHON) --version)"
	@echo "PyTorch: $$($(VENV)/bin/$(PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA: $$($(VENV)/bin/$(PYTHON) -c 'import torch; print(torch.cuda.is_available())')"
	@echo ""
	@echo "Checkpoints: $$(ls -1 checkpoints/*.pt 2>/dev/null | wc -l) files"
	@echo "ONNX models: $$(ls -1 checkpoints/*.onnx 2>/dev/null | wc -l) files"
	@echo "Graphs: $$(ls -1 graphs/*.png 2>/dev/null | wc -l) files"

count:
	@echo "$(GREEN)Lines of code:$(NC)"
	@find . -name "*.py" -not -path "./venv/*" | xargs wc -l | tail -1
