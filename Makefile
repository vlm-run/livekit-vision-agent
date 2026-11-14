# Reads .env.local / .env. `make help` lists the targets.
-include .env .env.local
export

.PHONY: help install dev frontend demo demo-local test

help: ## List the targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync --extra dev

dev: ## Run the agent worker with the demo page support
	uv run src/demo.py dev

frontend: ## Serve the demo page at http://localhost:8080
	uv run frontend/serve.py

demo: ## Worker + demo page together (Ctrl-C stops both)
	uv run src/demo.py dev & uv run frontend/serve.py & wait

demo-local: ## `demo` against a gateway from `make realtime-up` in vlm-playground (localhost:8001)
	VLMRUN_GATEWAY_WS=ws://localhost:8001/v1/ws VLMRUN_API_KEY=$${LOCAL_GATEWAY_KEY:-vlmrun-local} $(MAKE) demo

test: ## Run the unit tests
	uv run pytest
