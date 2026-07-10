# Convenience targets — thin wrappers around scripts/ (which do the real work)
.PHONY: setup run serve check ingest update docker docker-ollama docker-logs clean

setup:            ## Create venv, install deps, prepare .env
	./scripts/setup.sh

run:              ## Interactive CLI
	./scripts/run.sh

serve:            ## REST API on :8000
	./scripts/serve.sh

check:            ## Health-check provider + index
	./scripts/check.sh

ingest:           ## Ingest default open-source dataset (override: make ingest DATASET=ag-news DOCS=200)
	./scripts/ingest.sh --name $(or $(DATASET),wikipedia-simple) --max-docs $(or $(DOCS),500)

update:           ## Re-pull datasets + incremental re-index (cron target)
	./scripts/update_data.sh

docker:           ## Build & start API container
	docker compose up -d --build rag-api

docker-ollama:    ## Start API + local Ollama LLM
	docker compose --profile ollama up -d --build

docker-logs:
	docker compose logs -f rag-api

clean:            ## Remove generated state (index cache, logs) — corpus is kept
	rm -rf multi_agent_rag/index_state multi_agent_rag/logs multi_agent_rag/chroma_db .update.lock
