# intelligent-knowledge-base
## Install Ollama on Lightning Studio
https://techxplainator.com/install-and-run-ollama-on-lightning-ai/
### Start Ollama server
```bash
ollama serve
```
Get LLM
```bash
ollama pull mistral
```
verify LLM
```bash
ollama run mistral --verbose
/bye # to stop LLM
```
## Install PostgreSQL via Docker
https://medium.com/@adarsh.ajay/setting-up-postgresql-with-pgvector-in-docker-a-step-by-step-guide-d4203f6456bd
```bash
docker pull ankane/pgvector

docker run -e POSTGRES_USER=postgres \
           -e POSTGRES_PASSWORD=password \
           -e POSTGRES_DB=vector_db \
           --name vector_db_postgres \
           -p 5432:5432 \
           -d ankane/pgvector
```
### Install psql
```bash
sudo apt update
sudo apt install postgresql-client
```
Connect to database
```bash
# Connect to database
psql -h localhost -U postgres -d vector_db -p 5432

# Enabling pgvector
CREATE EXTENSION vector;

# Verifying the Installation
SELECT * FROM pg_extension;
```
## Clone repo
```bash
git clone https://github.com/wingatesv/intelligent-knowledge-base.git
cd intelligent-knowledge-base
```
## Install requirements
```bash
pip install -r requirements.txt
```
## Run the main.py
```bash
python main.py
```

## Llama deploy
https://docs.llamaindex.ai/en/stable/module_guides/llama_deploy/10_getting_started/
https://medium.com/google-cloud/deploying-llamaindex-workflows-to-cloud-run-with-llama-deploy-73429cfd74e3
Start an API server instance locally.
```bash
cd intelligent-knowledge-base
python -m llama_deploy.apiserver
```
From another shell, use llamactl to create the deployment:
```bash
cd intelligent-knowledge-base
llamactl deploy deployment.yml
```
```bash
Deployment successful: QuickStart
```
## Notes

