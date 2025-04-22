# intelligent-knowledge-base
## Install Ollama on Lightning Studio
https://techxplainator.com/install-and-run-ollama-on-lightning-ai/
## Install PostgreSQL via Docker
https://medium.com/@adarsh.ajay/setting-up-postgresql-with-pgvector-in-docker-a-step-by-step-guide-d4203f6456bd
```bash
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
psql -h localhost -U postgres -d vector_db -p 5432
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


