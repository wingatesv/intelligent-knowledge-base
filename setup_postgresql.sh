#!/bin/bash
sudo apt update
echo | sudo apt install -y postgresql-common
echo | sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
echo | sudo apt install postgresql-15-pgvector
sudo service postgresql start
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'password';"
sudo -u postgres psql -c "CREATE DATABASE vector_db;"
