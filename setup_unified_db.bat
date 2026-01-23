@echo off
echo ===================================================
echo     NerdLearn Unified Database Setup Script
echo ===================================================

echo.
echo 1. Stopping existing containers...
docker-compose down

echo.
echo 2. Building and starting containers (this may take a while)...
echo    Building custom Postgres image with pgvector and AGE...
docker-compose up -d --build

echo.
echo 3. Waiting for database to initialize (20 seconds)...
timeout /t 20

echo.
echo 4. Applying database migrations...
docker-compose exec api alembic upgrade head

echo.
echo ===================================================
echo     Setup Complete!
echo ===================================================
echo.
pause
