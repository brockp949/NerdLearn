#!/bin/bash

# NerdLearn Setup Script
# Automates the complete setup process

set -e  # Exit on error

echo "🧠 NerdLearn Setup Script"
echo "========================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ first."
    exit 1
fi
echo "✅ Node.js $(node --version)"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+ first."
    exit 1
fi
echo "✅ Python $(python3 --version)"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi
echo "✅ Docker $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi
echo "✅ Docker Compose $(docker-compose --version)"

echo ""
echo "All prerequisites met!"
echo ""

# Start databases
echo "🐳 Starting databases with Docker Compose..."
docker-compose up -d

echo "⏳ Waiting for databases to be ready..."
sleep 10

# Check database health
echo "🏥 Checking database health..."
until docker exec nerdlearn-postgres pg_isready -U nerdlearn; do
  echo "⏳ Waiting for PostgreSQL..."
  sleep 2
done
echo "✅ PostgreSQL ready"

until docker exec nerdlearn-redis redis-cli ping; do
  echo "⏳ Waiting for Redis..."
  sleep 2
done
echo "✅ Redis ready"

echo ""

# Install Node dependencies
echo "📦 Installing Node.js dependencies..."
npm install

echo ""

# Set up database
echo "🗄️ Setting up database..."
cd packages/db

if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

echo "Running Prisma migrations..."
npx prisma generate
npx prisma db push

cd ../..

echo ""

# Install Python dependencies
echo "🐍 Installing Python dependencies..."

echo "  📅 Scheduler service..."
cd services/scheduler
python3 -m pip install -r requirements.txt
cd ../..

echo "  📊 Telemetry service..."
cd services/telemetry
python3 -m pip install -r requirements.txt
cd ../..

echo "  🧠 Inference service..."
cd services/inference
python3 -m pip install -r requirements.txt
cd ../..

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start development:"
echo "   npm run dev"
echo ""
echo "📊 Service URLs:"
echo "   Frontend:    http://localhost:3000"
echo "   Scheduler:   http://localhost:8001/docs"
echo "   Telemetry:   http://localhost:8002/docs"
echo "   Inference:   http://localhost:8003/docs"
echo "   Neo4j:       http://localhost:7474"
echo ""
echo "Happy learning! 🧠"
