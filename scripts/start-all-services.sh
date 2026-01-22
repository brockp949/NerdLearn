#!/bin/bash

echo "🚀 Starting NerdLearn Services"
echo "=============================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;39m' # No Color

cd "$(dirname "$0")/.."

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to check if a port is in use
port_in_use() {
  lsof -i:$1 > /dev/null 2>&1
  return $?
}

# Function to wait for service to be ready
wait_for_service() {
  local port=$1
  local service=$2
  local max_wait=30
  local count=0

  echo -n "  Waiting for $service (port $port)..."
  while ! curl -s http://localhost:$port/health > /dev/null 2>&1 && [ $count -lt $max_wait ]; do
    sleep 1
    count=$((count + 1))
    echo -n "."
  done

  if [ $count -lt $max_wait ]; then
    echo -e " ${GREEN}✓${NC}"
    return 0
  else
    echo -e " ${YELLOW}⏱️  (timeout, check logs)${NC}"
    return 1
  fi
}

echo -e "${BLUE}Step 1: Starting databases...${NC}"

if command -v docker &> /dev/null; then
  if docker compose ps | grep -q "Up"; then
    echo -e "  ${GREEN}✓${NC} Databases already running"
  else
    docker compose up -d
    echo -e "  ${GREEN}✓${NC} Databases started"
    echo "  Waiting for databases to be ready..."
    sleep 10
  fi
else
  echo -e "  ${YELLOW}⚠️  Docker not found - skipping database startup${NC}"
  echo -e "  ${YELLOW}   Note: Services will fail if databases aren't running${NC}"
fi

echo ""
echo -e "${BLUE}Step 2: Starting Python services...${NC}"

# Array of services: name, port, directory
declare -a services=(
  "API Gateway:8000:api-gateway"
  "Scheduler:8001:scheduler"
  "Telemetry:8002:telemetry"
  "Inference:8003:inference"
  "Content Ingestion:8004:content-ingestion"
  "Orchestrator:8005:orchestrator"
)

for service in "${services[@]}"; do
  IFS=':' read -r name port dir <<< "$service"

  if port_in_use $port; then
    echo -e "  ${YELLOW}⚠️${NC}  $name (port $port already in use)"
  else
    echo -e "  Starting $name on port $port..."
    cd services/$dir
    nohup python main.py > ../../logs/$dir.log 2>&1 &
    echo $! > ../../logs/$dir.pid
    cd ../..
    echo -e "  ${GREEN}✓${NC} $name (PID: $(cat logs/$dir.pid))"
  fi
done

echo ""
echo -e "${BLUE}Step 3: Starting Next.js frontend...${NC}"

if port_in_use 3000; then
  echo -e "  ${YELLOW}⚠️${NC}  Next.js (port 3000 already in use)"
else
  echo -e "  Starting Next.js on port 3000..."
  cd apps/web
  nohup pnpm dev > ../../logs/frontend.log 2>&1 &
  echo $! > ../../logs/frontend.pid
  cd ../..
  echo -e "  ${GREEN}✓${NC} Next.js (PID: $(cat logs/frontend.pid))"
fi

echo ""
echo -e "${GREEN}🎉 All services started!${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Service Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}Backend Services:${NC}"
echo "  API Gateway:        http://localhost:8000"
echo "  Scheduler (FSRS):   http://localhost:8001"
echo "  Telemetry (ECD):    http://localhost:8002"
echo "  Inference (DKT):    http://localhost:8003"
echo "  Content Pipeline:   http://localhost:8004"
echo "  Orchestrator:       http://localhost:8005"
echo ""
echo -e "${BLUE}Frontend:${NC}"
echo "  Next.js App:        http://localhost:3000"
echo ""
echo -e "${BLUE}Databases:${NC}"
echo "  PostgreSQL:         localhost:5432"
echo "  Neo4j Browser:      http://localhost:7474"
echo "  TimescaleDB:        localhost:5433"
echo "  Redis:              localhost:6379"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}📝 Logs:${NC}"
echo "  All service logs are in ./logs/"
echo "  View logs: tail -f logs/<service>.log"
echo ""
echo -e "${BLUE}🛑 Stop services:${NC}"
echo "  ./scripts/stop-all-services.sh"
echo ""
echo -e "${BLUE}🔐 Login:${NC}"
echo "  Email: demo@nerdlearn.com"
echo "  Password: demo123"
echo ""
