#!/bin/bash

echo "🛑 Stopping NerdLearn Services"
echo "=============================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cd "$(dirname "$0")/.."

# Kill services by PID files
services=("api-gateway" "scheduler" "telemetry" "inference" "content-ingestion" "orchestrator" "frontend")

echo -e "${BLUE}Stopping services...${NC}"
for service in "${services[@]}"; do
  if [ -f "logs/$service.pid" ]; then
    pid=$(cat "logs/$service.pid")
    if kill -0 $pid 2>/dev/null; then
      kill $pid 2>/dev/null
      echo -e "  ${GREEN}✓${NC} Stopped $service (PID: $pid)"
    else
      echo -e "  ${YELLOW}⚠️${NC}  $service (not running)"
    fi
    rm "logs/$service.pid"
  fi
done

# Also kill any remaining Python main.py processes
pkill -f "python main.py" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Killed remaining Python processes"

# Kill Next.js
pkill -f "next dev" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Killed Next.js"

echo ""
echo -e "${BLUE}Stopping databases (optional)...${NC}"
echo -e "  ${YELLOW}Note: Databases are left running by default${NC}"
echo -e "  To stop databases, run: ${BLUE}docker compose down${NC}"

echo ""
echo -e "${GREEN}✅ All services stopped!${NC}"
