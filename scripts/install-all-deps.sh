#!/bin/bash

echo "📦 Installing NerdLearn Dependencies"
echo "===================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

cd "$(dirname "$0")/.."

# 1. Install Node.js dependencies
echo -e "${BLUE}📦 Installing Node.js dependencies...${NC}"
pnpm install
echo -e "${GREEN}✓ Node.js dependencies installed${NC}"
echo ""

# 2. Install Prisma and generate client
echo -e "${BLUE}🗄️  Setting up Prisma...${NC}"
cd packages/db
pnpm install
npx prisma generate
echo -e "${GREEN}✓ Prisma client generated${NC}"
cd ../..
echo ""

# 3. Install Python service dependencies
echo -e "${BLUE}🐍 Installing Python service dependencies...${NC}"

services=("scheduler" "telemetry" "inference" "content-ingestion" "api-gateway" "orchestrator")

for service in "${services[@]}"; do
  if [ -f "services/$service/requirements.txt" ]; then
    echo -e "  Installing ${service}..."
    pip install -q -r "services/$service/requirements.txt"
    echo -e "  ${GREEN}✓${NC} $service"
  else
    echo -e "  ${RED}✗${NC} $service (no requirements.txt)"
  fi
done

echo -e "${GREEN}✓ All Python dependencies installed${NC}"
echo ""

echo -e "${GREEN}🎉 All dependencies installed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Start databases: docker compose up -d"
echo "  2. Run migrations: cd packages/db && npx prisma db push"
echo "  3. Seed database: npx tsx prisma/seed.ts"
echo "  4. Start services: ./scripts/start-all-services.sh"
