# Database Setup Guide

## Prerequisites

- Docker installed and running
- Node.js and pnpm installed
- Python 3.11+ installed

## Quick Start

```bash
# 1. Install all dependencies
./scripts/install-all-deps.sh

# 2. Start databases
docker compose up -d

# 3. Wait for databases to be ready (30 seconds)
sleep 30

# 4. Run Prisma migrations
cd packages/db
npx prisma db push

# 5. Seed the database
npx tsx prisma/seed.ts

# 6. Start all services
cd ../..
./scripts/start-all-services.sh
```

## Detailed Steps

### Step 1: Start Databases

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** (port 5432) - Main relational database
- **Neo4j** (ports 7474, 7687) - Knowledge Graph
- **TimescaleDB** (port 5433) - Time-series telemetry data
- **Redis** (port 6379) - Caching and sessions
- **Redpanda** (port 9092) - Event streaming
- **Milvus** (port 19530) - Vector similarity search

**Verify databases are running:**

```bash
docker compose ps
```

All services should show status "Up".

---

### Step 2: Configure Prisma

```bash
cd packages/db
```

**Create .env file:**

```bash
cp .env.example .env
```

The .env file should contain:

```
DATABASE_URL="postgresql://nerdlearn:nerdlearn_dev_password@localhost:5432/nerdlearn"
```

---

### Step 3: Generate Prisma Client

```bash
npx prisma generate
```

This creates the TypeScript Prisma Client based on the schema.

---

### Step 4: Push Schema to Database

```bash
npx prisma db push
```

This creates all tables in PostgreSQL:
- User
- LearnerProfile
- Concept
- Card
- ScheduledItem
- CompetencyState
- Evidence
- and more...

**Verify tables were created:**

```bash
npx prisma studio
```

This opens a web UI at http://localhost:5555 where you can browse the database.

---

### Step 5: Configure Neo4j

**Open Neo4j Browser:** http://localhost:7474

**Login:**
- Username: `neo4j`
- Password: `nerdlearn_dev_password`

**Create constraints:**

```cypher
CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT concept_name IF NOT EXISTS
FOR (c:Concept) REQUIRE c.name IS UNIQUE;
```

**Verify:**

```cypher
SHOW CONSTRAINTS;
```

---

### Step 6: Seed Demo Data

```bash
npx tsx prisma/seed.ts
```

This creates:

**Users (3):**
- demo@nerdlearn.com (password: demo123)
- alice@example.com (password: alice123)
- bob@example.com (password: bob123)

**Concepts (10):**
- Python Variables
- Python Functions
- Python Loops
- Python Lists
- Python Dictionaries
- Control Flow
- Python Recursion
- Error Handling
- File I/O
- Python Classes

**Learning Cards (30):**
- 3 cards per concept
- Content with markdown explanations
- Questions and answers
- Difficulty ratings

**Knowledge Graph:**
- 10 concept nodes in Neo4j
- 7 prerequisite relationships

**Scheduled Items:**
- 30 items scheduled for demo user
- 10 due immediately
- 20 due in the coming days

**Expected Output:**

```
🌱 Starting seed...
🧹 Cleaning existing data...
✅ Cleaned existing data
👤 Creating demo users...
✅ Created 3 users: demo, alice, bob
💡 Creating Python concepts...
✅ Created 10 concepts
📝 Creating learning cards...
✅ Created 30 learning cards
🌳 Building Knowledge Graph in Neo4j...
✅ Created Knowledge Graph with 10 nodes and 7 edges
📅 Creating initial scheduled items...
✅ Created 30 scheduled items for demo user
📊 Creating initial competency states...
✅ Created initial competency states

🎉 Seed completed successfully!
```

---

### Step 7: Verify Data

**PostgreSQL (via Prisma Studio):**

```bash
npx prisma studio
```

Check:
- Users table has 3 rows
- Concept table has 10 rows
- Card table has 30 rows
- ScheduledItem table has 30 rows

**Neo4j (via Browser):**

Open http://localhost:7474 and run:

```cypher
// View all concepts
MATCH (c:Concept) RETURN c;

// View knowledge graph
MATCH (c:Concept)-[r:HAS_PREREQUISITE]->(target:Concept)
RETURN c, r, target;

// Count nodes and relationships
MATCH (c:Concept) RETURN count(c) as concepts;
MATCH ()-[r:HAS_PREREQUISITE]->() RETURN count(r) as prerequisites;
```

Expected:
- 10 concept nodes
- 7 prerequisite relationships

---

## Troubleshooting

### Database Connection Errors

**Error:** `connection refused` or `could not connect`

**Solution:**
1. Check Docker is running: `docker compose ps`
2. Wait longer for databases to initialize (try 60 seconds)
3. Check logs: `docker compose logs postgres`

---

### Prisma Generate Fails

**Error:** `Prisma schema not found`

**Solution:**
1. Make sure you're in `packages/db` directory
2. Check that `prisma/schema.prisma` exists
3. Run `pnpm install` first

---

### Seed Script Fails

**Error:** `Foreign key constraint failed`

**Solution:**
1. Run seed script again (it cleans data first)
2. Make sure Prisma migrations ran: `npx prisma db push`
3. Check database is accessible

**Error:** `Neo4j connection failed`

**Solution:**
- Seed will skip Neo4j if not available
- Check Neo4j is running: `docker compose ps neo4j`
- Verify credentials: neo4j / nerdlearn_dev_password
- Check port 7687 is accessible

---

### Password Hashing Note

The seed script stores passwords with `PLACEHOLDER_` prefix. These are **NOT hashed**.

When you first login via the API Gateway:
1. API Gateway will detect the `PLACEHOLDER_` prefix
2. Hash the password with bcrypt
3. Update the database with the hashed password

This is intentional to allow the seed script to run without bcrypt dependency in TypeScript.

**For production:** Always hash passwords before storing!

---

## Database Schema Overview

### Core Tables

**User** - Authentication and basic info
- id, email, username, passwordHash, fullName

**LearnerProfile** - Cognitive state and gamification
- userId (FK)
- cognitiveEmbedding (6D vector for Bloom's levels)
- fsrsStability, fsrsDifficulty (FSRS parameters)
- currentZpdLower, currentZpdUpper (ZPD bounds)
- totalXP, level, streakDays

**Concept** - Learning topics
- id, name, description, domain
- bloomLevel (REMEMBER, UNDERSTAND, APPLY, ANALYZE, EVALUATE, CREATE)
- estimatedDifficulty (1-10)
- conceptEmbedding (vector representation)

**Card** - Learning content
- id, conceptId (FK)
- content (markdown)
- question, correctAnswer
- difficulty, cardType

**ScheduledItem** - FSRS scheduling state
- id, learnerId (FK), cardId (FK)
- currentStability, currentDifficulty, retrievability
- intervalDays, nextDueDate
- reviewCount

**CompetencyState** - Learner knowledge per concept
- id, learnerId (FK), conceptId (FK)
- knowledgeProbability (0-1, from DKT)
- masteryLevel (0-1)
- lastAssessed, evidenceCount

**Evidence** - ECD observations
- id, learnerId (FK), cardId (FK)
- evidenceType (PERFORMANCE, ENGAGEMENT, DWELL_TIME, etc.)
- observableData (JSON)

---

## Database Migrations

### Making Schema Changes

1. Edit `packages/db/prisma/schema.prisma`
2. Push changes:
   ```bash
   npx prisma db push
   ```
3. Regenerate client:
   ```bash
   npx prisma generate
   ```

### Resetting Database

**⚠️ Warning: This deletes ALL data!**

```bash
npx prisma db push --force-reset
npx tsx prisma/seed.ts
```

---

## Production Considerations

### Security

- [ ] Change default database passwords
- [ ] Enable SSL for PostgreSQL
- [ ] Use secrets management (not .env files)
- [ ] Implement proper password hashing in seed
- [ ] Set up database backups

### Performance

- [ ] Add database indices (Prisma schema)
- [ ] Set up connection pooling
- [ ] Configure TimescaleDB hypertables
- [ ] Optimize Neo4j queries with indices

### Monitoring

- [ ] Set up database health checks
- [ ] Monitor query performance
- [ ] Track database size
- [ ] Set up alerting

---

## Next Steps

After database setup:

1. **Start all services:** `./scripts/start-all-services.sh`
2. **Test login:** http://localhost:3000
3. **Try learning:** Click "Start Learning"
4. **Check data:** Use Prisma Studio to see XP updates

---

## Resources

- [Prisma Docs](https://www.prisma.io/docs)
- [Neo4j Cypher Guide](https://neo4j.com/docs/cypher-manual/current/)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [TimescaleDB Docs](https://docs.timescale.com/)
