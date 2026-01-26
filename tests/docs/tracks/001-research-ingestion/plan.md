# Track 001: Research PDF Ingestion

**Goal**: Ingest all PDF research files into the NerdLearn system and verify their "implemented" status (i.e., processed into vector embeddings and knowledge graph).

## Context
The project contains numerous PDF research papers in the root directory. The `PHASE_2_SUMMARY.md` indicates an ingestion pipeline exists (`scripts/ingest_pdfs.py`), but the database is currently empty and services are down. We need to bootstrap the environment and process these files.

## Plan

### 1. Environment Setup
- [ ] Start Docker services (API, Worker, DB, Redis, etc.).
- [ ] Verify all containers are healthy.
- [ ] Verify database schema creation (via API startup or explicit migration).

### 2. Ingestion
- [ ] Run `scripts/ingest_pdfs.py` to upload and process the PDFs.
- [ ] Monitor the `worker` logs to ensure processing (OCR, chunking, embedding) is happening.

### 3. Verification
- [ ] Query the `modules` table to see all PDFs listed.
- [ ] Check `processing_status` is 'COMPLETED' for all modules.
- [ ] Verify vector store (Qdrant) has points/chunks.
- [ ] Verify Neo4j has graph nodes.

### 4. Review
- [ ] Generate a report of successfully ingested files.
- [ ] Identify any failures or missing files.
