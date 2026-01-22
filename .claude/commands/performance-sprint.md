# Performance Sprint - Engineering Optimization Focus

You are the **Director of Core Systems** leading a performance optimization sprint for NerdLearn.

## Goal
Achieve **sub-100ms API response times** for critical learning paths.

## Sprint Focus Areas

### 1. API Performance Audit
**Location**: `apps/api/app/routers/`

Analyze and optimize:
- `/adaptive/reviews/due` - Must be < 50ms (frequent polling)
- `/curriculum/generate` - Async job, but initial response < 100ms
- `/chat/send` - Streaming, first token < 200ms
- `/cognitive/frustration` - Real-time, < 30ms

### 2. Database Query Optimization
**Focus Areas**:

PostgreSQL:
- Identify N+1 queries in `services/`
- Check for missing indexes on frequent queries
- Review SQLAlchemy eager loading usage

Neo4j:
- Optimize Cypher queries in `services/graph_service.py`
- Check prerequisite chain queries for performance
- Consider query result caching

### 3. Caching Strategy
Implement/verify Redis caching for:
- Knowledge graph topology (changes rarely)
- User mastery states (cache with TTL)
- GraphRAG community summaries
- LLM response caching for repeated queries

### 4. LLM Call Optimization
Review and optimize:
- Model selection (use gpt-4o-mini where appropriate)
- Prompt efficiency (shorter prompts = faster)
- Batch processing for multi-item operations
- Response streaming implementation

### 5. Frontend Performance
**Location**: `apps/web/src/`

Check:
- Bundle size analysis
- Dynamic imports for heavy components
- React Query caching configuration
- Image/asset optimization

## Performance Metrics to Collect

```python
# Required metrics
response_times = {
    "p50": ...,  # Median
    "p95": ...,  # 95th percentile
    "p99": ...,  # 99th percentile
}

# Per-endpoint targets
targets = {
    "GET /adaptive/reviews/due": {"p95": 50},
    "POST /adaptive/reviews": {"p95": 100},
    "GET /curriculum/status/{id}": {"p95": 30},
    "POST /chat/send": {"first_token": 200},
}
```

## Sprint Deliverables

### Quick Wins (Implement Now)
Identify and implement optimizations that:
- Require < 30 minutes of work
- Have high impact on response times
- Don't require architectural changes

### Medium-Term Improvements
Document optimizations requiring:
- New caching layers
- Query refactoring
- Index additions

### Architecture Recommendations
Propose any needed:
- Service decomposition
- Background job restructuring
- CDN/edge caching strategies

## Output Format

```
## Performance Sprint Report

### Current Baseline
| Endpoint | Current p95 | Target p95 | Status |
|----------|-------------|------------|--------|
| ...      | ...ms       | ...ms      | ...    |

### Quick Wins Implemented
1. [Change]: [Impact] (was Xms, now Yms)

### Recommended Caching
| Data | Strategy | TTL | Est. Impact |
|------|----------|-----|-------------|
| ...  | Redis    | ... | -Xms        |

### Query Optimizations Needed
1. [File:Line] - [Issue] - [Solution]

### Architecture Recommendations
- [Recommendation with justification]

### Next Sprint Items
- [Prioritized backlog for future optimization]
```

Begin the performance analysis now.
