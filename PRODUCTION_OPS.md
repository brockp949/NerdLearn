# NerdLearn Production Operations Guide

Quick reference for common production operations and commands.

## Quick Commands Reference

### Service Management

```bash
# Start all services
docker compose -f docker-compose.prod.yml up -d

# Stop all services
docker compose -f docker-compose.prod.yml down

# Restart specific service
docker compose -f docker-compose.prod.yml restart api

# View service logs (tail)
docker compose -f docker-compose.prod.yml logs -f api

# View logs for all services
docker compose -f docker-compose.prod.yml logs -f

# Check service status
docker compose -f docker-compose.prod.yml ps
```

### Database Operations

```bash
# Run migrations
docker compose -f docker-compose.prod.yml exec api alembic upgrade head

# Rollback one migration
docker compose -f docker-compose.prod.yml exec api alembic downgrade -1

# Create new migration
docker compose -f docker-compose.prod.yml exec api alembic revision --autogenerate -m "description"

# Connect to PostgreSQL
docker exec -it nerdlearn-postgres psql -U nerdlearn -d nerdlearn

# Backup database
docker exec nerdlearn-postgres pg_dump -U nerdlearn nerdlearn > backup_$(date +%Y%m%d).sql

# Restore database
cat backup_20260108.sql | docker exec -i nerdlearn-postgres psql -U nerdlearn -d nerdlearn
```

### Application Updates

```bash
# Pull latest code
cd /opt/nerdlearn
git pull origin main

# Rebuild and restart services
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# Run any new migrations
docker compose -f docker-compose.prod.yml exec api alembic upgrade head

# Check health
curl https://yourdomain.com/health
```

### Monitoring

```bash
# Real-time resource usage
docker stats

# Check API health
curl https://yourdomain.com/health | jq

# View recent errors in logs
docker compose -f docker-compose.prod.yml logs api | grep ERROR | tail -50

# Check disk space
df -h

# Check memory
free -h

# Check CPU load
uptime
```

### Worker Management

```bash
# View worker logs
docker compose -f docker-compose.prod.yml logs -f worker

# Check Celery queue status
docker compose -f docker-compose.prod.yml exec worker celery -A app.celery_app inspect active

# Purge all tasks from queue
docker compose -f docker-compose.prod.yml exec worker celery -A app.celery_app purge

# Check registered tasks
docker compose -f docker-compose.prod.yml exec worker celery -A app.celery_app inspect registered
```

### Redis Operations

```bash
# Connect to Redis CLI
docker exec -it nerdlearn-redis redis-cli

# Check Redis memory usage
docker exec nerdlearn-redis redis-cli INFO memory

# Flush all Redis data (CAUTION!)
docker exec nerdlearn-redis redis-cli FLUSHALL

# View keys matching pattern
docker exec nerdlearn-redis redis-cli KEYS "rate_limit:*"
```

### Neo4j Operations

```bash
# Open Neo4j browser
open http://localhost:7474

# Run Cypher query
docker exec nerdlearn-neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n);"

# Backup Neo4j
docker exec nerdlearn-neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j_$(date +%Y%m%d).dump
```

### Qdrant Operations

```bash
# Check collection info
curl http://localhost:6333/collections/nerdlearn_embeddings

# View collection stats
curl http://localhost:6333/collections/nerdlearn_embeddings | jq

# Delete collection (CAUTION!)
curl -X DELETE http://localhost:6333/collections/nerdlearn_embeddings
```

### MinIO Operations

```bash
# Access MinIO console
open http://localhost:9001

# List buckets (via mc client in container)
docker exec nerdlearn-minio mc ls local/

# List objects in bucket
docker exec nerdlearn-minio mc ls local/nerdlearn-content/
```

### SSL Certificate Management

```bash
# Renew certificate
sudo certbot renew

# Force renew certificate
sudo certbot renew --force-renewal

# Check certificate expiry
sudo certbot certificates

# Test certificate renewal (dry run)
sudo certbot renew --dry-run
```

### Nginx Operations

```bash
# Test configuration
sudo nginx -t

# Reload configuration
sudo systemctl reload nginx

# Restart nginx
sudo systemctl restart nginx

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Performance Optimization

```bash
# Check slow API endpoints
docker compose -f docker-compose.prod.yml logs api | grep "Duration:" | sort -t: -k4 -n | tail -20

# Monitor database connections
docker exec nerdlearn-postgres psql -U nerdlearn -d nerdlearn -c "SELECT count(*) as connections, state FROM pg_stat_activity GROUP BY state;"

# Check query performance
docker exec nerdlearn-postgres psql -U nerdlearn -d nerdlearn -c "SELECT query, calls, total_time, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Monitor Redis connections
docker exec nerdlearn-redis redis-cli CLIENT LIST
```

### Cleanup Operations

```bash
# Remove unused Docker images
docker image prune -a

# Remove unused volumes
docker volume prune

# Clean system (CAUTION - removes everything unused)
docker system prune -a --volumes

# Clean old logs (older than 30 days)
find /opt/nerdlearn/apps/api/logs -name "*.log" -mtime +30 -delete
```

### Security Checks

```bash
# Check for running containers
docker ps -a

# Check open ports
sudo netstat -tulpn | grep LISTEN

# Check firewall status
sudo ufw status

# View failed SSH attempts
sudo grep "Failed password" /var/log/auth.log | tail -20

# Check disk encryption status
lsblk -f
```

### Backup Operations

```bash
# Full backup script
/opt/nerdlearn/scripts/backup.sh

# Manual PostgreSQL backup
docker exec nerdlearn-postgres pg_dump -U nerdlearn nerdlearn | gzip > /opt/backups/nerdlearn/db_$(date +%Y%m%d_%H%M%S).sql.gz

# Manual Neo4j backup
docker exec nerdlearn-neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j_$(date +%Y%m%d_%H%M%S).dump

# Manual MinIO backup
docker exec nerdlearn-minio mc mirror /data /backups/minio_$(date +%Y%m%d_%H%M%S)/

# List backups
ls -lh /opt/backups/nerdlearn/
```

### Emergency Procedures

#### Service Down - Quick Recovery

```bash
# 1. Check what's wrong
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs --tail=100

# 2. Try restart
docker compose -f docker-compose.prod.yml restart

# 3. If still down, force recreate
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d

# 4. Check health
curl https://yourdomain.com/health
```

#### Database Connection Pool Exhausted

```bash
# Check connections
docker exec nerdlearn-postgres psql -U nerdlearn -d nerdlearn -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
docker exec nerdlearn-postgres psql -U nerdlearn -d nerdlearn -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - interval '10 minutes';"

# Restart API to reset connection pool
docker compose -f docker-compose.prod.yml restart api
```

#### High Memory Usage

```bash
# Identify memory hog
docker stats --no-stream | sort -k4 -h

# Restart specific service
docker compose -f docker-compose.prod.yml restart <service-name>

# Clear Redis cache if needed
docker exec nerdlearn-redis redis-cli FLUSHDB
```

#### Disk Space Full

```bash
# Check disk usage
df -h

# Find large files
du -sh /* | sort -h | tail -10

# Clean Docker
docker system prune -a --volumes

# Clean old logs
find /opt/nerdlearn -name "*.log" -mtime +7 -delete

# Clean old backups
find /opt/backups -mtime +30 -delete
```

#### Rate Limiting Issues

```bash
# Check current rate limit settings
docker compose -f docker-compose.prod.yml exec api env | grep RATE_LIMIT

# Temporarily disable rate limiting
# Edit .env.prod: RATE_LIMIT_ENABLED=false
nano /opt/nerdlearn/.env.prod
docker compose -f docker-compose.prod.yml restart api

# Clear rate limit counters in Redis
docker exec nerdlearn-redis redis-cli KEYS "rate_limit:*" | xargs docker exec nerdlearn-redis redis-cli DEL
```

### API Testing

```bash
# Test health endpoint
curl -I https://yourdomain.com/health

# Test API endpoint with authentication
curl -X GET "https://yourdomain.com/api/courses" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json"

# Load test with Apache Bench
ab -n 1000 -c 10 https://yourdomain.com/health

# Load test specific endpoint
ab -n 100 -c 5 -H "Authorization: Bearer YOUR_TOKEN" https://yourdomain.com/api/courses
```

### Environment Management

```bash
# View current environment variables
docker compose -f docker-compose.prod.yml exec api env | grep -v "PASSWORD\|SECRET\|KEY"

# Update environment variable
nano /opt/nerdlearn/.env.prod
docker compose -f docker-compose.prod.yml restart api

# Validate environment file
docker compose -f docker-compose.prod.yml config
```

### Log Analysis

```bash
# Count errors by type
docker compose -f docker-compose.prod.yml logs api | grep ERROR | awk '{print $NF}' | sort | uniq -c | sort -rn

# Find slowest requests
docker compose -f docker-compose.prod.yml logs api | grep "Duration:" | awk -F'Duration: ' '{print $2}' | sort -n | tail -20

# Request count by endpoint
docker compose -f docker-compose.prod.yml logs api | grep "Request:" | awk '{print $4}' | sort | uniq -c | sort -rn

# Error rate over time
docker compose -f docker-compose.prod.yml logs --since 1h api | grep -c ERROR
```

### Scheduled Tasks

```bash
# View cron jobs
crontab -l

# Edit cron jobs
crontab -e

# View cron logs
grep CRON /var/log/syslog | tail -20
```

## Common Production Scenarios

### Deploying a New Feature

1. **Pull latest code**
   ```bash
   cd /opt/nerdlearn
   git pull origin main
   ```

2. **Build new images**
   ```bash
   docker compose -f docker-compose.prod.yml build
   ```

3. **Run migrations**
   ```bash
   docker compose -f docker-compose.prod.yml exec api alembic upgrade head
   ```

4. **Restart services**
   ```bash
   docker compose -f docker-compose.prod.yml up -d
   ```

5. **Verify deployment**
   ```bash
   curl https://yourdomain.com/health
   docker compose -f docker-compose.prod.yml logs -f api
   ```

### Rolling Back a Deployment

1. **Check git history**
   ```bash
   git log --oneline -10
   ```

2. **Revert to previous commit**
   ```bash
   git checkout <previous-commit-hash>
   ```

3. **Rebuild and restart**
   ```bash
   docker compose -f docker-compose.prod.yml build
   docker compose -f docker-compose.prod.yml up -d
   ```

4. **Rollback database if needed**
   ```bash
   docker compose -f docker-compose.prod.yml exec api alembic downgrade -1
   ```

### Scaling Services

```bash
# Scale API to 3 instances
docker compose -f docker-compose.prod.yml up -d --scale api=3

# Scale workers to 5 instances
docker compose -f docker-compose.prod.yml up -d --scale worker=5

# Verify scaling
docker compose -f docker-compose.prod.yml ps
```

## Monitoring Dashboards

### Key Metrics to Monitor

1. **API Response Time** - Target: < 200ms average
2. **Error Rate** - Target: < 1%
3. **Database Connections** - Target: < 80% of pool
4. **Memory Usage** - Target: < 80%
5. **Disk Usage** - Target: < 70%
6. **CPU Load** - Target: < 70%

### Setting Up Alerts

Create `/opt/nerdlearn/scripts/health_check.sh`:

```bash
#!/bin/bash

# Check API health
HEALTH=$(curl -s https://yourdomain.com/health | jq -r '.status')

if [ "$HEALTH" != "healthy" ]; then
    echo "ALERT: API is not healthy!" | mail -s "NerdLearn Alert" admin@yourdomain.com
fi

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')

if [ $DISK_USAGE -gt 80 ]; then
    echo "ALERT: Disk usage is at ${DISK_USAGE}%!" | mail -s "Disk Space Alert" admin@yourdomain.com
fi
```

Add to cron:
```bash
*/5 * * * * /opt/nerdlearn/scripts/health_check.sh
```

## Support Contacts

- **Technical Issues**: support@nerdlearn.com
- **Security Issues**: security@nerdlearn.com
- **On-Call**: +1-XXX-XXX-XXXX

---

**Last Updated**: 2026-01-08
**Version**: 1.0.0
