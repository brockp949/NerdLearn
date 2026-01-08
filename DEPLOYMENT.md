# NerdLearn Production Deployment Guide

This guide covers deploying the NerdLearn AI-powered adaptive learning platform to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Server Setup](#server-setup)
3. [Environment Configuration](#environment-configuration)
4. [Database Setup](#database-setup)
5. [Deployment](#deployment)
6. [Post-Deployment](#post-deployment)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Services

- **Server**: Ubuntu 22.04 LTS or similar (min 4GB RAM, 2 CPU cores, 50GB storage)
- **Docker**: v24.0 or higher
- **Docker Compose**: v2.20 or higher
- **Domain**: Registered domain with DNS access
- **SSL Certificate**: Let's Encrypt recommended
- **OpenAI API**: Active API key with GPT-4 access

### External Services

- **PostgreSQL**: 16+ (or use Docker service)
- **Redis**: 7+ (or use Docker service)
- **Neo4j**: 5+ (or use Docker service)
- **Qdrant**: Latest (or use Docker service)
- **MinIO**: Latest (or use AWS S3)

---

## Server Setup

### 1. Initial Server Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install essential tools
sudo apt install git nginx certbot python3-certbot-nginx -y
```

### 2. Clone Repository

```bash
# Clone to deployment directory
cd /opt
sudo git clone https://github.com/yourusername/nerdlearn.git
cd nerdlearn
sudo chown -R $USER:$USER .
```

### 3. Setup SSL Certificate

```bash
# Get SSL certificate with Let's Encrypt
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Note: Certificate will be at:
# /etc/letsencrypt/live/yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

---

## Environment Configuration

### 1. Create Production Environment File

```bash
# Copy example environment file
cp .env.prod.example .env.prod

# Edit with your values
nano .env.prod
```

### 2. Required Environment Variables

**Critical Settings:**

```bash
# Application
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=<generate-strong-random-key>

# Database
DATABASE_URL=postgresql+asyncpg://nerdlearn:CHANGE_THIS_PASSWORD@postgres:5432/nerdlearn

# OpenAI
OPENAI_API_KEY=sk-your-actual-api-key

# Security
JWT_SECRET_KEY=<generate-strong-random-key>
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Monitoring
SENTRY_DSN=<your-sentry-dsn-if-using>
```

**Generate Secrets:**

```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Update Nginx Configuration

```bash
# Update nginx.conf with your domain
sed -i 's/yourdomain.com/your-actual-domain.com/g' nginx.conf

# Copy SSL certificates to nginx directory
sudo mkdir -p /etc/nginx/ssl
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem /etc/nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem /etc/nginx/ssl/key.pem
```

---

## Database Setup

### 1. Initialize Database with Docker Compose

```bash
# Start only database services first
docker compose -f docker-compose.prod.yml up -d postgres redis neo4j qdrant minio

# Wait for services to be ready (30-60 seconds)
sleep 30

# Check service health
docker compose -f docker-compose.prod.yml ps
```

### 2. Run Database Migrations

```bash
# Run migrations from API container
docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head

# Alternatively, if you need to create initial migration:
docker compose -f docker-compose.prod.yml run --rm api alembic revision --autogenerate -m "Initial schema"
docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head
```

### 3. Create Initial Data (Optional)

```bash
# Create admin user or seed data
docker compose -f docker-compose.prod.yml run --rm api python -m app.scripts.create_admin
```

---

## Deployment

### 1. Build and Start Services

```bash
# Build all services
docker compose -f docker-compose.prod.yml build

# Start all services
docker compose -f docker-compose.prod.yml up -d

# Verify all containers are running
docker compose -f docker-compose.prod.yml ps
```

### 2. Configure Nginx Reverse Proxy

```bash
# Copy nginx configuration
sudo cp nginx.conf /etc/nginx/sites-available/nerdlearn

# Create symbolic link
sudo ln -s /etc/nginx/sites-available/nerdlearn /etc/nginx/sites-enabled/

# Test nginx configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 3. Verify Deployment

```bash
# Check API health
curl https://yourdomain.com/health

# Expected response:
# {
#   "status": "healthy",
#   "environment": "production",
#   "version": "1.0.0",
#   "services": {
#     "database": "healthy",
#     "redis": "healthy"
#   }
# }

# Check API docs
open https://yourdomain.com/docs

# Check logs
docker compose -f docker-compose.prod.yml logs -f api
```

---

## Post-Deployment

### 1. Setup Automated Backups

```bash
# Create backup script
sudo nano /opt/nerdlearn/scripts/backup.sh
```

**Backup Script:**

```bash
#!/bin/bash
BACKUP_DIR="/opt/backups/nerdlearn"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker exec nerdlearn-postgres pg_dump -U nerdlearn nerdlearn | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Backup Neo4j
docker exec nerdlearn-neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j_$DATE.dump

# Backup MinIO (if using)
docker exec nerdlearn-minio mc mirror /data $BACKUP_DIR/minio_$DATE/

# Remove backups older than 30 days
find $BACKUP_DIR -type f -mtime +30 -delete

echo "Backup completed: $DATE"
```

```bash
# Make executable
sudo chmod +x /opt/nerdlearn/scripts/backup.sh

# Add to cron (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/nerdlearn/scripts/backup.sh >> /var/log/nerdlearn_backup.log 2>&1") | crontab -
```

### 2. Setup Log Rotation

```bash
# Create logrotate configuration
sudo nano /etc/logrotate.d/nerdlearn
```

**Logrotate Config:**

```
/opt/nerdlearn/apps/api/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 root root
    sharedscripts
}
```

### 3. Configure Monitoring

**Sentry Setup:**

1. Create account at https://sentry.io
2. Create new project for "NerdLearn"
3. Copy DSN to `.env.prod` as `SENTRY_DSN`
4. Restart API: `docker compose -f docker-compose.prod.yml restart api`

**Server Monitoring:**

```bash
# Install monitoring tools
sudo apt install htop iotop nethogs -y

# Setup Docker stats dashboard
docker stats
```

---

## Monitoring & Maintenance

### Daily Checks

```bash
# Check service health
docker compose -f docker-compose.prod.yml ps
curl https://yourdomain.com/health

# Check logs for errors
docker compose -f docker-compose.prod.yml logs --tail=100 api | grep ERROR

# Check disk space
df -h

# Check memory usage
free -m
```

### Weekly Maintenance

```bash
# Update Docker images
docker compose -f docker-compose.prod.yml pull

# Restart services with new images
docker compose -f docker-compose.prod.yml up -d

# Clean up old images
docker image prune -a -f

# Check SSL certificate expiry
sudo certbot certificates
```

### Performance Monitoring

```bash
# Monitor API response times
docker compose -f docker-compose.prod.yml logs api | grep "Duration:"

# Monitor database connections
docker exec nerdlearn-postgres psql -U nerdlearn -c "SELECT count(*) FROM pg_stat_activity;"

# Monitor Redis memory
docker exec nerdlearn-redis redis-cli INFO memory
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker compose -f docker-compose.prod.yml logs <service-name>

# Check container status
docker compose -f docker-compose.prod.yml ps -a

# Restart specific service
docker compose -f docker-compose.prod.yml restart <service-name>

# Force recreate
docker compose -f docker-compose.prod.yml up -d --force-recreate <service-name>
```

### Database Connection Issues

```bash
# Test database connection
docker exec nerdlearn-postgres psql -U nerdlearn -d nerdlearn -c "SELECT 1;"

# Check database logs
docker compose -f docker-compose.prod.yml logs postgres

# Reset database connection pool
docker compose -f docker-compose.prod.yml restart api
```

### High Memory Usage

```bash
# Check memory by container
docker stats --no-stream

# Increase container limits in docker-compose.prod.yml
# Then restart:
docker compose -f docker-compose.prod.yml up -d
```

### Slow API Responses

```bash
# Check if rate limiting is too strict
# Edit .env.prod and increase RATE_LIMIT_PER_MINUTE
nano .env.prod

# Restart API
docker compose -f docker-compose.prod.yml restart api

# Check database query performance
docker exec nerdlearn-postgres psql -U nerdlearn -d nerdlearn -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
```

### SSL Certificate Issues

```bash
# Renew certificate
sudo certbot renew

# Check certificate status
sudo certbot certificates

# Test SSL configuration
curl -vI https://yourdomain.com
```

---

## Rolling Updates

### Update Application Code

```bash
# Pull latest code
cd /opt/nerdlearn
git pull origin main

# Rebuild and restart
docker compose -f docker-compose.prod.yml build api worker
docker compose -f docker-compose.prod.yml up -d api worker

# Run new migrations if any
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### Zero-Downtime Deployment (Blue-Green)

```bash
# Scale up new instances
docker compose -f docker-compose.prod.yml up -d --scale api=2

# Wait for health checks
sleep 10

# Remove old instances
docker compose -f docker-compose.prod.yml up -d --scale api=1
```

---

## Security Checklist

- [ ] All default passwords changed
- [ ] Firewall configured (UFW or iptables)
- [ ] SSH key-based authentication only
- [ ] SSL/TLS certificates installed and auto-renewing
- [ ] Environment variables secured (not in version control)
- [ ] Database backups automated
- [ ] Monitoring and alerting configured
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Security headers enabled
- [ ] Sentry or error monitoring active
- [ ] Log rotation configured
- [ ] Regular security updates scheduled

---

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:

1. ✅ Runs linting (Black, Flake8, isort)
2. ✅ Runs tests with coverage
3. ✅ Builds Docker images
4. ✅ Pushes to Docker Hub
5. ✅ Deploys to production (on main branch)

**Setup Required:**

Add these GitHub Secrets:

- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password
- `DEPLOY_SSH_KEY`: SSH private key for deployment server
- `DEPLOY_HOST`: Production server IP/hostname
- `DEPLOY_USER`: SSH user for deployment

---

## Support & Resources

- **Documentation**: https://docs.nerdlearn.com
- **API Docs**: https://yourdomain.com/docs
- **Issues**: https://github.com/yourusername/nerdlearn/issues
- **Email**: support@nerdlearn.com

---

## License

Copyright © 2026 NerdLearn. All rights reserved.
