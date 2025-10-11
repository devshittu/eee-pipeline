
# Deployment Guide

Complete guide for deploying the EEE Pipeline to production environments.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Pre-deployment Checklist](#pre-deployment-checklist)
3. [Local Development Deployment](#local-development-deployment)
4. [Docker Production Deployment](#docker-production-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Platform Deployments](#cloud-platform-deployments)
7. [GPU Configuration](#gpu-configuration)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Backup and Recovery](#backup-and-recovery)
10. [Security Hardening](#security-hardening)
11. [Performance Tuning](#performance-tuning)
12. [Scaling Strategies](#scaling-strategies)
13. [CI/CD Pipeline](#cicd-pipeline)
14. [Disaster Recovery](#disaster-recovery)

---

## Deployment Overview

### Architecture Components

The EEE Pipeline consists of 7 core services:

| Service | Purpose | Resource Requirements | Scaling Strategy |
|---------|---------|----------------------|------------------|
| **orchestrator-service** | API Gateway & Job Management | 2 CPU, 4GB RAM | Horizontal (Load Balanced) |
| **ner-service** | Named Entity Recognition | 2 CPU, 4GB RAM | Horizontal |
| **dp-service** | Dependency Parsing | 2 CPU, 4GB RAM | Horizontal |
| **event-llm-service** | LLM Event Extraction | 4 CPU, 16GB RAM, 1 GPU | Vertical (GPU) |
| **celery-worker** | Async Task Processing | 4 CPU, 8GB RAM | Horizontal |
| **redis** | Message Broker & Cache | 2 CPU, 4GB RAM | Single (with persistence) |
| **postgres** (optional) | Persistent Storage | 2 CPU, 8GB RAM | Single (with replication) |

### Deployment Patterns

**Pattern 1: All-in-One (Development)**
```

Single server running all services via docker-compose
├── Pros: Simple, low cost
└── Cons: No redundancy, limited scale

```

**Pattern 2: Microservices (Small Production)**
```

Separate containers per service
├── Pros: Independent scaling, fault isolation
└── Cons: Network overhead, orchestration complexity

```

**Pattern 3: Kubernetes (Large Production)**
```

K8s cluster with auto-scaling
├── Pros: Auto-scaling, self-healing, enterprise-grade
└── Cons: Complex setup, higher cost

```

---

## Pre-deployment Checklist

### Infrastructure Requirements

**Compute**:
- [ ] Minimum 16 CPU cores (32 recommended)
- [ ] Minimum 64GB RAM (128GB recommended)
- [ ] NVIDIA GPU with 16GB+ VRAM (RTX A4000, A5000, or better)
- [ ] 500GB SSD storage (1TB recommended)

**Network**:
- [ ] Static IP address or domain name
- [ ] Firewall rules configured
- [ ] SSL/TLS certificates obtained
- [ ] CDN configured (optional, for API responses)

**Software**:
- [ ] Docker 24.0+ installed
- [ ] Docker Compose v2.20+ installed
- [ ] NVIDIA Container Toolkit installed (GPU deployments)
- [ ] Backup solution configured

**Security**:
- [ ] Secrets management system (Vault, AWS Secrets Manager)
- [ ] API authentication mechanism decided
- [ ] Network policies defined
- [ ] Compliance requirements reviewed (GDPR, HIPAA, etc.)

### Configuration Preparation

**1. Create Production Configuration**
```bash
cp config/settings.yaml.example config/settings.production.yaml
```

**2. Set Environment-Specific Values**

```yaml
# config/settings.production.yaml
general:
  environment: "production"
  log_level: "INFO"  # Not DEBUG in production
  gpu_enabled: true

orchestrator_service:
  host: "0.0.0.0"
  port: 8000
  request_timeout_seconds: 300
  batch_processing_chunk_size: 20

event_llm_service:
  max_new_tokens: 8192
  temperature: 0.0  # Deterministic for production
  generation_max_retries: 3

celery:
  broker_url: "redis://redis:6379/0"
  result_backend: "redis://redis:6379/1"
  worker_concurrency: 8  # Match CPU cores

storage:
  backends:
    - type: "postgres"
      enabled: true
      connection_string: "${POSTGRES_CONNECTION_STRING}"
    - type: "jsonl"
      enabled: true
      output_directory: "/data/results"
```

**3. Environment Variables**

```bash
# Create .env file
cat > .env << EOF
# Database
POSTGRES_USER=eee_user
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=eee_pipeline
POSTGRES_CONNECTION_STRING=postgresql://eee_user:${POSTGRES_PASSWORD}@postgres:5432/eee_pipeline

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32)

# Security
SECRET_KEY=$(openssl rand -hex 32)
API_KEY=$(openssl rand -base64 32)

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# Model paths
MODEL_CACHE_DIR=/models
HF_HOME=/models/huggingface
EOF

chmod 600 .env
```

---

## Local Development Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/eee-pipeline.git
cd eee-pipeline

# Copy configuration
cp config/settings.yaml.example config/settings.yaml

# Start services (CPU mode)
docker compose up -d

# Check status
docker compose ps
docker compose logs -f

# Run health check
curl http://localhost:8000/health
```

### Development with Hot Reload

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  orchestrator-service:
    volumes:
      - ./src:/app/src:ro  # Mount source for hot reload
      - ./config:/app/config:ro
    environment:
      - PYTHONUNBUFFERED=1
      - RELOAD=true  # Enable auto-reload
    command: >
      uvicorn src.api.orchestrator_service:app
      --host 0.0.0.0
      --port 8000
      --reload
      --reload-dir /app/src
```

```bash
# Start with dev overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Local GPU Testing

```bash
# Verify GPU available
nvidia-smi

# Start with GPU support
docker compose -f docker-compose.gpu.yml up -d

# Verify GPU used by container
docker compose exec event-llm-service nvidia-smi
```

---

## Docker Production Deployment

### Production Docker Compose

**File: `docker-compose.production.yml`**

```yaml
version: '3.8'

services:
  orchestrator-service:
    image: your-registry.com/eee-pipeline/orchestrator:${VERSION}
    restart: always
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./config/settings.production.yaml:/app/config/settings.yaml:ro
      - logs:/app/logs
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      replicas: 3  # Run 3 instances behind load balancer
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  ner-service:
    image: your-registry.com/eee-pipeline/ner:${VERSION}
    restart: always
    env_file:
      - .env
    volumes:
      - ./config/settings.production.yaml:/app/config/settings.yaml:ro
      - models:/app/models
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G

  dp-service:
    image: your-registry.com/eee-pipeline/dp:${VERSION}
    restart: always
    env_file:
      - .env
    volumes:
      - ./config/settings.production.yaml:/app/config/settings.yaml:ro
      - models:/app/models
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G

  event-llm-service:
    image: your-registry.com/eee-pipeline/event-llm:${VERSION}
    restart: always
    env_file:
      - .env
    volumes:
      - ./config/settings.production.yaml:/app/config/settings.yaml:ro
      - models:/app/models
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0

  celery-worker:
    image: your-registry.com/eee-pipeline/celery:${VERSION}
    restart: always
    env_file:
      - .env
    volumes:
      - ./config/settings.production.yaml:/app/config/settings.yaml:ro
      - results:/data/results
    command: >
      celery -A src.core.celery_tasks worker
      --loglevel=info
      --concurrency=8
      --max-tasks-per-child=100
    depends_on:
      - redis
      - ner-service
      - dp-service
      - event-llm-service
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G

  redis:
    image: redis:7-alpine
    restart: always
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 4gb
      --maxmemory-policy allkeys-lru
      --appendonly yes
      --appendfsync everysec
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    restart: always
    env_file:
      - .env
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - logs:/var/log/nginx
    depends_on:
      - orchestrator-service

volumes:
  models:
    driver: local
  redis-data:
    driver: local
  postgres-data:
    driver: local
  results:
    driver: local
  logs:
    driver: local

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

### NGINX Load Balancer Configuration

**File: `nginx/nginx.conf`**

```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=batch:10m rate=10r/m;

    # Upstream services
    upstream orchestrator {
        least_conn;
        server orchestrator-service-1:8000 max_fails=3 fail_timeout=30s;
        server orchestrator-service-2:8000 max_fails=3 fail_timeout=30s;
        server orchestrator-service-3:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # HTTP -> HTTPS redirect
    server {
        listen 80;
        server_name api.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name api.yourdomain.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # API endpoints
        location /v1/documents {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://orchestrator;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            
            # Timeouts for simple documents
            proxy_connect_timeout 10s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        location /v1/documents/enriched {
            limit_req zone=batch burst=5 nodelay;
            
            proxy_pass http://orchestrator;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # No timeout for async endpoints
            proxy_connect_timeout 10s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /v1/documents/batch {
            limit_req zone=batch burst=2 nodelay;
            
            proxy_pass http://orchestrator;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            proxy_connect_timeout 10s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /v1/jobs {
            limit_req zone=api burst=50 nodelay;
            
            proxy_pass http://orchestrator;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 10s;
        }

        location /health {
            proxy_pass http://orchestrator;
            access_log off;
        }

        location /docs {
            proxy_pass http://orchestrator;
        }

        # Deny all other paths
        location / {
            return 404;
        }
    }
}
```

### Deployment Script

**File: `scripts/deploy-production.sh`**

```bash
#!/bin/bash
set -e

# Configuration
REGISTRY="your-registry.com/eee-pipeline"
VERSION="${1:-latest}"
COMPOSE_FILE="docker-compose.production.yml"

echo "========================================="
echo "EEE Pipeline Production Deployment"
echo "========================================="
echo "Version: $VERSION"
echo "Registry: $REGISTRY"
echo ""

# Pre-flight checks
echo "[1/8] Running pre-flight checks..."
command -v docker >/dev/null 2>&1 || { echo "Error: docker not installed"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Error: docker-compose not installed"; exit 1; }

if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    exit 1
fi

if [ ! -f "config/settings.production.yaml" ]; then
    echo "Error: Production config not found"
    exit 1
fi

# Pull latest images
echo "[2/8] Pulling Docker images..."
docker compose -f $COMPOSE_FILE pull

# Stop old containers (graceful shutdown)
echo "[3/8] Stopping old containers..."
docker compose -f $COMPOSE_FILE down --timeout 60

# Backup database (if exists)
echo "[4/8] Backing up database..."
if docker volume ls | grep -q postgres-data; then
    BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR
    docker run --rm -v eee-pipeline_postgres-data:/data \
        -v $BACKUP_DIR:/backup alpine \
        tar czf /backup/postgres-data.tar.gz -C /data .
    echo "Backup saved to: $BACKUP_DIR"
fi

# Start new containers
echo "[5/8] Starting new containers..."
export VERSION=$VERSION
docker compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo "[6/8] Waiting for services to be healthy..."
sleep 30

# Health check
echo "[7/8] Running health checks..."
MAX_RETRIES=12
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "✓ Health check passed"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "✗ Health check failed after $MAX_RETRIES attempts"
        docker compose -f $COMPOSE_FILE logs --tail=100
        exit 1
    fi
    
    echo "  Waiting for services... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 10
done

# Smoke tests
echo "[8/8] Running smoke tests..."
SMOKE_TEST_RESULT=$(curl -s -X POST http://localhost:8000/v1/documents \
    -H "Content-Type: application/json" \
    -d '{"text": "Test deployment successful."}' | jq -r '.job_id')

if [ -n "$SMOKE_TEST_RESULT" ]; then
    echo "✓ Smoke test passed (job_id: $SMOKE_TEST_RESULT)"
else
    echo "✗ Smoke test failed"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ Deployment completed successfully!"
echo "========================================="
echo "Services:"
echo "  - API: https://api.yourdomain.com"
echo "  - Health: https://api.yourdomain.com/health"
echo "  - Docs: https://api.yourdomain.com/docs"
echo ""
echo "Monitoring:"
docker compose -f $COMPOSE_FILE ps
echo ""
echo "View logs:"
echo "  docker compose -f $COMPOSE_FILE logs -f"
```

**Make executable and run**:

```bash
chmod +x scripts/deploy-production.sh
./scripts/deploy-production.sh v1.0.0
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### Namespace and Secrets

```bash
# Create namespace
kubectl create namespace eee-pipeline

# Create secrets
kubectl create secret generic eee-secrets \
  --from-env-file=.env \
  --namespace=eee-pipeline

# Create image pull secret (if using private registry)
kubectl create secret docker-registry regcred \
  --docker-server=your-registry.com \
  --docker-username=your-username \
  --docker-password=your-password \
  --namespace=eee-pipeline
```

### ConfigMap

**File: `k8s/configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: eee-config
  namespace: eee-pipeline
data:
  settings.yaml: |
    general:
      environment: "production"
      log_level: "INFO"
      gpu_enabled: true
    
    orchestrator_service:
      host: "0.0.0.0"
      port: 8000
      request_timeout_seconds: 300
      batch_processing_chunk_size: 20
    
    event_llm_service:
      max_new_tokens: 8192
      temperature: 0.0
      generation_max_retries: 3
    
    celery:
      broker_url: "redis://redis-service:6379/0"
      result_backend: "redis://redis-service:6379/1"
      worker_concurrency: 8
```

```bash
kubectl apply -f k8s/configmap.yaml
```

### Persistent Volumes

**File: `k8s/storage.yaml`**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: eee-pipeline
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: results-pvc
  namespace: eee-pipeline
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: standard

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: eee-pipeline
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: eee-pipeline
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: fast-ssd
```

```bash
kubectl apply -f k8s/storage.yaml
```

### Redis Deployment

**File: `k8s/redis.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: eee-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
          - redis-server
          - --requirepass
          - $(REDIS_PASSWORD)
          - --maxmemory
          - 4gb
          - --maxmemory-policy
          - allkeys-lru
          - --appendonly
          - "yes"
        ports:
        - containerPort: 6379
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: eee-secrets
              key: REDIS_PASSWORD
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: eee-pipeline
spec:
  selector:
    app: redis
  ports:
  - protocol: TCP
    port: 6379
    targetPort: 6379
  clusterIP: None  # Headless service
```

```bash
kubectl apply -f k8s/redis.yaml
```

### PostgreSQL Deployment

**File: `k8s/postgres.yaml`**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: eee-pipeline
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: eee-secrets
              key: POSTGRES_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: eee-secrets
              key: POSTGRES_PASSWORD
        - name: POSTGRES_DB
          valueFrom:
            secretKeyRef:
              name: eee-secrets
              key: POSTGRES_DB
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - $(POSTGRES_USER)
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - $(POSTGRES_USER)
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: eee-pipeline
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
  clusterIP: None
```

```bash
kubectl apply -f k8s/postgres.yaml
```

### NER Service Deployment

**File: `k8s/ner-service.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ner-service
  namespace: eee-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ner-service
  template:
    metadata:
      labels:
        app: ner-service
    spec:
      imagePullSecrets:
      - name: regcred
      containers:
      - name: ner-service
        image: your-registry.com/eee-pipeline/ner:v1.0.0
        ports:
        - containerPort: 8001
        envFrom:
        - secretRef:
            name: eee-secrets
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: eee-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: ner-service
  namespace: eee-pipeline
spec:
  selector:
    app: ner-service
  ports:
  - protocol: TCP
    port: 8001
    targetPort: 8001
  type: ClusterIP
```

```bash
kubectl apply -f k8s/ner-service.yaml
```

### DP Service Deployment

**File: `k8s/dp-service.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dp-service
  namespace: eee-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dp-service
  template:
    metadata:
      labels:
        app: dp-service
    spec:
      imagePullSecrets:
      - name: regcred
      containers:
      - name: dp-service
        image: your-registry.com/eee-pipeline/dp:v1.0.0
        ports:
        - containerPort: 8002
        envFrom:
        - secretRef:
            name: eee-secrets
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: eee-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: dp-service
  namespace: eee-pipeline
spec:
  selector:
    app: dp-service
  ports:
  - protocol: TCP
    port: 8002
    targetPort: 8002
  type: ClusterIP
```

```bash
kubectl apply -f k8s/dp-service.yaml
```

### Event LLM Service Deployment (GPU)

**File: `k8s/event-llm-service.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: event-llm-service
  namespace: eee-pipeline
spec:
  replicas: 2  # Scale based on available GPUs
  selector:
    matchLabels:
      app: event-llm-service
  template:
    metadata:
      labels:
        app: event-llm-service
    spec:
      imagePullSecrets:
      - name: regcred
      nodeSelector:
        accelerator: nvidia-gpu  # Schedule on GPU nodes only
      containers:
      - name: event-llm-service
        image: your-registry.com/eee-pipeline/event-llm:v1.0.0
        ports:
        - containerPort: 8003
        envFrom:
        - secretRef:
            name: eee-secrets
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8003
          initialDelaySeconds: 120
          periodSeconds: 60
          timeoutSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8003
          initialDelaySeconds: 90
          periodSeconds: 30
          timeoutSeconds: 20
      volumes:
      - name: config
        configMap:
          name: eee-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: event-llm-service
  namespace: eee-pipeline
spec:
  selector:
    app: event-llm-service
  ports:
  - protocol: TCP
    port: 8003
    targetPort: 8003
  type: ClusterIP
```

```bash
kubectl apply -f k8s/event-llm-service.yaml
```

### Celery Worker Deployment

**File: `k8s/celery-worker.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
  namespace: eee-pipeline
spec:
  replicas: 5  # Scale based on workload
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      imagePullSecrets:
      - name: regcred
      containers:
      - name: celery-worker
        image: your-registry.com/eee-pipeline/celery:v1.0.0
        command:
          - celery
          - -A
          - src.core.celery_tasks
          - worker
          - --loglevel=info
          - --concurrency=8
          - --max-tasks-per-child=100
          - --prefetch-multiplier=2
        envFrom:
        - secretRef:
            name: eee-secrets
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: results
          mountPath: /data/results
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          exec:
            command:
            - celery
            - -A
            - src.core.celery_tasks
            - inspect
            - ping
          initialDelaySeconds: 60
          periodSeconds: 60
          timeoutSeconds: 30
      volumes:
      - name: config
        configMap:
          name: eee-config
      - name: results
        persistentVolumeClaim:
          claimName: results-pvc

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: celery-worker-hpa
  namespace: eee-pipeline
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: celery-worker
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

```bash
kubectl apply -f k8s/celery-worker.yaml
```

### Orchestrator Service Deployment

**File: `k8s/orchestrator-service.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator-service
  namespace: eee-pipeline
spec:
  replicas: 5
  selector:
    matchLabels:
      app: orchestrator-service
  template:
    metadata:
      labels:
        app: orchestrator-service
    spec:
      imagePullSecrets:
      - name: regcred
      containers:
      - name: orchestrator-service
        image: your-registry.com/eee-pipeline/orchestrator:v1.0.0
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: eee-secrets
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: eee-config
      - name: logs
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: orchestrator-service
  namespace: eee-pipeline
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  selector:
    app: orchestrator-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: orchestrator-hpa
  namespace: eee-pipeline
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: orchestrator-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

```bash
kubectl apply -f k8s/orchestrator-service.yaml
```

### Ingress Configuration

**File: `k8s/ingress.yaml`**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: eee-ingress
  namespace: eee-pipeline
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: eee-tls-secret
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: orchestrator-service
            port:
              number: 80
```

```bash
kubectl apply -f k8s/ingress.yaml
```

### Complete Deployment Script

**File: `scripts/deploy-k8s.sh`**

```bash
#!/bin/bash
set -e

NAMESPACE="eee-pipeline"
VERSION="${1:-v1.0.0}"

echo "========================================="
echo "EEE Pipeline Kubernetes Deployment"
echo "========================================="
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"
echo ""

# Check prerequisites
echo "[1/10] Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { echo "Error: kubectl not installed"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "Error: helm not installed"; exit 1; }

# Verify cluster access
kubectl cluster-info >/dev/null 2>&1 || { echo "Error: Cannot access cluster"; exit 1; }

# Create namespace
echo "[2/10] Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Create secrets
echo "[3/10] Creating secrets..."
if [ -f ".env" ]; then
    kubectl create secret generic eee-secrets \
        --from-env-file=.env \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
else
    echo "Warning: .env file not found, skipping secrets"
fi

# Apply storage
echo "[4/10] Creating persistent volumes..."
kubectl apply -f k8s/storage.yaml

# Apply ConfigMap
echo "[5/10] Applying configuration..."
kubectl apply -f k8s/configmap.yaml

# Deploy infrastructure services
echo "[6/10] Deploying infrastructure services..."
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/postgres.yaml

# Wait for infrastructure
echo "[7/10] Waiting for infrastructure to be ready..."
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s

# Deploy application services
echo "[8/10] Deploying application services..."
kubectl apply -f k8s/ner-service.yaml
kubectl apply -f k8s/dp-service.yaml
kubectl apply -f k8s/event-llm-service.yaml
kubectl apply -f k8s/celery-worker.yaml
kubectl apply -f k8s/orchestrator-service.yaml

# Wait for services
echo "[9/10] Waiting for services to be ready..."
kubectl wait --for=condition=ready pod -l app=ner-service -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=dp-service -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=event-llm-service -n $NAMESPACE --timeout=600s
kubectl wait --for=condition=ready pod -l app=orchestrator-service -n $NAMESPACE --timeout=300s

# Apply ingress
echo "[10/10] Configuring ingress..."
kubectl apply -f k8s/ingress.yaml

echo ""
echo "========================================="
echo "✓ Deployment completed successfully!"
echo "========================================="

# Get service info
echo ""
echo "Services:"
kubectl get pods -n $NAMESPACE
echo ""
echo "Ingress:"
kubectl get ingress -n $NAMESPACE
echo ""
echo "LoadBalancer:"
kubectl get svc orchestrator-service -n $NAMESPACE

echo ""
echo "Next steps:"
echo "1. Update DNS to point to LoadBalancer IP"
echo "2. Monitor pods: kubectl logs -f <pod-name> -n $NAMESPACE"
echo "3. Test API: curl https://api.yourdomain.com/health"
```

**Make executable and run**:

```bash
chmod +x scripts/deploy-k8s.sh
./scripts/deploy-k8s.sh v1.0.0
```

---

## Cloud Platform Deployments

### AWS EKS Deployment

**Prerequisites**:

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Install AWS CLI
pip install awscli
aws configure
```

**Create EKS Cluster**:

**File: `aws/eks-cluster.yaml`**

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: eee-pipeline-cluster
  region: us-west-2
  version: "1.28"

vpc:
  cidr: 10.0.0.0/16
  nat:
    gateway: HighlyAvailable

iam:
  withOIDC: true

managedNodeGroups:
  - name: cpu-workers
    instanceType: m5.2xlarge
    minSize: 3
    maxSize: 10
    desiredCapacity: 5
    volumeSize: 100
    privateNetworking: true
    labels:
      workload: cpu
    tags:
      nodegroup-role: cpu-workers

  - name: gpu-workers
    instanceType: g4dn.xlarge  # NVIDIA T4 GPU
    minSize: 1
    maxSize: 5
    desiredCapacity: 2
    volumeSize: 200
    privateNetworking: true
    labels:
      workload: gpu
      accelerator: nvidia-gpu
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    tags:
      nodegroup-role: gpu-workers

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

cloudWatch:
  clusterLogging:
    enableTypes: ["all"]
```

**Deploy EKS Cluster**:

```bash
# Create cluster (takes 15-20 minutes)
eksctl create cluster -f aws/eks-cluster.yaml

# Verify
kubectl get nodes
kubectl get nodes -l accelerator=nvidia-gpu
```

**Install NVIDIA Device Plugin**:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

**Install EBS CSI Driver**:

```bash
# Create IAM policy
aws iam create-policy \
  --policy-name AmazonEKS_EBS_CSI_Driver_Policy \
  --policy-document file://aws/ebs-csi-policy.json

# Attach to node role
NODE_ROLE=$(aws eks describe-nodegroup \
  --cluster-name eee-pipeline-cluster \
  --nodegroup-name cpu-workers \
  --query "nodegroup.nodeRole" --output text)

aws iam attach-role-policy \
  --role-name ${NODE_ROLE##*/} \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy
```

**Deploy Application**:

```bash
# Deploy to EKS
./scripts/deploy-k8s.sh v1.0.0

# Get LoadBalancer URL
kubectl get svc orchestrator-service -n eee-pipeline -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

**Auto-scaling Configuration**:

```bash
# Install cluster autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Configure for your cluster
kubectl -n kube-system edit deployment cluster-autoscaler

# Add these flags:
#   --balance-similar-node-groups
#   --skip-nodes-with-system-pods=false
#   --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/eee-pipeline-cluster
```

**Cost Optimization**:

```yaml
# Use Spot Instances for Celery workers
# File: aws/eks-spot-nodegroup.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: eee-pipeline-cluster
  region: us-west-2

managedNodeGroups:
  - name: celery-spot
    instanceTypes: ["m5.2xlarge", "m5a.2xlarge", "m5n.2xlarge"]
    spot: true
    minSize: 2
    maxSize: 20
    desiredCapacity: 5
    labels:
      workload: celery
    taints:
      - key: workload
        value: celery
        effect: NoSchedule
```

```bash
eksctl create nodegroup -f aws/eks-spot-nodegroup.yaml
```

### Google Cloud GKE Deployment

**Prerequisites**:

```bash
# Install gcloud SDK
curl https://sdk.cloud.google.com | bash
gcloud init

# Install gke-gcloud-auth-plugin
gcloud components install gke-gcloud-auth-plugin
```

**Create GKE Cluster**:

```bash
# Enable APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com

# Create cluster with GPU node pool
gcloud container clusters create eee-pipeline-cluster \
  --region us-central1 \
  --num-nodes 3 \
  --machine-type n1-standard-8 \
  --disk-size 100 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --enable-stackdriver-kubernetes \
  --addons HorizontalPodAutoscaling,HttpLoadBalancing

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster eee-pipeline-cluster \
  --region us-central1 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-8 \
  --num-nodes 2 \
  --min-nodes 1 \
  --max-nodes 5 \
  --enable-autoscaling \
  --disk-size 200

# Get credentials
gcloud container clusters get-credentials eee-pipeline-cluster --region us-central1
```

**Install NVIDIA Drivers**:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

**Deploy Application**:

```bash
./scripts/deploy-k8s.sh v1.0.0
```

**Setup Cloud Load Balancer**:

```bash
# Get external IP
kubectl get svc orchestrator-service -n eee-pipeline

# Reserve static IP
gcloud compute addresses create eee-api-ip --region us-central1

# Update service
kubectl patch svc orchestrator-service -n eee-pipeline -p '{"spec":{"loadBalancerIP":"<STATIC_IP>"}}'
```

### Azure AKS Deployment

**Prerequisites**:

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login
```

**Create AKS Cluster**:

```bash
# Create resource group
az group create --name eee-pipeline-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group eee-pipeline-rg \
  --name eee-pipeline-cluster \
  --node-count 3 \
  --node-vm-size Standard_D8s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10 \
  --generate-ssh-keys \
  --enable-managed-identity \
  --network-plugin azure

# Add GPU node pool
az aks nodepool add \
  --resource-group eee-pipeline-rg \
  --cluster-name eee-pipeline-cluster \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --min-count 1 \
  --max-count 5 \
  --enable-cluster-autoscaler

# Get credentials
az aks get-credentials \
  --resource-group eee-pipeline-rg \
  --name eee-pipeline-cluster
```

**Install NVIDIA Device Plugin**:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

**Deploy Application**:

```bash
./scripts/deploy-k8s.sh v1.0.0
```

---

## GPU Configuration

### NVIDIA Container Toolkit Setup

**Ubuntu/Debian**:

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Configure docker-compose for GPU**:

```yaml
services:
  event-llm-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
```

### Multi-GPU Configuration

**Round-robin GPU allocation**:

```yaml
# docker-compose.multi-gpu.yml
services:
  event-llm-service-1:
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  event-llm-service-2:
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
```

**Load balancer for multi-GPU**:

```nginx
upstream llm_backend {
    least_conn;
    server event-llm-service-1:8003;
    server event-llm-service-2:8003;
}

server {
    location /generate-events {
        proxy_pass http://llm_backend;
    }
}
```

### GPU Monitoring

**Install nvidia_gpu_exporter**:

```bash
docker run -d \
  --name gpu-exporter \
  --gpus all \
  -p 9835:9835 \
  nvidia/dcgm-exporter:latest
```

**Prometheus scrape config**:

```yaml
scrape_configs:
  - job_name: 'gpu-metrics'
    static_configs:
      - targets: ['gpu-exporter:9835']
```

---

## Monitoring and Observability

### Prometheus Setup

**File: `monitoring/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'orchestrator'
    static_configs:
      - targets: ['orchestrator-service:8000']
    metrics_path: '/metrics'

  - job_name: 'ner-service'
    static_configs:
      - targets: ['ner-service:8001']

  - job_name: 'dp-service'
    static_configs:
      - targets: ['dp-service:8002']

  - job_name: 'event-llm-service'
    static_configs:
      - targets: ['event-llm-service:8003']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['gpu-exporter:9835']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

**Deploy Prometheus**:

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### Grafana Dashboards

**Deploy Grafana**:

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana
```

**Access**: <http://localhost:3000> (admin/admin)

**Import Dashboards**:

1. Go to Dashboards → Import
2. Use dashboard IDs:
   - **Node Exporter**: 1860
   - **Redis**: 11835
   - **PostgreSQL**: 9628
   - **NVIDIA GPU**: 12239

**Custom EEE Dashboard** (`monitoring/eee-dashboard.json`):

```json
{
  "dashboard": {
    "title": "EEE Pipeline Metrics",
    "panels": [
      {
        "title": "API Requests/sec",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Event Extraction Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_generation_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL"
          }
        ]
      },
      {
        "title": "Celery Queue Length",
        "targets": [
          {
            "expr": "celery_queue_length"
          }
        ]
      }
    ]
  }
}
```

### Logging Stack (ELK)

**File: `monitoring/docker-compose.elk.yml`**

```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es-data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  es-data:
```

**File: `monitoring/logstash.conf`**

```
input {
  tcp {
    port => 5000
    codec => json
  }
}

filter {
  if [level] == "ERROR" {
    mutate {
      add_tag => ["error"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "eee-logs-%{+YYYY.MM.dd}"
  }
}
```

**Configure services to send logs**:

```yaml
# Add to each service in docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
    labels: "service"
```

---

## Backup and Recovery

### Database Backup

**Automated PostgreSQL Backup Script**:

**File: `scripts/backup-postgres.sh`**

```bash
#!/bin/bash
set -e

BACKUP_DIR="/backups/postgres"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.sql.gz"

echo "Starting PostgreSQL backup..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
docker compose exec -T postgres pg_dumpall -U ${POSTGRES_USER} | gzip > $BACKUP_FILE

# Verify backup
if [ $? -eq 0 ]; then
echo "✓ Backup successful: $BACKUP_FILE"
