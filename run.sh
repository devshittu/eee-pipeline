#!/bin/bash

# === CONFIGURATION ===
SCRIPT_NAME="run.sh"
COMPOSE_FILE="docker-compose.yml"
LOGS_DIR="./logs"

# === ANSI COLORS ===
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# === HELP TEXT ===
show_help() {
  echo -e "${BLUE}Usage: ./$SCRIPT_NAME <command> [options]${NC}"
  echo ""
  echo -e "${YELLOW}Commands:${NC}"
  echo "  start [services]       Start services (default: all)"
  echo "  rebuild [services]     Rebuild services (default: all)"
  echo "  rebuild-no-cache [services]  Rebuild with --no-cache"
  echo "  clean                  Clean logs and stop containers"
  echo "  down                   Stop and remove containers"
  echo "  status                 Show container statuses"
  echo "  logs [service]         Show logs for a service"
  echo "  help                   Show this help"
  echo ""
  echo -e "${YELLOW}Examples:${NC}"
  echo "  Start all services:          ./$SCRIPT_NAME start"
  echo "  Rebuild NER service:         ./$SCRIPT_NAME rebuild ner-service"
  echo "  Clean and rebuild all:       ./$SCRIPT_NAME clean && ./$SCRIPT_NAME rebuild"
  echo "  Rebuild LLM with no cache:   ./$SCRIPT_NAME rebuild-no-cache event-llm-service"
  echo "  Check service logs:          ./$SCRIPT_NAME logs orchestrator-service"
  echo ""
  exit 0
}

# === UTILITY FUNCTIONS ===
clean_logs() {
  echo -e "${YELLOW}[*] Cleaning logs...${NC}"
  sudo rm -rf "$LOGS_DIR"/* 2>/dev/null || true
}

prune_builder() {
  echo -e "${YELLOW}[*] Pruning Docker builder...${NC}"
  docker builder prune -f
}

compose_down() {
  echo -e "${YELLOW}[*] Stopping containers...${NC}"
  docker compose -f "$COMPOSE_FILE" down -v
}

rebuild_service() {
  local service="$1"
  local no_cache="$2"
  if [[ "$no_cache" == "true" ]]; then
    echo -e "${GREEN}[+] Rebuilding $service with --no-cache${NC}"
    docker compose -f "$COMPOSE_FILE" build --no-cache "$service"
  else
    echo -e "${GREEN}[+] Rebuilding $service${NC}"
    docker compose -f "$COMPOSE_FILE" build "$service"
  fi
}

start_services() {
  local services=("$@")
  if [[ "${services[*]}" == "all" || ${#services[@]} -eq 0 ]]; then
    echo -e "${GREEN}[+] Starting all services...${NC}"
    docker compose -f "$COMPOSE_FILE" up -d
  else
    echo -e "${GREEN}[+] Starting services: ${services[*]}${NC}"
    docker compose -f "$COMPOSE_FILE" up -d "${services[@]}"
  fi
}

# === MAIN LOGIC ===
case "$1" in
  start)
    shift
    start_services "$@"
    ;;
  
  rebuild)
    shift
    prune_builder
    if [[ -z "$1" ]]; then
      docker compose -f "$COMPOSE_FILE" build
    else
      while [[ "$1" ]]; do
        rebuild_service "$1" "false"
        shift
      done
    fi
    ;;
  
  rebuild-no-cache)
    shift
    prune_builder
    if [[ -z "$1" ]]; then
      docker compose -f "$COMPOSE_FILE" build --no-cache
    else
      while [[ "$1" ]]; do
        rebuild_service "$1" "true"
        shift
      done
    fi
    ;;
  
  clean)
    clean_logs
    compose_down
    ;;
  
  down)
    compose_down
    ;;
  
  status)
    echo -e "${BLUE}[*] Container Status:${NC}"
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  
  logs)
    shift
    if [[ -z "$1" ]]; then
      echo -e "${RED}[-] Please specify a service name!${NC}"
      exit 1
    fi
    docker compose -f "$COMPOSE_FILE" logs -f "$1"
    ;;
  
  help|*)
    show_help
    ;;
esac


# ./run.sh clean
# ./run.sh rebuild
# ./run.sh start