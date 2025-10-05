#!/bin/bash

# === CONFIGURATION ===
SCRIPT_NAME="run.sh"
COMPOSE_PROD="docker-compose.yml"
COMPOSE_DEV="docker-compose.dev.yml"
LOGS_DIR="./logs"

# === ANSI COLORS ===
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# === HELP TEXT ===
show_help() {
  echo -e "${BLUE}Usage: ./$SCRIPT_NAME <command> [options] [dev|prod]${NC}"
  echo ""
  echo -e "${YELLOW}Commands:${NC}"
  echo "  start [services]       Start services (default: all)"
  echo "  rebuild [services]     Rebuild services (default: all)"
  echo "  rebuild-no-cache [services]  Rebuild with --no-cache"
  echo "  clean                  Clean logs and stop containers"
  echo "  down                   Stop and remove containers"
  echo "  status                 Show container statuses"
  echo "  logs [service]         Show logs for a service"
  echo "  cli [args]             Run CLI commands inside orchestrator container"
  echo "  help                   Show this help"
  echo ""
  echo -e "${YELLOW}Modes:${NC}"
  echo "  prod (default)         Use production config"
  echo "  dev                    Use development config (hot-reloading)"
  echo ""
  echo -e "${YELLOW}Examples:${NC}"
  echo "  Start all services (prod):          ./$SCRIPT_NAME start"
  echo "  Start all services (dev):           ./$SCRIPT_NAME start dev"
  echo "  Rebuild NER service (dev):          ./$SCRIPT_NAME rebuild ner-service dev"
  echo "  Clean and rebuild all (prod):       ./$SCRIPT_NAME clean && ./$SCRIPT_NAME rebuild"
  echo "  Check service logs:                 ./$SCRIPT_NAME logs orchestrator-service"
  echo ""
  echo -e "${YELLOW}CLI Examples (via docker exec):${NC}"
  echo "  Process single document:            ./$SCRIPT_NAME cli documents process 'Text here'"
  echo "  Submit batch job:                   ./$SCRIPT_NAME cli documents batch /app/data/input.jsonl -o /app/data/output.jsonl"
  echo "  Check job status:                   ./$SCRIPT_NAME cli jobs status <job-id>"
  echo "  Health check:                       ./$SCRIPT_NAME cli admin health"
  echo ""
  exit 0
}

# === UTILITY FUNCTIONS ===
get_compose_files() {
  local mode="$1"
  if [[ "$mode" == "dev" ]]; then
    echo "-f $COMPOSE_PROD -f $COMPOSE_DEV"
  else
    echo "-f $COMPOSE_PROD"
  fi
}

clean_logs() {
  echo -e "${YELLOW}[*] Cleaning logs...${NC}"
  sudo rm -rf "$LOGS_DIR"/* 2>/dev/null || true
}

prune_builder() {
  echo -e "${YELLOW}[*] Pruning Docker builder...${NC}"
  docker builder prune -f
}

compose_down() {
  local compose_files=$(get_compose_files "$mode")
  echo -e "${YELLOW}[*] Stopping containers...${NC}"
  docker compose $compose_files down -v
}

rebuild_service() {
  local service="$1"
  local no_cache="$2"
  local compose_files=$(get_compose_files "$mode")
  if [[ "$no_cache" == "true" ]]; then
    echo -e "${GREEN}[+] Rebuilding $service with --no-cache${NC}"
    docker compose $compose_files build --no-cache "$service"
  else
    echo -e "${GREEN}[+] Rebuilding $service${NC}"
    docker compose $compose_files build "$service"
  fi
}

start_services() {
  local services=("$@")
  local compose_files=$(get_compose_files "$mode")
  if [[ "${services[*]}" == "all" || ${#services[@]} -eq 0 ]]; then
    echo -e "${GREEN}[+] Starting all services...${NC}"
    docker compose $compose_files up -d
  else
    echo -e "${GREEN}[+] Starting services: ${services[*]}${NC}"
    docker compose $compose_files up -d "${services[@]}"
  fi
}

run_cli() {
  local args=("$@")
  echo -e "${BLUE}[*] Executing CLI command in orchestrator container...${NC}"
  docker exec -it orchestrator-service python -m src.cli.main "${args[@]}"
}

# === MAIN LOGIC ===
if [[ $# -eq 0 ]]; then
  show_help
fi

command="$1"
mode="prod"
if [[ "$2" == "dev" ]]; then
  mode="dev"
  shift 2
else
  shift 1
fi

case "$command" in
  start)
    start_services "$@"
    ;;
  
  rebuild)
    prune_builder
    if [[ -z "$1" ]]; then
      local compose_files=$(get_compose_files "$mode")
      docker compose $compose_files build
    else
      while [[ "$1" ]]; do
        rebuild_service "$1" "false"
        shift
      done
    fi
    ;;
  
  rebuild-no-cache)
    prune_builder
    if [[ -z "$1" ]]; then
      local compose_files=$(get_compose_files "$mode")
      docker compose $compose_files build --no-cache
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
    local compose_files=$(get_compose_files "$mode")
    echo -e "${BLUE}[*] Container Status:${NC}"
    docker compose $compose_files ps
    ;;
  
  logs)
    if [[ -z "$1" ]]; then
      echo -e "${RED}[-] Please specify a service name!${NC}"
      exit 1
    fi
    local compose_files=$(get_compose_files "$mode")
    docker compose $compose_files logs -f "$1"
    ;;
  
  cli)
    run_cli "$@"
    ;;
  
  help|*)
    show_help
    ;;
esac

# run.sh
# File path: run.sh