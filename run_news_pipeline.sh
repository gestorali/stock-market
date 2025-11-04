#!/bin/bash
# === run_news_pipeline.sh ===
# Automates: (optional Docker) → wait for LibreTranslate → fetch news → process news → optional stop Docker
# Logs everything to logs/pipeline_YYYYMMDD_HHMMSS.log

set -e  # Stop on first error

# === Configuration ===
PROJECT_DIR="/home/michal/PycharmProjects/stock-market"          # ⚠️ CHANGE to your actual project path
LT_DIR=~/libretranslate-offline
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"
LT_URL="http://localhost:5000/languages"

# === Change working directory immediately ===
cd "$PROJECT_DIR"

# === Prepare log directory ===
mkdir -p "$LOG_DIR"

# === Log setup ===
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🕒 Starting full news pipeline run at $TIMESTAMP"
echo "📂 Logs saved to: $LOG_FILE"
echo "--------------------------------------------"

# === Function: check if LibreTranslate is running ===
is_libretranslate_running() {
    curl -s --max-time 2 "$LT_URL" > /dev/null
}

# === Detect LibreTranslate status ===
echo "🔍 Checking if LibreTranslate is already running..."
if is_libretranslate_running; then
    echo "✅ LibreTranslate is already running on port 5000."
    LT_ALREADY_RUNNING=true
else
    echo "❌ LibreTranslate not detected — will start Docker."
    LT_ALREADY_RUNNING=false
fi

# 1️⃣ Start Docker (if LibreTranslate not running)
if [ "$LT_ALREADY_RUNNING" = false ]; then
    echo "🚀 Starting LibreTranslate via Docker..."
    cd "$LT_DIR"
    docker compose up -d --build
    STARTED_DOCKER=true
else
    STARTED_DOCKER=false
fi

# 2️⃣ Wait for LibreTranslate to be ready
echo "⏳ Waiting for LibreTranslate to become available..."
PYTHONPATH="$PROJECT_DIR" python3 "$PROJECT_DIR/src/utils/wait_for_libretranslate.py"


# 3️⃣ Run fetch_news.py
#echo "📰 Starting fetch_news.py ..."
#cd "$PROJECT_DIR"
#python3 src/data/fetch_news.py

# 4️⃣ Run process_news.py
echo "🧠 Starting process_news.py ..."
PYTHONPATH="$PROJECT_DIR" python3 "$PROJECT_DIR/src/data/process_news.py"

# 5️⃣ Stop Docker containers (only if we started them)
if [ "$STARTED_DOCKER" = true ]; then
    echo "🛑 Stopping LibreTranslate containers..."
    cd "$LT_DIR"
    docker compose down
else
    echo "⚙️  LibreTranslate was already running — not stopping Docker."
fi

echo "✅ Pipeline completed successfully at $(date)"
echo "--------------------------------------------"
echo "📜 Full log available at: $LOG_FILE"
