#!/usr/bin/env bash
# Sync chexit to the droplet at /opt/chexit and restart the API.
#
#   bash scripts/deploy-to-droplet.sh
#   bash scripts/deploy-to-droplet.sh --code-only
#   bash scripts/deploy-to-droplet.sh --assets-only
#
set -euo pipefail

DROPLET="${CHEXIT_DROPLET:-root@143.198.99.111}"
REMOTE="/opt/chexit"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE=all
for arg in "$@"; do
  case "$arg" in
    --assets-only) MODE=assets ;;
    --code-only) MODE=code ;;
    -h|--help)
      sed -n '2,7p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

RSYNC_EXCLUDES=(
  --exclude node_modules
  --exclude .git
  --exclude chexit-backend/.venv
  --exclude chexit-backend/chexit-backend
  --exclude dist
  --exclude .env
  --exclude .env.local
)

echo "Target: $DROPLET:$REMOTE (mode=$MODE)"
echo ""

if [[ "$MODE" == all || "$MODE" == code ]]; then
  echo "==> Syncing application code..."
  rsync -avz --progress "${RSYNC_EXCLUDES[@]}" \
    --exclude assets \
    "$REPO_ROOT/" "$DROPLET:$REMOTE/"
fi

if [[ "$MODE" == all || "$MODE" == assets ]]; then
  echo "==> Syncing assets/ (~1GB)..."
  rsync -avz --progress \
    "$REPO_ROOT/assets/" "$DROPLET:$REMOTE/assets/"
fi

if [[ "$MODE" == assets ]]; then
  echo "Assets sync done."
  exit 0
fi

echo "==> Restarting API on droplet..."
ssh "$DROPLET" bash -s <<EOF
set -euo pipefail
REMOTE="$REMOTE"
mkdir -p /var/chexit/uploads
cd "\$REMOTE/chexit-backend"
if [[ ! -d .venv ]]; then
  python3.11 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
install -m 644 "\$REMOTE/deploy/chexit-api.service" /etc/systemd/system/chexit-api.service
systemctl daemon-reload
systemctl enable chexit-api
systemctl restart chexit-api
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 3
done
systemctl is-active --quiet chexit-api
curl -sf http://127.0.0.1:8000/health
echo ""
EOF

echo "Deploy complete. API: https://api.chexit.app"
