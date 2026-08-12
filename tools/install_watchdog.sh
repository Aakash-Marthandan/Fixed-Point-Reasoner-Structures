#!/bin/bash
# Installs the TPU watchdog as a user launchd agent (no sudo): every 15 min,
# survives reboots/app restarts. Idempotent — reinstall to update.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.qhrrn2.tpuwatchdog.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.qhrrn2.tpuwatchdog</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$REPO/tools/tpu_watchdog.sh</string></array>
  <key>StartInterval</key><integer>900</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/tmp/qhrrn2_watchdog.out</string>
  <key>StandardErrorPath</key><string>/tmp/qhrrn2_watchdog.err</string>
</dict></plist>
EOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "watchdog installed: $(launchctl list | grep qhrrn2 || echo 'NOT LOADED')"
