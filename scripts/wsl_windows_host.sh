#!/usr/bin/env bash
# Print the Windows host IP as seen from WSL2 NAT networking.
#
# Use this as the IBKR host when TWS / IB Gateway runs on the Windows host and
# WSL is in its default NAT networking mode:
#
#   export FUTURESRLX_IBKR_HOST="$(./scripts/wsl_windows_host.sh)"
#   .venv/bin/python -m src.execution.runner --strategy configs/strategies/tsmom_vol_target.yaml
#
# With WSL mirrored networking ([wsl2] networkingMode=mirrored in %USERPROFILE%\.wslconfig)
# the Windows loopback is shared, so FUTURESRLX_IBKR_HOST=127.0.0.1 works instead.
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "error: this helper is for WSL; use 127.0.0.1 on a native Linux/Windows host" >&2
  exit 1
fi

host="$(ip route show default 2>/dev/null | awk 'NR==1 {print $3}')"
if [[ -z "${host}" ]]; then
  echo "error: could not determine the Windows host IP (no default route)" >&2
  exit 1
fi
printf '%s\n' "${host}"
