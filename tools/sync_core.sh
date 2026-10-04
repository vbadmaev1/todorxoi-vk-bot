#!/usr/bin/env bash
# Обновить ядро из Telegram-версии бота.
#
# core/, model/ и assets/ — копия из репозитория todorxoi-bot: вся
# лингвистика (транслитерация, тодо бичиг, рендер, распознавание фото)
# живёт там, здесь только VK-слой. После правок ядра в todorxoi-bot:
#
#   tools/sync_core.sh ../todorxoi-bot            # с текущего коммита (HEAD)
#   tools/sync_core.sh ../todorxoi-bot main       # с ветки или коммита
#   python -m tests.smoke                          # проверить, что всё живо
#
# Берутся только закоммиченные файлы (git archive), поэтому
# незакоммиченные эксперименты в todorxoi-bot сюда не попадут. Откуда и
# с какого коммита взято ядро — в CORE_VERSION.

set -euo pipefail

SRC="${1:?путь к репозиторию todorxoi-bot}"
REF="${2:-HEAD}"
DST="$(cd "$(dirname "$0")/.." && pwd)"

COMMIT="$(git -C "$SRC" rev-parse "$REF")"
rm -rf "$DST/core" "$DST/model" "$DST/assets"
git -C "$SRC" archive "$COMMIT" core model assets | tar -x -C "$DST"

{
  echo "# Ядро скопировано из todorxoi-bot — tools/sync_core.sh"
  echo "commit $COMMIT"
  echo "subject $(git -C "$SRC" log -1 --format=%s "$COMMIT")"
  echo "date $(git -C "$SRC" log -1 --format=%cs "$COMMIT")"
} > "$DST/CORE_VERSION"

echo "ядро обновлено до $COMMIT"
git -C "$DST" status --short -- core model assets CORE_VERSION | head -20
