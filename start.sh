#!/usr/bin/env bash
# findBeeQueen をワンコマンドで起動する。
#   スマホの配信アプリ（Android: IP Webcam / iPhone・Android: DroidCam）を LAN から自動で探し、
#   見つかったら女王蜂検出を始める。見つかるまで 5 秒ごとに探し続ける。
#
# 使い方:
#   ./start.sh                     # 自動検索
#   ./start.sh 192.168.0.23        # IP を指定（アプリの種類は自動判定）
#   ./start.sh --tiles 2           # detect_stream.py のオプションはそのまま渡せる
#   WEB_PORT=0 ./start.sh          # ブラウザ配信を無効化（既定は http://localhost:8090/）
set -u
cd "$(dirname "$(readlink -f "$0")")"

if [ ! -x .venv/bin/python ]; then
    echo "[起動] .venv がありません。README のセットアップを先に行ってください。" >&2
    exit 1
fi
PY=.venv/bin/python

ip_arg=""
if [[ $# -gt 0 && $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    ip_arg=$1
    shift
fi
WEB_PORT=${WEB_PORT:-8090}

while true; do
    if [ -n "$ip_arg" ]; then
        found=$($PY tools/find_phone.py --ip "$ip_arg")
    else
        found=$($PY tools/find_phone.py)
    fi
    [ -n "$found" ] && break
    echo "[起動] スマホが見つかりません。同じ Wi-Fi でアプリの配信を開始してください（5 秒後に再検索、Ctrl+C で終了）" >&2
    sleep 5
done

read -r app ip <<< "$found"
echo "[起動] $app を検出: $ip"
web_opt=()
if [ "$WEB_PORT" != "0" ]; then
    web_opt=(--web "$WEB_PORT")
    echo "[起動] ブラウザでも見られます: http://localhost:$WEB_PORT/"
fi
exec $PY detect_stream.py --app "$app" --source "$ip" "${web_opt[@]}" "$@"
