#!/bin/sh
# Start nginx and Next.js; exit (and let Docker restart the container)
# if either process dies unexpectedly.
set -e

node server.js &
NODE_PID=$!

nginx -g 'daemon off;' &
NGINX_PID=$!

while true; do
    if ! kill -0 "$NODE_PID" 2>/dev/null; then
        echo "node server.js exited unexpectedly" >&2
        kill "$NGINX_PID" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "$NGINX_PID" 2>/dev/null; then
        echo "nginx exited unexpectedly" >&2
        kill "$NODE_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done
