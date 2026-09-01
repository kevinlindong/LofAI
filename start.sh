#!/bin/bash

# Foreground supervisor for the complete lofAI application. Each service runs
# in its own session/process group, and this supervisor keeps an exclusive lock
# until every descendant has been verified stopped.

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
STATE_DIR="${LOFAI_STATE_DIR:-$SCRIPT_DIR}"
LOCK_FILE="$STATE_DIR/.lofai.lock"
PYTHON_LAUNCHER="$(command -v python3 || true)"
BACKEND_SCRIPT="${LOFAI_BACKEND_SCRIPT:-$SCRIPT_DIR/start-backend.sh}"
FRONTEND_SCRIPT="${LOFAI_FRONTEND_SCRIPT:-$SCRIPT_DIR/start-frontend.sh}"
STARTUP_DELAY="${LOFAI_STARTUP_DELAY:-3}"
TAIL_PID=""
STARTUP_WAIT_PID=""
BACKEND_PID=""
FRONTEND_PID=""
LOCK_RECORD=""
LOG_RECORD=""
PROVISIONAL_PIDS=()

cd "$SCRIPT_DIR"
if ! mkdir -p "$STATE_DIR"; then
    echo "Error: could not create lofAI state directory: $STATE_DIR" >&2
    exit 1
fi

if [ -z "$PYTHON_LAUNCHER" ]; then
    echo "Error: python3 is required to launch isolated service groups."
    exit 1
fi

if ! [[ "$STARTUP_DELAY" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: LOFAI_STARTUP_DELAY must be a non-negative number."
    exit 1
fi

if [ ! -x "$BACKEND_SCRIPT" ] || [ ! -x "$FRONTEND_SCRIPT" ]; then
    echo "Error: backend and frontend launch scripts must be executable."
    exit 1
fi

process_started_at() {
    ps -o lstart= -p "$1" 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

process_group_of() {
    ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]'
}

pid_is_live() {
    ps -o state= -p "$1" 2>/dev/null | awk 'NF && $1 !~ /^Z/ { found=1 } END { exit !found }'
}

metadata_record() {
    local mode="$1"
    local pid="$2"
    local started
    started="$(process_started_at "$pid")"
    [ -n "$started" ] || return 1
    printf '%s|%s|%s|%s' "$mode" "$pid" "$started" "$SCRIPT_DIR"
}

remove_if_unchanged() {
    local file="$1"
    local expected="$2"
    local current=""
    [ -f "$file" ] || return 0
    current="$(< "$file")"
    if [ "$current" = "$expected" ]; then
        rm -f "$file"
    fi
}

write_pid_file() {
    local file="$1"
    local mode="$2"
    local pid="$3"
    local record
    local temporary="${file}.tmp.$$"

    if ! record="$(metadata_record "$mode" "$pid")"; then
        echo "Error: process $pid exited before it could be tracked." >&2
        return 1
    fi
    if ! (umask 077; printf '%s\n' "$record" > "$temporary"); then
        rm -f "$temporary"
        return 1
    fi
    if ! mv "$temporary" "$file"; then
        rm -f "$temporary"
        return 1
    fi
    if [ "$file" = "$STATE_DIR/logs.pid" ]; then
        LOG_RECORD="$record"
    fi
}

acquire_lock() {
    local record
    if ! record="$(metadata_record pid "$$")"; then
        return 1
    fi
    if (set -o noclobber; umask 077; printf '%s\n' "$record" > "$LOCK_FILE") 2>/dev/null; then
        LOCK_RECORD="$record"
        return 0
    fi
    return 1
}

wait_for_isolated_group() {
    local pid="$1"
    local attempt=0
    local group
    while pid_is_live "$pid"; do
        group="$(process_group_of "$pid")"
        if [ "$group" = "$pid" ]; then
            return 0
        fi
        attempt=$((attempt + 1))
        [ "$attempt" -ge 200 ] && return 1
        sleep 0.01
    done
    return 1
}

signal_provisional() {
    local signal="$1"
    local pid="$2"
    local group
    pid_is_live "$pid" || return 0
    group="$(process_group_of "$pid")"
    if [ "$group" = "$pid" ]; then
        kill -"$signal" -- "-$pid" 2>/dev/null || true
    else
        kill -"$signal" "$pid" 2>/dev/null || true
    fi
}

launch_group() {
    local script="$1"
    local log_file="$2"
    local pid_file="$3"

    "$PYTHON_LAUNCHER" -c \
        'import os, sys; os.setsid(); os.execv(sys.argv[1], [sys.argv[1]])' \
        "$script" > "$log_file" 2>&1 &
    LAUNCHED_PID=$!
    PROVISIONAL_PIDS+=("$LAUNCHED_PID")

    # Publish group metadata only after setsid has made PID == PGID. Until then
    # the supervisor lock plus PROVISIONAL_PIDS cover an interrupt safely.
    if ! wait_for_isolated_group "$LAUNCHED_PID" || \
       ! write_pid_file "$pid_file" group "$LAUNCHED_PID"; then
        signal_provisional TERM "$LAUNCHED_PID"
        wait "$LAUNCHED_PID" 2>/dev/null || true
        return 1
    fi
}

cleanup() {
    local status=$?
    local children_stopped=0
    local pid

    trap - EXIT INT TERM HUP

    if [ -n "$STARTUP_WAIT_PID" ]; then
        kill -TERM "$STARTUP_WAIT_PID" 2>/dev/null || true
        wait "$STARTUP_WAIT_PID" 2>/dev/null || true
        STARTUP_WAIT_PID=""
    fi

    if [ -n "$TAIL_PID" ]; then
        kill -TERM "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
        remove_if_unchanged "$STATE_DIR/logs.pid" "$LOG_RECORD"
    fi

    # These cover the pre-publication setsid window. Published groups receive
    # the same TERM again through stop.sh, which is harmless and idempotent.
    for pid in "${PROVISIONAL_PIDS[@]}"; do
        signal_provisional TERM "$pid"
    done

    if LOFAI_STATE_DIR="$STATE_DIR" "$SCRIPT_DIR/stop.sh" --children --quiet; then
        children_stopped=1
    else
        status=1
    fi

    # Reap only after verified shutdown. If native work somehow survives KILL,
    # do not turn a reported failure into an unbounded wait.
    if [ "$children_stopped" -eq 1 ]; then
        if [ -n "$BACKEND_PID" ]; then
            wait "$BACKEND_PID" 2>/dev/null || true
        fi
        if [ -n "$FRONTEND_PID" ]; then
            wait "$FRONTEND_PID" 2>/dev/null || true
        fi
        remove_if_unchanged "$LOCK_FILE" "$LOCK_RECORD"
    fi

    exit "$status"
}

echo "Starting lofAI Application..."

# The lock itself is the supervisor metadata and is created atomically. A
# second launcher first stops the complete older run, then retries ownership.
if ! acquire_lock; then
    echo "Stopping the previous lofAI run..."
    LOFAI_STATE_DIR="$STATE_DIR" "$SCRIPT_DIR/stop.sh" --quiet || exit 1
    if ! acquire_lock; then
        echo "Error: another lofAI launcher owns $LOCK_FILE." >&2
        exit 1
    fi
fi

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Clean metadata/groups from a crash that predated the lock, without targeting
# the supervisor record we have just acquired.
if ! LOFAI_STATE_DIR="$STATE_DIR" "$SCRIPT_DIR/stop.sh" --children --quiet; then
    exit 1
fi

echo "Starting backend server..."
launch_group "$BACKEND_SCRIPT" "$STATE_DIR/backend.log" "$STATE_DIR/backend.pid" || exit 1
BACKEND_PID="$LAUNCHED_PID"
echo "   Backend process group: $BACKEND_PID"

# Background sleep + wait is interruptible by Bash traps. A foreground sleep
# can defer TERM until it finishes, which is unacceptable during startup.
sleep "$STARTUP_DELAY" &
STARTUP_WAIT_PID=$!
wait "$STARTUP_WAIT_PID" 2>/dev/null || true
STARTUP_WAIT_PID=""
if ! pid_is_live "$BACKEND_PID"; then
    echo "Backend exited during startup. See $STATE_DIR/backend.log."
    exit 1
fi

echo "Starting frontend server..."
launch_group "$FRONTEND_SCRIPT" "$STATE_DIR/frontend.log" "$STATE_DIR/frontend.pid" || exit 1
FRONTEND_PID="$LAUNCHED_PID"
echo "   Frontend process group: $FRONTEND_PID"

echo ""
echo "lofAI is running."
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop the application and all processing."
echo ""

tail -F "$STATE_DIR/backend.log" "$STATE_DIR/frontend.log" &
TAIL_PID=$!
if ! write_pid_file "$STATE_DIR/logs.pid" pid "$TAIL_PID"; then
    exit 1
fi

# Bash 3 on macOS has no wait -n. Poll non-zombie group leaders; if either
# exits, the application is incomplete and the EXIT trap tears down the rest.
while pid_is_live "$BACKEND_PID" && pid_is_live "$FRONTEND_PID"; do
    sleep 0.2
done

echo "A lofAI service stopped; shutting down the remaining processing."
exit 1
