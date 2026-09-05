#!/bin/bash

# Shared process-group supervision for the standalone backend/frontend
# launchers. This file is sourced; call lofai_service_lifecycle_init once and
# run every command that may create workers with lofai_run_service_command.

LOFAI_SERVICE_PID=""
LOFAI_SERVICE_PYTHON=""
LOFAI_SERVICE_TERM_TIMEOUT="${LOFAI_SERVICE_TERM_TIMEOUT:-8}"
LOFAI_SERVICE_KILL_TIMEOUT="${LOFAI_SERVICE_KILL_TIMEOUT:-2}"

lofai_service_pid_is_live() {
    ps -o state= -p "$1" 2>/dev/null | \
        awk 'NF && $1 !~ /^Z/ { found=1 } END { exit !found }'
}

lofai_service_group_is_live() {
    ps -axo pgid=,state= 2>/dev/null | awk -v wanted="$1" \
        '$1 == wanted && $2 !~ /^Z/ { found=1 } END { exit !found }'
}

lofai_service_group_of() {
    ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]'
}

lofai_service_target_is_live() {
    local pid="$1"
    lofai_service_group_is_live "$pid" || lofai_service_pid_is_live "$pid"
}

lofai_service_signal() {
    local signal="$1"
    local pid="$2"
    local group

    # The child normally becomes its own session and process-group leader.
    # During the few instructions before setsid(), only signal that PID: using
    # its inherited group could also target this launcher or the user's shell.
    group="$(lofai_service_group_of "$pid")"
    if [ "$group" = "$pid" ] || lofai_service_group_is_live "$pid"; then
        kill -"$signal" -- "-$pid" 2>/dev/null || true
    elif lofai_service_pid_is_live "$pid"; then
        kill -"$signal" "$pid" 2>/dev/null || true
    fi
}

lofai_service_wait_for_group() {
    local pid="$1"
    local attempt=0

    while lofai_service_pid_is_live "$pid"; do
        if [ "$(lofai_service_group_of "$pid")" = "$pid" ]; then
            return 0
        fi
        attempt=$((attempt + 1))
        [ "$attempt" -ge 200 ] && return 1
        sleep 0.01
    done
    return 1
}

lofai_service_wait_until_stopped() {
    local pid="$1"
    local timeout="$2"
    local deadline=$(( $(date +%s) + timeout ))

    while lofai_service_target_is_live "$pid"; do
        [ "$(date +%s)" -ge "$deadline" ] && return 1
        sleep 0.1
    done
    return 0
}

lofai_stop_service_group() {
    local pid="$1"

    lofai_service_target_is_live "$pid" || return 0
    lofai_service_signal TERM "$pid"
    if ! lofai_service_wait_until_stopped "$pid" "$LOFAI_SERVICE_TERM_TIMEOUT"; then
        echo "Service workers did not stop gracefully; forcing them down..." >&2
        lofai_service_signal KILL "$pid"
        if ! lofai_service_wait_until_stopped "$pid" "$LOFAI_SERVICE_KILL_TIMEOUT"; then
            echo "Error: service process group $pid is still running." >&2
            return 1
        fi
    fi
}

lofai_service_cleanup() {
    local status=$?
    local stopped=0

    trap - EXIT
    # The application supervisor and the terminal can race to deliver the same
    # shutdown signal. Ignore repeats while cleanup is already enforcing its
    # deadline; restoring defaults here could kill this wrapper before it has
    # stopped the nested service group.
    trap '' INT TERM HUP
    if [ -n "$LOFAI_SERVICE_PID" ]; then
        if lofai_stop_service_group "$LOFAI_SERVICE_PID"; then
            stopped=1
        else
            status=1
        fi
        # Reap the group leader after all of its descendants are confirmed
        # gone. wait is bounded indirectly by the group checks above.
        if [ "$stopped" -eq 1 ]; then
            wait "$LOFAI_SERVICE_PID" 2>/dev/null || true
        fi
        LOFAI_SERVICE_PID=""
    fi
    exit "$status"
}

lofai_service_lifecycle_init() {
    if ! [[ "$LOFAI_SERVICE_TERM_TIMEOUT" =~ ^(0|[1-9][0-9]*)$ ]] || \
       ! [[ "$LOFAI_SERVICE_KILL_TIMEOUT" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "Service shutdown timeouts must be canonical whole seconds." >&2
        return 2
    fi

    LOFAI_SERVICE_PYTHON="$(command -v python3 || true)"
    if [ -z "$LOFAI_SERVICE_PYTHON" ]; then
        echo "Error: python3 is required to supervise service processes." >&2
        return 1
    fi

    trap lofai_service_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
}

lofai_run_service_command() {
    local status

    "$LOFAI_SERVICE_PYTHON" -c \
        'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
        "$@" &
    LOFAI_SERVICE_PID=$!

    if ! lofai_service_wait_for_group "$LOFAI_SERVICE_PID"; then
        # The command may have exited normally before the parent observed
        # setsid(). Capture that status below; otherwise make sure a failed
        # launch cannot leave an untracked provisional child.
        if lofai_service_target_is_live "$LOFAI_SERVICE_PID"; then
            lofai_stop_service_group "$LOFAI_SERVICE_PID" || true
        fi
    fi

    if wait "$LOFAI_SERVICE_PID"; then
        status=0
    else
        status=$?
    fi

    # Some tools let their group leader exit while a worker is still alive.
    # Keep the PID/PGID published until those descendants are also gone.
    if lofai_service_group_is_live "$LOFAI_SERVICE_PID"; then
        if ! lofai_stop_service_group "$LOFAI_SERVICE_PID"; then
            status=1
        fi
    fi
    LOFAI_SERVICE_PID=""
    return "$status"
}
