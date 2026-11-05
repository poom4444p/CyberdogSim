#!/bin/bash
# The hardware-in-the-loop rig, driven from the Mac: the world here (go2_world:
# physics, people, the lift, the Mid-360), everything the dog carries on the
# J4012 over USB-C (scripts/hil/onboard.sh there). See scripts/hil/README.md.
#
#   bash scripts/hil/run_hil.sh sync              copy Cyberdog and go2_board, as checked out, to the board
#   bash scripts/hil/run_hil.sh clock             set the board's clock from this Mac (sudo there)
#   bash scripts/hil/run_hil.sh walk              stand, walk 4 s, stand: is the board's loop sound? (~1 min)
#   bash scripts/hil/run_hil.sh route DEST [N]    the whole route to DEST, VAMOS asked for N paths (2)
#   bash scripts/hil/run_hil.sh stop              stop the world here and the board and brain there
#
# Each run writes its logs to $OUT (output/hil/<time>-<what>) and ends with a
# summary: the brain's outcome, and how regularly states and commands moved.
# The world, the LiDAR and the board are stopped at the end, or on Ctrl-C;
# VAMOS is left up (it takes ~25 s to load): `onboard.sh stop all` there stops it.
#
# Settings (environment):
#   BOARD        ssh target of the J4012            (jittarin@192.168.55.1)
#   MAC_IF       this Mac's interface on the link   (en9)
#   MAC_IP       this Mac's address on the link     (192.168.55.100)
#   BOARD_IF     the board's interface on the link  (l4tbr0)
#   GO2_WORLD    the go2_world checkout             (~/Work/unitree/go2_world)
#   GO2_BOARD    the go2_board checkout             (~/Work/unitree/go2_board)
#   PY           Python with go2_world's packages   (python)
#   PEOPLE       people walking in the world        (4)
#   WORLD_ARGS   more run_world arguments, e.g. --held-pd
#   TRACE=1      the brain prints its commands
#   ROUTE_S      a route is stopped after this long, in seconds  (560)
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
CYBERDOG=$(cd "$HERE/../.." && pwd)
BOARD=${BOARD:-jittarin@192.168.55.1}
MAC_IF=${MAC_IF:-en9}
MAC_IP=${MAC_IP:-192.168.55.100}
BOARD_IF=${BOARD_IF:-l4tbr0}
GO2_WORLD=${GO2_WORLD:-$HOME/Work/unitree/go2_world}
GO2_BOARD=${GO2_BOARD:-$HOME/Work/unitree/go2_board}
PY=${PY:-python}
PEOPLE=${PEOPLE:-4}
WORLD_ARGS=${WORLD_ARGS:-}
TRACE=${TRACE:-0}
ONBOARD='~/Cyberdog/scripts/hil/onboard.sh'
BOARD_PY='. ~/go2env/bin/activate; python'

on_board() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$BOARD" "$@"; }

new_out() {
    OUT=${OUT:-$CYBERDOG/output/hil/$(date +%Y%m%d-%H%M%S)-$1}
    mkdir -p "$OUT"
    echo "logs: $OUT"
}

clock_offset() {    # board minus Mac, whole seconds
    local b; b=$(on_board 'date -u +%s') || return 1
    echo $(( b - $(date -u +%s) ))
}

check_clock() {
    local off; off=$(clock_offset) || { echo "board: unreachable at $BOARD"; exit 1; }
    if [ "${off#-}" -gt 2 ]; then
        echo "WARNING: the board's clock is ${off} s off this Mac's (it has no RTC). The rig runs"
        echo "         on monotonic clocks, so this only spoils log times: run_hil.sh clock fixes it."
    fi
}

# Background jobs: only the program goes in the background, with every fd off
# ours -- a backgrounded `cd && prog` keeps a shell holding our stdout, and
# whatever reads it (a pipe, ssh) waits for it to end. And [x]yz patterns, so
# pkill -f matches the process and not a shell whose command line holds it.
stop_world() { pkill -f "scripts/[r]un_world.py"; pkill -f "scripts/[r]un_lidar.py"; }

start_world() {
    stop_world
    on_board "$ONBOARD stop" > /dev/null 2>&1
    sleep 1
    # shellcheck disable=SC2086
    (cd "$GO2_WORLD" || exit 1; nohup "$PY" -u scripts/run_world.py --headless --people="$PEOPLE" \
        --lift-listen="$MAC_IP" $WORLD_ARGS "$MAC_IF" > "$OUT/world.log" 2>&1 < /dev/null &)
    for _ in $(seq 1 30); do grep -q "people walking" "$OUT/world.log" 2>/dev/null && break; sleep 0.5; done
    grep -q "people walking" "$OUT/world.log" || { echo "world: did not start, see $OUT/world.log"; exit 1; }
    grep -E "^world:|^lift:" "$OUT/world.log"
    (cd "$GO2_WORLD" || exit 1; nohup "$PY" -u scripts/run_lidar.py --interface "$MAC_IF" \
        --host "${BOARD#*@}" > "$OUT/lidar.log" 2>&1 < /dev/null &)
    sleep 13                                   # the emulator loads the scene, then streams
    grep -m1 "^mid360:" "$OUT/lidar.log" || { echo "lidar: not streaming, see $OUT/lidar.log"; exit 1; }
}

start_gaps() {
    on_board "($BOARD_PY ~/Cyberdog/scripts/hil/dds_gaps.py 3600 $BOARD_IF > ~/hil_gaps.log 2>&1 < /dev/null &)"
}

show_gaps() {
    on_board 'pkill -TERM -f "[d]ds_gaps.py"; for i in $(seq 1 20); do grep -q lowcmd ~/hil_gaps.log && break; sleep 0.5; done; cat ~/hil_gaps.log' \
        | grep -v multicast | tee "$OUT/gaps.log"
}

stop_all() {
    stop_world
    on_board "pkill -f '[d]ds_gaps.py'; $ONBOARD stop" > /dev/null 2>&1
}

case "${1:-}" in
  sync)
    # What is checked out is what goes: say which branches, and ask. A sync
    # from the wrong checkout put an old go2_board on the board once, and
    # nothing said so until its walk fell.
    for r in "$CYBERDOG" "$GO2_BOARD"; do
        dirty=$(git -C "$r" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
        printf '  %-10s %s @ %s%s\n' "$(basename "$r")" "$(git -C "$r" branch --show-current)" \
            "$(git -C "$r" log -1 --format='%h %s' | cut -c1-60)" \
            "$([ "$dirty" != 0 ] && echo "  (+$dirty uncommitted)")"
    done
    if [ -t 0 ]; then
        read -r -p "copy these to $BOARD? [y/N] " yes
        [ "$yes" = y ] || { echo "not synced"; exit 1; }
    elif [ "${SYNC_YES:-0}" != 1 ]; then
        echo "not synced: no terminal to ask; SYNC_YES=1 to sync anyway"; exit 1
    fi
    rsync -a --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' --exclude .git \
        --exclude models --exclude output --exclude vendor/VAMOS --exclude '*.docx' --exclude '*.pptx' \
        --exclude vamos_train.tgz "$CYBERDOG/" "$BOARD:~/Cyberdog/" && echo "Cyberdog -> $BOARD:~/Cyberdog"
    rsync -a --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' --exclude .git \
        "$GO2_BOARD/" "$BOARD:~/go2_board/" && echo "go2_board -> $BOARD:~/go2_board" ;;

  clock)
    # A sudo password prompt needs a terminal: run this one yourself, not from a script.
    ssh -t "$BOARD" "sudo date -u -s @$(date -u +%s) > /dev/null" && echo "board clock off by $(clock_offset) s" ;;

  walk)
    new_out walk; check_clock
    trap stop_all EXIT
    start_world
    on_board "$ONBOARD board" | tail -1
    start_gaps
    on_board "cd ~/go2_board && $BOARD_PY -u scripts/brain_cli.py stand:4 walk:4:0.54,0,0.47 stand:2" \
        > "$OUT/brain_cli.log" 2>&1
    grep "t=" "$OUT/brain_cli.log" | sed 's/  */ /g' | tail -n +8
    show_gaps
    if grep -q "tripped=tilted" "$OUT/brain_cli.log"; then echo "RESULT: FELL"; exit 1; fi
    echo "RESULT: walked" ;;

  route)
    dest=${2:?"route DEST [N]"}
    new_out route; check_clock
    trap stop_all EXIT
    start_world
    on_board "$ONBOARD vamos"
    on_board "$ONBOARD board" | tail -1
    start_gaps
    on_board "TRACE=$TRACE timeout ${ROUTE_S:-560} $ONBOARD brain '$dest' ${3:-2}" > "$OUT/brain.log" 2>&1
    grep -E "ARRIVED|FAILED|HALTED|leg |obstacles|collisions|VAMOS floor|VLM ASYNC|help|planner" \
        "$OUT/brain.log" | awk '!seen[$0]++' | cut -c1-230
    show_gaps
    on_board "$ONBOARD log" | tail -n +3
    grep -q "^ARRIVED" "$OUT/brain.log" ;;

  stop)
    stop_all; echo "stopped (VAMOS left up)" ;;

  *)
    sed -n '2,28p' "$0"; exit 2 ;;
esac
