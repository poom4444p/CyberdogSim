#!/bin/bash
# ^ The first line ("shebang") says: run this file with bash.
#
# check.sh -- run the project's checks in order, stop at the first failure.
#
#   bash scripts/check.sh                          # pytest + selftest
#   bash scripts/check.sh gate "room 201" 1 4 7 8  # + Gate D on one route, those seeds
#   bash scripts/check.sh gate all                 # + the full Gate D campaign (~50 min)
#
# Anything after a "#" is a comment: bash ignores it.

set -e
# ^ "exit on error": if any command fails, stop the script right there,
#   so a failing test is never followed by a misleading "all good".

cd "$(dirname "$0")/.."
# ^ $0 is this script's own path. dirname cuts it to its folder (scripts/),
#   and /.. goes one up, to the repo root. $( ... ) runs a command and puts
#   its output in place. The quotes keep paths with spaces in one piece.

eval "$(conda shell.bash hook)"
conda activate cyberdog_sim
# ^ Same as typing "conda activate cyberdog_sim" yourself. Scripts need the
#   eval line first, because conda's "activate" is a shell function that a
#   fresh script does not have yet.

echo "== 1/3 unit tests"
# ^ echo prints a line, so you can see which step is running.
pytest -q

echo "== 2/3 self-test"
python tests/selftest.py | tee /tmp/selftest.txt
# ^ The | ("pipe") sends the output of the left command into the right one.
#   tee shows it on screen AND saves a copy in a file.
if ! grep -q "ALL PASS" /tmp/selftest.txt; then
    # ^ grep -q searches the file quietly; it "succeeds" if it finds the text.
    #   ! flips that, so this block runs when ALL PASS is NOT there.
    echo "self-test FAILED -- see the lines marked [FAIL] above"
    exit 1
    # ^ exit 1 ends the script and reports failure (0 means success).
fi
# ^ "fi" is "if" backwards: it closes the if-block.

if [ "$1" != "gate" ]; then
    # ^ $1 is the first word you typed after the script name.
    #   [ ... ] is a test; != means "is not equal to".
    echo "== done: pytest and self-test pass (add 'gate' to also run Gate D)"
    exit 0
fi

echo "== 3/3 Gate D"
URL="http://127.0.0.1:8009"
# ^ A variable. No spaces around "=" in bash, or it breaks.
if ! curl -s -m 5 "$URL/health" > /dev/null; then
    # ^ curl asks the VAMOS server if it is up. "> /dev/null" throws the
    #   answer away -- only whether it succeeded matters here.
    echo "The VAMOS server is not running. Start it in another terminal:"
    echo "  conda activate vamos_mac"
    echo "  cd vendor/VAMOS/server && bash start_server.sh"
    exit 1
fi

if [ "$2" = "all" ]; then
    python scripts/gate_d.py --seeds 1 2 3 4 5 6 7 8 9 10
else
    ROUTE="$2"
    shift 2
    # ^ shift 2 drops the first two words ("gate" and the route), so that
    #   "$@" below is just the seed numbers you typed.
    python scripts/gate_d.py --routes "$ROUTE" --seeds "$@"
fi
