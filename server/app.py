# ========================= #
# run this file for backend #
# ========================= #

import os, sys, subprocess
from flask import Flask

HERE = os.path.dirname(os.path.abspath(__file__))
GAME_PATH = os.path.join(HERE, "game_loop.py")

app = Flask(__name__)

# keep a handle so we don’t spawn multiples accidentally
game_proc = None

@app.route("/start", methods=["POST"])
def start():
    global game_proc
    # if not started or it already exited, start again
    if (game_proc is None) or (game_proc.poll() is not None):
        print("[backend] launching game process...", flush=True)
        # use python.exe (not pythonw.exe) so you see logs if something fails
        game_proc = subprocess.Popen(
            [sys.executable, GAME_PATH],
            cwd=HERE,
            close_fds=False  # allow window/handles on Windows
        )
    else:
        print("[backend] game already running (pid=%s)" % game_proc.pid, flush=True)
    return ("", 204)

@app.route("/status")
def status():
    global game_proc
    if game_proc is None:
        return "stopped", 200
    return ("running" if game_proc.poll() is None else "stopped", 200)

if __name__ == "__main__":
    # IMPORTANT: no reloader, to avoid double route registration
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)