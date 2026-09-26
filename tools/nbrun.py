"""Run notebook cells one at a time in a persistent Jupyter kernel and save outputs into the .ipynb.

  python tools/nbrun.py start                      # launch kernel (keep running in background)
  python tools/nbrun.py run  <nb.ipynb> <i>[,j..|a-b] # execute code cells by index (0-based)
  python tools/nbrun.py list <nb.ipynb>             # show code-cell indices
"""
import os, re, sys, time
import nbformat
from jupyter_client import BlockingKernelClient

CONN = os.path.join(os.path.dirname(__file__), "..", "cache", "kernel.json")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def code_cells(nb):
    return [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]


def parse(sel, n):
    out = []
    for part in sel.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def client():
    kc = BlockingKernelClient(connection_file=CONN); kc.load_connection_file(); kc.start_channels()
    kc.wait_for_ready(timeout=4 * 3600)  # waits while the kernel finishes a running cell
    return kc


def exec_code(code):
    """run ad-hoc code in the kernel (not saved to the notebook)"""
    kc = client()
    msg_id = kc.execute(code)
    while True:
        msg = kc.get_iopub_msg(timeout=None)
        if msg["parent_header"].get("msg_id") != msg_id: continue
        mt, c = msg["msg_type"], msg["content"]
        if mt == "stream": print(c["text"], end="", flush=True)
        elif mt in ("execute_result", "display_data"): print(c["data"].get("text/plain", ""))
        elif mt == "error": print(ANSI.sub("", "\n".join(c["traceback"]))); sys.exit(1)
        elif mt == "status" and c["execution_state"] == "idle": break


def run(nb_path, sel):
    kc = client()
    nb = nbformat.read(nb_path, as_version=4)
    idx = code_cells(nb)
    for ci in parse(sel, len(idx)):
        cell = nb.cells[idx[ci]]
        print(f"\n===== cell {ci} =====", flush=True)
        t = time.time()
        msg_id = kc.execute(cell.source)
        outs, err = [], False
        while True:
            msg = kc.get_iopub_msg(timeout=None)
            if msg["parent_header"].get("msg_id") != msg_id: continue
            mt, c = msg["msg_type"], msg["content"]
            if mt == "stream":
                print(c["text"], end="", flush=True)
                if outs and outs[-1].output_type == "stream" and outs[-1].name == c["name"]:
                    outs[-1].text += c["text"]
                else:
                    outs.append(nbformat.v4.new_output("stream", name=c["name"], text=c["text"]))
            elif mt in ("execute_result", "display_data"):
                print(c["data"].get("text/plain", ""), flush=True)
                kw = {"execution_count": c.get("execution_count")} if mt == "execute_result" else {}
                outs.append(nbformat.v4.new_output(mt, data=c["data"], metadata=c.get("metadata", {}), **kw))
            elif mt == "error":
                print(ANSI.sub("", "\n".join(c["traceback"])), flush=True)
                outs.append(nbformat.v4.new_output("error", ename=c["ename"], evalue=c["evalue"],
                                                   traceback=c["traceback"]))
                err = True
            elif mt == "execute_input":
                cell.execution_count = c.get("execution_count")
            elif mt == "status" and c["execution_state"] == "idle":
                break
        cell.outputs = outs
        nbformat.write(nb, nb_path)
        print(f"----- cell {ci} {'FAILED' if err else 'ok'} in {time.time()-t:.0f}s", flush=True)
        if err:
            sys.exit(1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "start":
        import subprocess  # (os.execvp detaches on Windows; wait instead so the kernel lives with this process)
        if os.path.exists(CONN): os.remove(CONN)
        sys.exit(subprocess.call([sys.executable, "-m", "ipykernel_launcher", "-f", os.path.abspath(CONN)],
                                 cwd=os.path.join(os.path.dirname(__file__), "..")))
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
    elif cmd == "exec":
        exec_code(sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read())
    elif cmd == "list":
        nb = nbformat.read(sys.argv[2], as_version=4)
        for k, i in enumerate(code_cells(nb)):
            print(k, nb.cells[i].source.splitlines()[1][:90] if len(nb.cells[i].source.splitlines()) > 1 else "")
