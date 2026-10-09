"""One small client for the GPU rental providers the H100 column can use, behind one interface.

    export RUNPOD_API_KEY=...   (or LAMBDA_API_KEY=...)
    .venv/bin/python scripts/gpu/provider.py availability --provider runpod
    .venv/bin/python scripts/gpu/provider.py launch --provider runpod --hours 4
    .venv/bin/python scripts/gpu/provider.py status --provider runpod --id <id>
    .venv/bin/python scripts/gpu/provider.py terminate --provider runpod --id <id>
    .venv/bin/python scripts/gpu/provider.py session --provider runpod --hours 16 --ssh-key ~/.ssh/id_ed25519   # launch, sync, run, fetch, terminate

Providers (REST, bearer tokens, stdlib HTTP only):
  runpod   REST v2 at https://api.runpod.io/v2 (bearer key, also for the catalog): POST /pods {name, image,
           gpu:{id,count}, cloud, disk, ports:["22/tcp"], startSsh:true}; GET /pods/{id} -> publicIp / portMappings or
           the ssh block; DELETE /pods/{id}. The account's registered SSH keys are provisioned by startSsh.
  lambda   POST https://cloud.lambda.ai/api/v1/instance-operations/launch {region_name, instance_type_name,
           ssh_key_names, name}; GET /instances/{id} -> status, ip; POST /instance-operations/terminate {instance_ids}.
           SSH as ubuntu on port 22. Launch is rate limited (one per 12 s).
Every session appends to the private ledger (docs/infra/gpu_ledger.json: provider, id, launched_at, hours cap,
terminated_at, hourly price) and terminates the machine on every exit path, including a wall-clock cap.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
LEDGER = REPO / "docs" / "infra" / "gpu_ledger.json"
FETCH_EVERY_S = 900  # incremental result fetch interval during a session
ENV_FILE = REPO / ".env"  # gitignored and excluded from state bundles; KEY=value lines, loaded if the variable is not already set


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
CAP_USD = 200.0

PRICES = {  # on-demand $/GPU-hour read from the providers' pricing pages on 2026-09-27; re-check before each session
    ("runpod", "NVIDIA H100 PCIe", "SECURE"): 2.89, ("runpod", "NVIDIA H100 80GB HBM3", "SECURE"): 3.49,
    ("runpod", "NVIDIA H100 PCIe", "COMMUNITY"): 1.99, ("lambda", "gpu_1x_h100_pcie", ""): 3.29,
}


class Http:
    def __init__(self, base: str, token: str):
        self.base, self.token = base.rstrip("/"), token

    def __call__(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                                              "User-Agent": "onepass-gpu-client/1.0"})  # the API front door rejects the default urllib agent (1010)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode()
                return json.loads(txt) if txt.strip() else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()[:400]}") from None


class Provider:
    name = ""
    user = "root"

    def __init__(self, http: Http):
        self.http = http

    def availability(self) -> list[dict]: ...
    def launch(self, ssh_public_key: str, name: str, **kw) -> str: ...
    def status(self, iid: str) -> dict: ...  # {"state": str, "ip": str|None, "port": int}
    def terminate(self, iid: str) -> None: ...


class RunPod(Provider):
    name = "runpod"
    user = "root"
    IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

    def __init__(self, token: str, gpu: str = "NVIDIA H100 PCIe", cloud: str = "SECURE", http: Http | None = None):
        super().__init__(http or Http("https://api.runpod.io/v2", token))
        self.gpu, self.cloud = gpu, cloud
        # priority list: PCIe first (cheaper), SXM when PCIe has no stock; the ledger price is taken from the pod that lands
        # only an H100 request may fall through between the PCIe and SXM parts; any other GPU is launched as asked or not at all
        h100 = ("NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3")
        self.gpu_priority = [gpu] + [g for g in h100 if g != gpu] if gpu in h100 else [gpu]

    def availability(self) -> list[dict]:
        out = []
        for g in self.gpu_priority:
            d = self.http("GET", f"/catalog/gpus/{urllib.parse.quote(g)}?include=AVAILABILITY&product=POD&cloud={self.cloud}")
            out.append({"id": d.get("id"), "availability": d.get("availability"), "price": (d.get("price") or {}).get(self.cloud.lower()),
                        "dataCenters": [dc.get("id") for dc in d.get("dataCenters", []) if dc.get("availability") not in (None, "NONE")]})
        return out

    def launch(self, ssh_public_key: str, name: str, disk_gb: int = 120, **kw) -> str:
        """Tries each GPU type in priority order; the account's registered SSH keys are injected by startSsh, and the
        given public key is also passed explicitly so a pod never comes up without SSH."""
        last = None
        for g in self.gpu_priority:
            gpu = {"id": g, "count": 1, **({"minCudaVersion": min(kw["cuda_versions"])} if kw.get("cuda_versions") else {})}
            body = {"name": name, "image": self.IMAGE, "gpu": gpu, "cloud": self.cloud, "disk": disk_gb,
                    "ports": ["22/tcp"], "startSsh": True,
                    **({"dataCenterIds": list(kw["data_centers"])} if kw.get("data_centers") else {}),
                    # SSH_PUBLIC_KEY is the per-pod override the platform documents; PUBLIC_KEY is what the official
                    # image's start script reads. Set both so the key lands in authorized_keys either way.
                    "env": {"SSH_PUBLIC_KEY": ssh_public_key.strip(), "PUBLIC_KEY": ssh_public_key.strip()}}
            try:
                return self.http("POST", "/pods", body)["id"]
            except RuntimeError as e:  # no stock for this type: try the next one
                last = e
        raise RuntimeError(f"no pod could be created: {last}")

    def status(self, iid: str) -> dict:
        d = self.http("GET", f"/pods/{iid}")
        ip, port = d.get("publicIp") or None, None
        pm = d.get("portMappings") or {}
        if pm.get("22"):
            port = int(pm["22"])
        ssh = d.get("ssh") or {}
        direct = ssh.get("direct") if isinstance(ssh, dict) else None
        if isinstance(direct, dict):
            ip, port = direct.get("host") or direct.get("ip") or ip, int(direct.get("port") or port or 0) or None
        gpu = d.get("gpu") if isinstance(d.get("gpu"), dict) else {}
        return {"state": d.get("desiredStatus") or d.get("status") or "?", "ip": ip, "port": port,
                "gpu": gpu.get("id") or d.get("gpuTypeId"), "usd_per_hour": d.get("costPerHr") or d.get("price"), "raw": d}

    def terminate(self, iid: str) -> None:
        self.http("DELETE", f"/pods/{iid}")


class Lambda(Provider):
    name = "lambda"
    user = "ubuntu"

    def __init__(self, token: str, instance_type: str = "gpu_1x_h100_pcie", ssh_key_name: str = "onepass", http: Http | None = None):
        super().__init__(http or Http("https://cloud.lambda.ai/api/v1", token))
        self.instance_type, self.ssh_key_name = instance_type, ssh_key_name

    def availability(self) -> list[dict]:
        d = self.http("GET", "/instance-types").get("data", {})
        out = []
        for k, v in d.items():
            if "h100" in k:
                out.append({"id": k, "regions": [r.get("name") for r in v.get("regions_with_capacity_available", [])],
                            "price_cents_per_hour": v.get("instance_type", {}).get("price_cents_per_hour")})
        return out

    def launch(self, ssh_public_key: str, name: str, region: str | None = None, **kw) -> str:
        if region is None:
            av = [a for a in self.availability() if a["id"] == self.instance_type and a["regions"]]
            if not av:
                raise RuntimeError(f"no capacity for {self.instance_type}")
            region = av[0]["regions"][0]
        body = {"region_name": region, "instance_type_name": self.instance_type, "ssh_key_names": [self.ssh_key_name], "name": name}
        return self.http("POST", "/instance-operations/launch", body)["data"]["instance_ids"][0]

    def status(self, iid: str) -> dict:
        d = self.http("GET", f"/instances/{iid}").get("data", {})
        return {"state": d.get("status", "?"), "ip": d.get("ip") or None, "port": 22, "raw": d}

    def terminate(self, iid: str) -> None:
        self.http("POST", "/instance-operations/terminate", {"instance_ids": [iid]})


class RunPodCPU(RunPod):
    """CPU-only pod on the documented REST API (rest.runpod.io/v1): computeType CPU, a flavor list and a vCPU count.
    Shares the status and terminate parsing with the GPU client; the same image (python 3.11, sshd) runs without a GPU."""

    name = "runpod-cpu"

    def __init__(self, token: str, flavor: str = "cpu5c", vcpu: int = 32, cloud: str = "SECURE", http: Http | None = None):
        Provider.__init__(self, http or Http("https://rest.runpod.io/v1", token))
        self.flavor, self.vcpu, self.cloud = flavor, vcpu, cloud
        self.gpu_priority = []

    def availability(self) -> list[dict]:
        return [{"id": self.flavor, "vcpu": self.vcpu, "note": "CPU pods have no availability endpoint; launch reports stock"}]

    def launch(self, ssh_public_key: str, name: str, disk_gb: int = 100, **kw) -> str:
        """Tries the requested flavor and size first, then compute-optimized alternatives and smaller sizes; the size that
        lands is recorded on the instance (status reports vcpuCount) and the chain only ever uses 8 timing threads."""
        flavors = [self.flavor] + [f for f in ("cpu5c", "cpu3c", "cpu5g", "cpu3g") if f != self.flavor]
        sizes = [v for v in (self.vcpu, 24, 16, 8) if v <= self.vcpu]  # 8 vCPU still serves the 8-thread timing cells
        last = None
        for vcpu in sizes:
            for flavor in flavors:
                body = {"name": name, "imageName": self.IMAGE, "computeType": "CPU", "cpuFlavorIds": [flavor], "cpuFlavorPriority": "custom",
                        "vcpuCount": vcpu, "cloudType": self.cloud, "containerDiskInGb": disk_gb, "volumeInGb": 0, "ports": ["22/tcp"],
                        "env": {"SSH_PUBLIC_KEY": ssh_public_key.strip(), "PUBLIC_KEY": ssh_public_key.strip()}}
                try:
                    iid = self.http("POST", "/pods", body)["id"]
                    self.flavor, self.vcpu = flavor, vcpu
                    print(f"[gpu] cpu pod: {flavor} x{vcpu}", flush=True)
                    return iid
                except RuntimeError as e:
                    last = e
                    time.sleep(3)
        raise RuntimeError(f"no cpu pod could be created: {last}")

    def status(self, iid: str) -> dict:
        d = self.http("GET", f"/pods/{iid}")
        ip, port = d.get("publicIp") or None, None
        pm = d.get("portMappings") or {}
        if pm.get("22"):
            port = int(pm["22"])
        return {"state": d.get("desiredStatus") or "?", "ip": ip, "port": port, "gpu": f"{self.flavor} x{self.vcpu}",
                "usd_per_hour": d.get("costPerHr"), "raw": d}

    def terminate(self, iid: str) -> None:
        self.http("DELETE", f"/pods/{iid}")


def make(provider: str, **kw) -> Provider:
    load_env_file()
    if provider == "runpod":
        return RunPod(os.environ["RUNPOD_API_KEY"], **kw)
    if provider == "runpod-cpu":
        return RunPodCPU(os.environ["RUNPOD_API_KEY"], **kw)
    if provider == "lambda":
        return Lambda(os.environ["LAMBDA_API_KEY"], **kw)
    raise SystemExit(f"unknown provider {provider}")


# ---- ledger -------------------------------------------------------------------------------------------------------
def ledger_load() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"cap_usd": CAP_USD, "sessions": []}


def ledger_save(d: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(d, indent=1) + "\n")


def ledger_spent(d: dict) -> float:
    total = 0.0
    for s in d["sessions"]:
        end = s.get("terminated_at") or time.time()
        total += (end - s["launched_at"]) / 3600 * s["usd_per_hour"]
    return total


def ledger_open(provider: str, iid: str, hours: float, usd_per_hour: float) -> None:
    d = ledger_load()
    if ledger_spent(d) + hours * usd_per_hour > d["cap_usd"]:
        raise SystemExit(f"session would exceed the cap: spent {ledger_spent(d):.2f} + {hours * usd_per_hour:.2f} > {d['cap_usd']}")
    d["sessions"].append({"provider": provider, "id": iid, "launched_at": time.time(), "hours": hours, "usd_per_hour": usd_per_hour})
    ledger_save(d)


def ledger_close(iid: str) -> None:
    d = ledger_load()
    for s in d["sessions"]:
        if s["id"] == iid and not s.get("terminated_at"):
            s["terminated_at"] = time.time()
    ledger_save(d)


# ---- session -----------------------------------------------------------------------------------------------------
def wait_ssh(p: Provider, iid: str, key: str, timeout_s: int = 1800) -> tuple[str, int]:
    """Polls until the instance answers SSH. Container images of several GB take 10 to 20 minutes to pull on a fresh
    host, so the default deadline is 30 minutes; each poll logs the provider's state so a stuck launch is diagnosable."""
    t0 = time.time()
    last, denied_since = None, None
    while time.time() - t0 < timeout_s:
        st = p.status(iid)
        line = f"[gpu] {int(time.time() - t0)}s state={st['state']} ip={st['ip']} port={st['port']} gpu={st.get('gpu')}"
        if line.split("s ", 1)[1] != (last or "").split("s ", 1)[-1]:
            print(line, flush=True)
            last = line
        if st["ip"] and st["port"]:
            r = subprocess.run(["ssh", "-i", key, "-p", str(st["port"]), "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
                                f"{p.user}@{st['ip']}", "true"], capture_output=True)
            if r.returncode == 0:
                return st["ip"], st["port"]
            err = (r.stderr or b"").decode(errors="replace").strip().splitlines()
            msg = err[-1] if err else f"exit {r.returncode}"
            if msg != getattr(wait_ssh, "_last_err", None):  # log each new ssh failure once, so a dead host is diagnosable
                print(f"[gpu] ssh not ready: {msg}", flush=True)
                wait_ssh._last_err = msg
            # a running sshd that keeps refusing the key will not start accepting it: give up early instead of paying
            # for the whole deadline (two hosts in a row did exactly this for 30 minutes)
            if "Permission denied" in msg:
                denied_since = denied_since or time.time()
                if time.time() - denied_since > 600:
                    raise RuntimeError("sshd refuses the key for 10 minutes: the key was not provisioned")
        time.sleep(15)
    raise RuntimeError("instance did not become reachable in time")


def rsync(key: str, port: int, src: str, dst: str, excludes=(), attempts: int = 4) -> None:
    """rsync with partial-transfer resume and retries: a multi-gigabyte upload over a home uplink drops now and then,
    and a dropped sync must not cost the session."""
    # --update: never replace a newer file on the receiving side, so two concurrent sessions that both carry an
    # earlier result directory cannot roll back each other's fresher copies on the workstation
    cmd = ["rsync", "-az", "--update", "--partial", "--timeout=120", "-e", f"ssh -i {key} -p {port} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20"] \
        + [f"--exclude={e}" for e in excludes] + [src, dst]
    for i in range(attempts):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return
        print(f"[gpu] rsync attempt {i + 1}/{attempts} failed with code {r.returncode}; retrying", flush=True)
        time.sleep(20)
    raise subprocess.CalledProcessError(r.returncode, cmd)


def run_chain(p: Provider, iid: str, ip: str, port: int, key: str, deadline: float, tasks: str = "", fetch_models: str | None = None) -> None:
    """Run the chain on a prepared host under a deadline, fetching results (and models) every few minutes and once more
    at the end, then the host's logs. The caller terminates the host."""
    ssh = ["ssh", "-i", key, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", f"{p.user}@{ip}"]
    remaining = int(deadline - time.time() - 600)
    chain = tasks or "bash scripts/host_chain.sh cuda 8"  # --tasks names the whole remote command, e.g. "bash scripts/gpu/h100_chain.sh"
    cmd = f"cd onepass && timeout {max(remaining, 60)} {chain} 2>&1 | tee -a logs/remote_chain.log"
    (REPO / "results").mkdir(exist_ok=True)
    proc = subprocess.Popen(ssh + [f"bash -lc '{cmd}'"])
    # training checkpoints stay on the host: ckpt.pt, and best.pt (the best-dev weights, rewritten while a run trains and
    # deleted when it ends; a fetch mid-run copied a stale 0.6 to 1.5 GB file per run, or failed reading one mid-write)
    model_ex = ("*.onnx", "compiled4", "compiled8", "ckpt.pt", "best.pt")

    def pull_models():
        if fetch_models:
            Path(fetch_models).mkdir(parents=True, exist_ok=True)
            rsync(key, port, f"{p.user}@{ip}:onepass/models/", str(Path(fetch_models)) + "/", model_ex)

    # fetch results (and models, when asked) while the chain runs, so a dead host or session never costs finished work
    while proc.poll() is None:
        for _ in range(FETCH_EVERY_S // 5):
            if proc.poll() is not None:
                break
            time.sleep(5)
        try:
            rsync(key, port, f"{p.user}@{ip}:onepass/results/", str(REPO / "results") + "/")
            pull_models()
            print(f"[gpu] {time.strftime('%H:%M')} results fetched", flush=True)
        except subprocess.CalledProcessError as e:
            print("[gpu] periodic fetch failed:", e, flush=True)
    rsync(key, port, f"{p.user}@{ip}:onepass/results/", str(REPO / "results") + "/")
    pull_models()
    if fetch_models:
        print(f"[gpu] models fetched into {fetch_models}", flush=True)
    rsync(key, port, f"{p.user}@{ip}:onepass/logs/", str(REPO / "logs" / f"gpu_{iid}") + "/")
    subprocess.run(ssh + ["nvidia-smi"], capture_output=True)


def session(p: Provider, hours: float, key: str, usd_per_hour: float, tasks: str = "", dry: bool = False, extra_excludes=(),
            fetch_models: str | None = None, upload: tuple = (), disk_gb: int | None = None, data_centers: tuple = (),
            cuda_versions: tuple = ()) -> None:
    """Launch, sync the repo, set up, run `tasks` on the host, fetch results every few minutes, terminate. `fetch_models`: a local directory that receives the host's models/ (weights, head, config, tokenizer; no ONNX
    exports, compiled kernels or training checkpoints) with every results fetch and before termination."""
    pub = Path(key + ".pub").read_text()
    name = f"onepass-{int(time.time())}"
    iid = p.launch(pub, name, **({"disk_gb": disk_gb} if disk_gb else {}), **({"data_centers": data_centers} if data_centers else {}),
                   **({"cuda_versions": cuda_versions} if cuda_versions else {}))
    print(f"[gpu] launched {p.name} {iid}", flush=True)
    ledger_open(p.name, iid, hours, usd_per_hour)
    deadline = time.time() + hours * 3600
    try:
        ip, port = wait_ssh(p, iid, key)
        print(f"[gpu] reachable at {ip}:{port}", flush=True)
        ssh = ["ssh", "-i", key, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", f"{p.user}@{ip}"]
        # the official images ship python3.11 but not rsync; install whatever is missing before the sync
        subprocess.run(ssh + ["bash -lc '(command -v python3.11 >/dev/null && command -v rsync >/dev/null) || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.11 python3.11-venv rsync >/dev/null); command -v rsync && (nvidia-smi -L 2>/dev/null || echo no-gpu); nproc'"], check=True)
        ex = (".venv", ".git", "docs", ".claude", ".research-kit", "logs", "results", "__pycache__", "*.onnx",
              ".env", "CLAUDE.md", "*.pem") + tuple(extra_excludes)  # never ship secrets or private notes to a rented host
        rsync(key, port, str(REPO) + "/", f"{p.user}@{ip}:onepass/", ex)  # code, pools' manifests, models/ (fine-tuned weights), data/pools
        # results already fetched from an earlier session on this host class go back up, so the chain resumes cell by cell
        prior = [d for d in (REPO / "results" / "latency").glob("x86_64-linux-*") if d.is_dir()] + \
                [d for d in (REPO / "results" / "path_scores").glob("*-x86_64-linux") if d.is_dir()]
        if prior:
            subprocess.run(ssh + ["mkdir -p onepass/results/latency onepass/results/path_scores onepass/logs"], check=True)
        for d in prior:
            rsync(key, port, str(d), f"{p.user}@{ip}:onepass/results/{d.parent.name}/")
        if prior:
            print(f"[gpu] uploaded {len(prior)} prior result dirs for resume", flush=True)
        for rel in upload:  # individual files the chain needs from the (otherwise excluded) results tree
            subprocess.run(ssh + [f"mkdir -p onepass/{Path(rel).parent}"], check=True)
            src = REPO / rel  # a directory is synced as its contents, so onepass/<rel> mirrors it rather than nesting it
            # the main sync's excludes apply here too: a model directory must not carry its ONNX exports or compiled
            # kernels to a host that never uses them (2 GB instead of 0.6 GB for one base model)
            rsync(key, port, str(src) + ("/" if src.is_dir() else ""), f"{p.user}@{ip}:onepass/{rel}", ex + ("compiled4", "compiled8", "ckpt.pt", "best.pt"))
        if upload:
            print(f"[gpu] uploaded {len(upload)} extra paths", flush=True)
        # MODELS=0: the pinned-model download in setup.sh includes the 26 GB System 2 conversion the GPU host never uses;
        # the encoders and the decoder baseline are fetched on first use at their pinned revisions
        subprocess.run(ssh + ["bash -lc 'cd onepass && MODELS=0 bash scripts/env/setup.sh && .venv/bin/python scripts/env/pin_models.py --from-lock --no-download'"], check=True)
        run_chain(p, iid, ip, port, key, deadline, tasks, fetch_models)
    finally:
        print(f"[gpu] terminating {iid}", flush=True)
        for _ in range(5):
            try:
                p.terminate(iid)
                break
            except Exception as e:  # keep trying: an orphaned GPU is the one failure that costs money
                print("[gpu] terminate failed, retrying:", e)
                time.sleep(20)
        ledger_close(iid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["availability", "launch", "status", "terminate", "session", "attach", "ledger"])
    ap.add_argument("--provider", default="runpod", choices=["runpod", "runpod-cpu", "lambda"])
    ap.add_argument("--flavor", default="cpu5c", help="runpod-cpu: cpu flavor id (cpu3c, cpu3g, cpu3m, cpu5c, cpu5g, cpu5m)")
    ap.add_argument("--vcpu", type=int, default=32, help="runpod-cpu: vCPU count (max 32)")
    ap.add_argument("--gpu", default=None, help="runpod gpuTypeId or lambda instance type")
    ap.add_argument("--cloud", default="SECURE")
    ap.add_argument("--id")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--disk", type=int, default=None, help="container disk in GB (default: the provider's)")
    ap.add_argument("--data-centers", nargs="*", default=[], help="runpod: allowed data center ids (avoid a host that fails to come up)")
    ap.add_argument("--cuda-versions", nargs="*", default=[], help="runpod: lowest acceptable host CUDA driver version, e.g. 13.0 (the locked torch needs it)")
    ap.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/id_ed25519"))
    ap.add_argument("--usd-per-hour", type=float, default=None)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--exclude", nargs="*", default=[], help="extra rsync excludes for the upload (models the chain does not use)")
    ap.add_argument("--upload", nargs="*", default=[], help="repo-relative files to upload despite the excludes (inputs the chain needs)")
    ap.add_argument("--fetch-models", default=None, help="local directory that receives the host's models/ during and after the run")
    a = ap.parse_args()
    kw = {}
    if a.provider == "runpod":
        kw = {"gpu": a.gpu or "NVIDIA H100 PCIe", "cloud": a.cloud}
    elif a.provider == "runpod-cpu":
        kw = {"flavor": a.flavor, "vcpu": a.vcpu, "cloud": a.cloud}
    elif a.gpu:
        kw = {"instance_type": a.gpu}
    if a.cmd == "ledger":
        d = ledger_load()
        print(json.dumps({"cap_usd": d["cap_usd"], "spent_usd": round(ledger_spent(d), 2), "sessions": len(d["sessions"])}, indent=1))
        sys.exit(0)
    p = make(a.provider, **kw)
    if a.provider == "runpod-cpu":
        price = a.usd_per_hour or {"cpu3c": 0.03, "cpu3g": 0.04, "cpu3m": 0.055, "cpu5c": 0.035, "cpu5g": 0.046, "cpu5m": 0.065}.get(a.flavor, 0.05) * a.vcpu
    else:
        price = a.usd_per_hour or PRICES.get((a.provider, kw.get("gpu", kw.get("instance_type", "")), kw.get("cloud", "")), 3.5)
    if a.cmd == "availability":
        print(json.dumps(p.availability(), indent=1))
    elif a.cmd == "launch":
        iid = p.launch(Path(a.ssh_key + ".pub").read_text(), f"onepass-{int(time.time())}")
        ledger_open(a.provider, iid, a.hours, price)
        print(iid)
    elif a.cmd == "status":
        print(json.dumps(p.status(a.id), indent=1, default=str))
    elif a.cmd == "terminate":
        p.terminate(a.id)
        ledger_close(a.id)
        print("terminated", a.id)
    elif a.cmd == "session":
        session(p, a.hours, a.ssh_key, price, a.tasks, extra_excludes=a.exclude, fetch_models=a.fetch_models, upload=tuple(a.upload), disk_gb=a.disk, data_centers=tuple(a.data_centers), cuda_versions=tuple(a.cuda_versions))
    elif a.cmd == "attach":
        # a host already synced and set up (its session's local process died): run the chain with the same periodic
        # fetches, then terminate it; --hours is the remaining budget
        st = p.status(a.id)
        try:
            run_chain(p, a.id, st["ip"], st["port"], a.ssh_key, time.time() + a.hours * 3600, a.tasks, a.fetch_models)
        finally:
            print(f"[gpu] terminating {a.id}", flush=True)
            for _ in range(5):
                try:
                    p.terminate(a.id)
                    break
                except Exception as e:
                    print("[gpu] terminate failed, retrying:", e)
                    time.sleep(20)
            ledger_close(a.id)
