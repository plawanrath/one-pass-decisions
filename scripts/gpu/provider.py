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
        self.gpu_priority = [gpu] + [g for g in ("NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3") if g != gpu]

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
            body = {"name": name, "image": self.IMAGE, "gpu": {"id": g, "count": 1}, "cloud": self.cloud, "disk": disk_gb,
                    "ports": ["22/tcp"], "startSsh": True, "env": {"PUBLIC_KEY": ssh_public_key.strip()}}
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


def make(provider: str, **kw) -> Provider:
    load_env_file()
    if provider == "runpod":
        return RunPod(os.environ["RUNPOD_API_KEY"], **kw)
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
def wait_ssh(p: Provider, iid: str, key: str, timeout_s: int = 900) -> tuple[str, int]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = p.status(iid)
        if st["ip"] and st["port"]:
            r = subprocess.run(["ssh", "-i", key, "-p", str(st["port"]), "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
                                f"{p.user}@{st['ip']}", "true"], capture_output=True)
            if r.returncode == 0:
                return st["ip"], st["port"]
        time.sleep(15)
    raise RuntimeError("instance did not become reachable in time")


def rsync(key: str, port: int, src: str, dst: str, excludes=()) -> None:
    cmd = ["rsync", "-az", "-e", f"ssh -i {key} -p {port} -o StrictHostKeyChecking=accept-new"] + [f"--exclude={e}" for e in excludes] + [src, dst]
    subprocess.run(cmd, check=True)


def session(p: Provider, hours: float, key: str, usd_per_hour: float, tasks: str = "", dry: bool = False) -> None:
    pub = Path(key + ".pub").read_text()
    name = f"onepass-{int(time.time())}"
    iid = p.launch(pub, name)
    print(f"[gpu] launched {p.name} {iid}", flush=True)
    ledger_open(p.name, iid, hours, usd_per_hour)
    deadline = time.time() + hours * 3600
    try:
        ip, port = wait_ssh(p, iid, key)
        print(f"[gpu] reachable at {ip}:{port}", flush=True)
        ssh = ["ssh", "-i", key, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", f"{p.user}@{ip}"]
        subprocess.run(ssh + ["bash -lc 'command -v python3.11 >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.11 python3.11-venv rsync >/dev/null); nvidia-smi -L'"], check=True)
        ex = (".venv", ".git", "docs", ".claude", ".research-kit", "logs", "results", "__pycache__", "*.onnx")
        rsync(key, port, str(REPO) + "/", f"{p.user}@{ip}:onepass/", ex)  # code, pools' manifests, models/ (fine-tuned weights), data/pools
        subprocess.run(ssh + ["bash -lc 'cd onepass && bash scripts/env/setup.sh && .venv/bin/python scripts/env/pin_models.py --from-lock --no-download'"], check=True)
        remaining = int(deadline - time.time() - 600)
        cmd = f"cd onepass && timeout {max(remaining, 60)} bash scripts/host_chain.sh cuda 8 {tasks}"
        subprocess.run(ssh + [f"bash -lc '{cmd}'"], check=False)
        (REPO / "results").mkdir(exist_ok=True)
        rsync(key, port, f"{p.user}@{ip}:onepass/results/", str(REPO / "results") + "/")
        rsync(key, port, f"{p.user}@{ip}:onepass/logs/", str(REPO / "logs" / f"gpu_{iid}") + "/")
        subprocess.run(ssh + ["nvidia-smi"], capture_output=True)
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
    ap.add_argument("cmd", choices=["availability", "launch", "status", "terminate", "session", "ledger"])
    ap.add_argument("--provider", default="runpod", choices=["runpod", "lambda"])
    ap.add_argument("--gpu", default=None, help="runpod gpuTypeId or lambda instance type")
    ap.add_argument("--cloud", default="SECURE")
    ap.add_argument("--id")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/id_ed25519"))
    ap.add_argument("--usd-per-hour", type=float, default=None)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    kw = {}
    if a.provider == "runpod":
        kw = {"gpu": a.gpu or "NVIDIA H100 PCIe", "cloud": a.cloud}
    elif a.gpu:
        kw = {"instance_type": a.gpu}
    if a.cmd == "ledger":
        d = ledger_load()
        print(json.dumps({"cap_usd": d["cap_usd"], "spent_usd": round(ledger_spent(d), 2), "sessions": len(d["sessions"])}, indent=1))
        sys.exit(0)
    p = make(a.provider, **kw)
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
        session(p, a.hours, a.ssh_key, price, a.tasks)
