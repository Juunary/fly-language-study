from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import time

from .protocol import BUDGETS, write_json


class Ledger:
    """Process-safe reservations. GPU-hours are charged once; CPU time stays separate."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def transaction(self):
        lock = self.path.with_suffix(".lock")
        deadline = time.monotonic()+10
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode()); os.close(fd)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Ledger lock busy: {lock}; inspect the owning process before recovering")
                time.sleep(.05)
        try:
            data = json.loads(self.path.read_text()) if self.path.exists() else dict(allocations=dict(BUDGETS), runs={}, transfers=[])
            yield data
            write_json(self.path, data)
        finally:
            lock.unlink()

    def reserve_bundle(self, requests):
        with self.transaction() as data:
            if len({r["run_id"] for r in requests}) != len(requests):
                raise ValueError("Duplicate reservation ids")
            for request in requests:
                if request["run_id"] in data["runs"]:
                    raise ValueError("Reservation already exists")
                if request["category"] not in BUDGETS or request["hours"] <= 0:
                    raise ValueError("Invalid reservation")
            for category, allowance in data["allocations"].items():
                committed = sum(max(r["reserved"], r["used"]) if r["status"] == "reserved" else r["used"]
                                for r in data["runs"].values() if r["category"] == category)
                additional = sum(r["hours"] for r in requests if r["category"] == category)
                if committed + additional > allowance + 1e-9:
                    raise ValueError(f"{category} budget exceeded; reserve complete bundles only")
            for r in requests:
                data["runs"][r["run_id"]] = dict(category=r["category"], reserved=r["hours"], used=0., status="reserved")

    def charge(self, run_id, hours, status):
        if hours < 0 or status not in ("completed", "technical_failure", "budget_interruption"):
            raise ValueError("Invalid accounting entry")
        with self.transaction() as data:
            run = data["runs"][run_id]
            if run["status"] != "reserved":
                raise ValueError("Run already charged; do not double charge")
            run.update(used=hours, status=status, exceeded_reservation=hours > run["reserved"])

    def reservation(self, run_id):
        with self.transaction() as data:
            run = dict(data["runs"][run_id])
            if run["status"] != "reserved":
                raise ValueError("Reservation is not active")
            return run

    def release_auxiliary(self, category):
        order = ("review", "shuffle", "tokenizer")
        with self.transaction() as data:
            if category not in order or any(data["allocations"][c] != 0 for c in order[:order.index(category)]):
                raise ValueError("Release review, then shuffle, then tokenizer")
            if any(r["category"] == category for r in data["runs"].values()):
                raise ValueError("Cannot reassign committed auxiliary funds")
            hours = data["allocations"][category]
            data["allocations"][category] = 0
            data["allocations"]["main"] += hours
            data["transfers"].append(dict(source=category, destination="main", hours=hours))

    def transfer_reserve(self, destination, hours):
        if destination not in BUDGETS or destination == 'reserve' or hours <= 0:
            raise ValueError('Invalid contingency transfer')
        with self.transaction() as data:
            committed=sum(max(r['reserved'],r['used']) if r['status']=='reserved' else r['used']
                          for r in data['runs'].values() if r['category']=='reserve')
            if data['allocations']['reserve']-committed < hours:
                raise ValueError('Insufficient uncommitted contingency budget')
            data['allocations']['reserve']-=hours
            data['allocations'][destination]+=hours
            data['transfers'].append(dict(source='reserve',destination=destination,hours=hours))


def choose_design(candidates):
    feasible = [r for r in candidates if r["measurement_passed"] and r["power_lower"] >= .8
                and r["null_fwer_upper"] <= .06 and r["gpu_hours"]*1.25 <= r["available_main_hours"]
                and r["calendar_hours"]*1.25 <= r["available_calendar_hours"]]
    return min(feasible, key=lambda r: (-r["n"], r["eval_interval"])) if feasible else None
