#!/usr/bin/env python3
import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


CACHE_COLUMNS = [
    "L",
    "nhead",
    "dhead",
    "dbyte",
    "pim_type",
    "power_constraint",
    "cycle",
    "mac",
    "softmax",
    "mvgb",
    "mvsb",
    "wrgb",
]

KEY_COLUMNS = ["L", "nhead", "dhead", "dbyte", "pim_type", "power_constraint"]


def _parse_simple_yaml_map(yaml_text: str) -> dict:
    root = {}
    section = None
    for raw in yaml_text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            root.setdefault(section, {})
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            parsed = value[1:-1]
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        target = root[section] if raw.startswith(" ") and section else root
        target[key] = parsed
    return root


def load_yaml_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except Exception:
        data = _parse_simple_yaml_map(text)
    if not isinstance(data, dict):
        raise ValueError(f"invalid config yaml at {path}")
    return data.get("pregen", data)


def model_num_heads(model_name: str) -> int:
    table = {
        "GPT-175B": 96,
        "GPT-89B": 96,
        "GPT-13B": 40,
        "LLAMA-7B": 32,
        "LLAMA-65B": 64,
        "MT-76B": 40,
        "MT-146B": 80,
        "MT-310B": 128,
        "MT-530B": 128,
        "MT-1008B": 160,
        "OPT-66B": 72,
        "Qwen3-4B": 32,
        "Qwen3-32B": 64,
        "Mistral-Devstral2-123B": 96,
        "Llama-3.1-405B": 128,
    }
    if model_name not in table:
        raise ValueError(f"unsupported model for num_heads lookup: {model_name}")
    return int(table[model_name])


def model_hidden_dim(model_name: str) -> int:
    table = {
        "GPT-175B": 12288,
        "GPT-89B": 12288,
        "GPT-13B": 5120,
        "LLAMA-7B": 4096,
        "LLAMA-65B": 8192,
        "MT-76B": 10240,
        "MT-146B": 12288,
        "MT-310B": 16384,
        "MT-530B": 20480,
        "MT-1008B": 25600,
        "OPT-66B": 9216,
        "Qwen3-4B": 2560,
        "Qwen3-32B": 5120,
        "Mistral-Devstral2-123B": 12288,
        "Llama-3.1-405B": 16384,
    }
    if model_name not in table:
        raise ValueError(f"unsupported model for hidden-dim lookup: {model_name}")
    return int(table[model_name])


def model_attention_dhead(model_name: str) -> int:
    # Must match the runtime layer shape built by Transformer, which uses
    # hdim / num_heads for attention score/context dimensions.
    return int(model_hidden_dim(model_name) / model_num_heads(model_name))


def resolve_dhead(model_name: str, cli_dhead, cfg: Dict) -> int:
    model_dhead = model_attention_dhead(model_name)
    if cli_dhead is not None:
        return int(cli_dhead)

    cfg_dhead = cfg.get("dhead", None)
    if cfg_dhead is None:
        return model_dhead

    cfg_dhead = int(cfg_dhead)
    if cfg_dhead != model_dhead:
        print(
            "[PREGEN] ignoring config dhead={} for model={} and using model-derived dhead={}".format(
                cfg_dhead,
                model_name,
                model_dhead,
            )
        )
        return model_dhead
    return cfg_dhead


def parse_power_modes(raw: str) -> List[bool]:
    out: List[bool] = []
    for token in str(raw).split(","):
        t = token.strip().lower()
        if t in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif t in {"0", "false", "f", "no", "n"}:
            out.append(False)
        elif t == "":
            continue
        else:
            raise ValueError(f"invalid power mode token: {token}")
    if not out:
        raise ValueError("power modes cannot be empty")
    # keep order, remove duplicates
    seen = set()
    dedup = []
    for v in out:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def normalize_power(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"true", "1", "t", "yes", "y"}


def parse_task_mode(raw) -> str:
    mode = str(raw).strip().lower()
    if mode not in {"derived", "direct"}:
        raise ValueError(f"invalid task mode: {raw}")
    return mode


def _split_scalar_or_list(raw) -> List[str]:
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = [raw]
    out: List[str] = []
    for item in items:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _dedupe_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def parse_int_values(raw, field_name: str) -> List[int]:
    tokens = _split_scalar_or_list(raw)
    values: List[int] = []
    for token in tokens:
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"invalid {field_name} range: {token}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0:
                raise ValueError(f"{field_name} range step must be positive: {token}")
            if stop < start:
                raise ValueError(f"{field_name} range stop must be >= start: {token}")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(token))
    values = sorted(set(values))
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    return values


def parse_bool_values(raw, field_name: str) -> List[bool]:
    tokens = _split_scalar_or_list(raw)
    values = [parse_bool(token) for token in tokens]
    values = _dedupe_preserve_order(values)
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    return values


def parse_pim_type_values(raw, field_name: str) -> List[str]:
    tokens = _split_scalar_or_list(raw)
    values: List[str] = []
    for token in tokens:
        pim_type = str(token).strip().upper()
        if pim_type != "BA":
            raise ValueError(
                f"{field_name} currently supports only BA because pregen_ramulator_bank.py uses the bank trace generator"
            )
        values.append(pim_type)
    values = _dedupe_preserve_order(values)
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    return values


def key_from_row(row: Dict[str, str]) -> Tuple[int, int, int, int, str, bool]:
    return (
        int(float(row["L"])),
        int(float(row["nhead"])),
        int(float(row["dhead"])),
        int(float(row["dbyte"])),
        str(row["pim_type"]).strip().upper(),
        normalize_power(row["power_constraint"]),
    )


def read_cache_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            # Skip malformed rows.
            if any(col not in row for col in CACHE_COLUMNS):
                continue
            rows.append({col: row[col] for col in CACHE_COLUMNS})
        return rows


def write_cache_rows_atomic(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CACHE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def merge_rows(existing: List[Dict[str, str]], new_rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    by_key: Dict[Tuple[int, int, int, int, str, bool], Dict[str, str]] = {}
    for row in existing:
        by_key[key_from_row(row)] = row
    for row in new_rows:
        by_key[key_from_row(row)] = row
    merged = list(by_key.values())
    merged.sort(key=lambda r: key_from_row(r))
    return merged


def parse_ramulator_output(stdout: str) -> Dict[str, int]:
    n_cmds = {"mac": 0, "softmax": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
    cycle = 0
    for line in stdout.strip().splitlines():
        if "mac" in line:
            n_cmds["mac"] += int(line.split()[-1])
        elif "softmax_requests" in line:
            n_cmds["softmax"] += int(line.split()[-1])
        elif "move_to_gemv_buffer" in line:
            n_cmds["mvgb"] += int(line.split()[-1])
        elif "move_to_softmax_buffer" in line:
            n_cmds["mvsb"] += int(line.split()[-1])
        elif "write_to_gemv_buffer" in line:
            n_cmds["wrgb"] += int(line.split()[-1])
        elif "memory_system_cycles" in line:
            cycle += int(line.split()[-1])
    return {
        "cycle": cycle,
        "mac": n_cmds["mac"],
        "softmax": n_cmds["softmax"],
        "mvgb": n_cmds["mvgb"],
        "mvsb": n_cmds["mvsb"],
        "wrgb": n_cmds["wrgb"],
    }


def make_yaml_text(trace_path: Path, power_constraint: bool) -> str:
    lines = []
    lines.append("Frontend:")
    lines.append("  impl: PIMLoadStoreTrace")
    lines.append(f"  path: {trace_path}")
    lines.append("  clock_ratio: 1")
    lines.append("")
    lines.append("  Translation:")
    lines.append("    impl: NoTranslation")
    lines.append("    max_addr: 2147483648")
    lines.append("")
    lines.append("MemorySystem:")
    lines.append("  impl: PIMDRAM")
    lines.append("  clock_ratio: 1")
    lines.append("  DRAM:")
    lines.append("    impl: HBM3-PIM")
    lines.append("    org:")
    lines.append("      preset: HBM3_8Gb_2R")
    lines.append("      channel: 16")
    lines.append("    timing:")
    lines.append(f"      preset: {'HBM3_5.2Gbps' if power_constraint else 'HBM3_5.2Gbps_NPC'}")
    lines.append("")
    lines.append("  Controller:")
    lines.append("    impl: HBM3-PIM")
    lines.append("    Scheduler:")
    lines.append("      impl: PIM")
    lines.append("    RefreshManager:")
    lines.append("      impl: AllBankHBM3")
    lines.append("  AddrMapper:")
    lines.append("    impl: HBM3-PIM")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Task:
    L: int
    nhead: int
    dhead: int
    dbyte: int
    pim_type: str
    power_constraint: bool
    maxlen: int

    @property
    def key(self) -> Tuple[int, int, int, int, str, bool]:
        return (self.L, self.nhead, self.dhead, self.dbyte, self.pim_type, self.power_constraint)

    @property
    def basename(self) -> str:
        pc = 1 if self.power_constraint else 0
        pim_suffix = "" if self.pim_type == "BA" else f"_pim{self.pim_type.lower()}"
        return f"attacc_l{self.L}_nattn{self.nhead}_dhead{self.dhead}_dbyte{self.dbyte}_pc{pc}{pim_suffix}"


def run_one_task(
    task: Task,
    python_exe: str,
    trace_gen_script: Path,
    ramulator_bin: Path,
    tmp_trace_dir: Path,
    tmp_yaml_dir: Path,
    cleanup_temp: bool,
    lock: threading.Lock,
) -> Dict[str, str]:
    if task.pim_type != "BA":
        raise ValueError(f"unsupported pim_type for bank pregen: {task.pim_type}")
    trace_path = tmp_trace_dir / f"{task.basename}.trace"
    yaml_path = tmp_yaml_dir / f"{task.basename}.yaml"

    # Generate trace file (bank-level only).
    gen_cmd = [
        python_exe,
        str(trace_gen_script),
        "--dhead",
        str(task.dhead),
        "--nhead",
        str(task.nhead),
        "--seqlen",
        str(task.L),
        "--maxlen",
        str(task.maxlen),
        "--dbyte",
        str(task.dbyte),
        "--output",
        str(trace_path),
    ]
    try:
        subprocess.run(gen_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        out = (exc.stdout or "").strip()
        tail = out[-1200:] if out else "<no stdout>"
        raise RuntimeError(f"trace_gen failed rc={exc.returncode}: {tail}")

    yaml_text = make_yaml_text(trace_path=trace_path, power_constraint=task.power_constraint)
    yaml_path.write_text(yaml_text, encoding="utf-8")

    # Run ramulator and parse counters.
    ram_cmd = [str(ramulator_bin), "-f", str(yaml_path)]
    try:
        result = subprocess.run(ram_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        out = (exc.stdout or "").strip()
        tail = out[-1200:] if out else "<no stdout>"
        raise RuntimeError(f"ramulator failed rc={exc.returncode}: {tail}")
    stats = parse_ramulator_output(result.stdout)

    if cleanup_temp:
        with lock:
            if trace_path.exists():
                trace_path.unlink()
            if yaml_path.exists():
                yaml_path.unlink()

    return {
        "L": str(task.L),
        "nhead": str(task.nhead),
        "dhead": str(task.dhead),
        "dbyte": str(task.dbyte),
        "pim_type": task.pim_type,
        "power_constraint": "True" if task.power_constraint else "False",
        "cycle": str(stats["cycle"]),
        "mac": str(stats["mac"]),
        "softmax": str(stats["softmax"]),
        "mvgb": str(stats["mvgb"]),
        "mvsb": str(stats["mvsb"]),
        "wrgb": str(stats["wrgb"]),
    }


def discover_ramulator_bin(script_dir: Path) -> Path:
    cand = [script_dir.parent / "ramulator2", script_dir.parent / "build" / "ramulator2"]
    for c in cand:
        if c.exists():
            return c
    raise FileNotFoundError("cannot find ramulator2 binary under ramulator2/ or ramulator2/build/")


def build_tasks_derived(
    seqlen_min: int,
    seqlen_max: int,
    batch_min: int,
    batch_max: int,
    dhead: int,
    dbyte: int,
    num_heads: int,
    ngpu: int,
    num_hbm: int,
    power_modes: List[bool],
    maxlen_floor: int,
    existing_keys: Set[Tuple[int, int, int, int, str, bool]],
) -> List[Task]:
    heads_per_device = float(num_heads) / float(ngpu)
    nhead_values = []
    for bs in range(batch_min, batch_max + 1):
        nhead = int(math.ceil((heads_per_device * bs) / float(num_hbm)))
        if nhead > 0:
            nhead_values.append(nhead)
    nhead_values = sorted(set(nhead_values))

    tasks: List[Task] = []
    for L in range(seqlen_min, seqlen_max + 1):
        maxlen = max(maxlen_floor, int(L))
        for nhead in nhead_values:
            for power_mode in power_modes:
                task = Task(
                    L=int(L),
                    nhead=int(nhead),
                    dhead=int(dhead),
                    dbyte=int(dbyte),
                    pim_type="BA",
                    power_constraint=bool(power_mode),
                    maxlen=int(maxlen),
                )
                if task.key in existing_keys:
                    continue
                tasks.append(task)
    return tasks


def build_tasks_direct(
    L_values: List[int],
    nhead_values: List[int],
    dhead_values: List[int],
    dbyte_values: List[int],
    pim_type_values: List[str],
    power_constraint_values: List[bool],
    maxlen_floor: int,
    existing_keys: Set[Tuple[int, int, int, int, str, bool]],
) -> List[Task]:
    tasks: List[Task] = []
    for L in L_values:
        if int(L) <= 0:
            raise ValueError(f"L must be positive: {L}")
        maxlen = max(maxlen_floor, int(L))
        for nhead in nhead_values:
            if int(nhead) <= 0:
                raise ValueError(f"nhead must be positive: {nhead}")
            for dhead in dhead_values:
                if int(dhead) <= 0:
                    raise ValueError(f"dhead must be positive: {dhead}")
                for dbyte in dbyte_values:
                    if int(dbyte) <= 0:
                        raise ValueError(f"dbyte must be positive: {dbyte}")
                    for pim_type in pim_type_values:
                        for power_constraint in power_constraint_values:
                            task = Task(
                                L=int(L),
                                nhead=int(nhead),
                                dhead=int(dhead),
                                dbyte=int(dbyte),
                                pim_type=str(pim_type).upper(),
                                power_constraint=bool(power_constraint),
                                maxlen=int(maxlen),
                            )
                            if task.key in existing_keys:
                                continue
                            tasks.append(task)
    tasks.sort(key=lambda task: task.key)
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Parallel pre-generation of BA ramulator cache entries into ramulator.out",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="pregen_bank.yaml", help="optional YAML config")
    parser.add_argument("--task-mode", type=str, default=None, help="derived=model/ngpu/batch expansion, direct=explicit cache-key tuples")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ngpu", type=int, default=None)
    parser.add_argument("--num-hbm", type=int, default=None)
    parser.add_argument("--batch-min", type=int, default=None)
    parser.add_argument("--batch-max", type=int, default=None)
    parser.add_argument("--seqlen-min", type=int, default=None)
    parser.add_argument("--seqlen-max", type=int, default=None)
    parser.add_argument("--maxlen-floor", type=int, default=None)
    parser.add_argument("--L", dest="direct_L", type=str, default=None, help="direct mode sequence length(s): scalar, comma list, or start:end[:step]")
    parser.add_argument("--nhead", dest="direct_nhead", type=str, default=None, help="direct mode heads per PIM die: scalar, comma list, or start:end[:step]")
    parser.add_argument("--dhead", type=str, default=None)
    parser.add_argument("--dbyte", type=str, default=None)
    parser.add_argument("--pim-type", type=str, default=None, help="direct mode pim type; currently BA only")
    parser.add_argument("--power-constraint", type=str, default=None, help="direct mode power constraint(s): bool/int scalar or comma list")
    parser.add_argument("--power-modes", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=None)
    parser.add_argument("--print-failures", type=int, default=5)
    parser.add_argument("--keep-temp", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--ramulator-out", type=str, default=None)
    parser.add_argument("--tmp-dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    cfg = load_yaml_config(config_path if config_path.exists() else None)

    def pick(key: str, cli_value, default):
        if cli_value is not None:
            return cli_value
        if key in cfg:
            return cfg[key]
        return default

    task_mode = parse_task_mode(pick("task_mode", args.task_mode, "derived"))
    maxlen_floor = int(pick("maxlen_floor", args.maxlen_floor, 4096))
    workers = int(pick("workers", args.workers, 100))
    flush_every = int(pick("flush_every", args.flush_every, 200))
    print_failures = int(pick("print_failures", args.print_failures, 5))
    keep_temp = parse_bool(pick("keep_temp", args.keep_temp, False))
    dry_run = parse_bool(pick("dry_run", args.dry_run, False))

    ramulator_out = Path(args.ramulator_out) if args.ramulator_out else (repo_root / "ramulator.out")
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else (script_dir / "tmp")
    tmp_trace_dir = tmp_dir / "traces"
    tmp_yaml_dir = tmp_dir / "yamls"
    tmp_trace_dir.mkdir(parents=True, exist_ok=True)
    tmp_yaml_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    trace_gen_script = script_dir / "gen_trace_attacc_bank.py"
    ramulator_bin = discover_ramulator_bin(script_dir)

    existing_rows = read_cache_rows(ramulator_out)
    existing_keys = set(key_from_row(r) for r in existing_rows)

    print("[PREGEN] config_path={}".format(config_path if config_path.exists() else "None"))
    print("[PREGEN] task_mode={}".format(task_mode))

    if task_mode == "derived":
        model = str(pick("model", args.model, "GPT-175B"))
        ngpu = int(pick("ngpu", args.ngpu, 8))
        num_hbm = int(pick("num_hbm", args.num_hbm, 8))
        batch_min = int(pick("batch_min", args.batch_min, 1))
        batch_max = int(pick("batch_max", args.batch_max, 16))
        seqlen_min = int(pick("seqlen_min", args.seqlen_min, 1))
        seqlen_max = int(pick("seqlen_max", args.seqlen_max, 8192))
        dhead = resolve_dhead(model, args.dhead, cfg)
        dbyte = int(pick("dbyte", args.dbyte, 2))
        power_modes = parse_power_modes(str(pick("power_modes", args.power_modes, "1")))
        num_heads = model_num_heads(model)
        tasks = build_tasks_derived(
            seqlen_min=seqlen_min,
            seqlen_max=seqlen_max,
            batch_min=batch_min,
            batch_max=batch_max,
            dhead=dhead,
            dbyte=dbyte,
            num_heads=num_heads,
            ngpu=ngpu,
            num_hbm=num_hbm,
            power_modes=power_modes,
            maxlen_floor=maxlen_floor,
            existing_keys=existing_keys,
        )
        print(
            "[PREGEN] model={} num_heads={} ngpu={} num_hbm={} batches=[{},{}] seqlen=[{},{}] maxlen_floor={} power_modes={}".format(
                model,
                num_heads,
                ngpu,
                num_hbm,
                batch_min,
                batch_max,
                seqlen_min,
                seqlen_max,
                maxlen_floor,
                [int(v) for v in power_modes],
            )
        )
    else:
        L_raw = pick("L", args.direct_L, None)
        nhead_raw = pick("nhead", args.direct_nhead, None)
        if L_raw is None:
            raise ValueError("direct mode requires L via --L or pregen.L in config")
        if nhead_raw is None:
            raise ValueError("direct mode requires nhead via --nhead or pregen.nhead in config")

        L_values = parse_int_values(L_raw, "L")
        nhead_values = parse_int_values(nhead_raw, "nhead")
        dhead_values = parse_int_values(pick("dhead", args.dhead, 80), "dhead")
        dbyte_values = parse_int_values(pick("dbyte", args.dbyte, 2), "dbyte")
        pim_type_values = parse_pim_type_values(pick("pim_type", args.pim_type, "BA"), "pim_type")
        power_constraint_values = parse_bool_values(
            pick("power_constraint", args.power_constraint, 1),
            "power_constraint",
        )
        tasks = build_tasks_direct(
            L_values=L_values,
            nhead_values=nhead_values,
            dhead_values=dhead_values,
            dbyte_values=dbyte_values,
            pim_type_values=pim_type_values,
            power_constraint_values=power_constraint_values,
            maxlen_floor=maxlen_floor,
            existing_keys=existing_keys,
        )
        print(
            "[PREGEN] direct L={} nhead={} dhead={} dbyte={} pim_type={} power_constraint={} maxlen_floor={}".format(
                L_values,
                nhead_values,
                dhead_values,
                dbyte_values,
                pim_type_values,
                [int(v) for v in power_constraint_values],
                maxlen_floor,
            )
        )
    print("[PREGEN] ramulator_out={} existing_rows={}".format(ramulator_out, len(existing_rows)))
    print("[PREGEN] tmp_dir={} workers={}".format(tmp_dir, workers))
    print("[PREGEN] todo_tasks={}".format(len(tasks)))
    if dry_run:
        print("[PREGEN] dry-run enabled; no tasks executed")
        return

    lock = threading.Lock()
    new_rows: List[Dict[str, str]] = []
    failures: List[str] = []
    done = 0
    start_ts = time.time()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_to_task = {
            pool.submit(
                run_one_task,
                task=task,
                python_exe=python_exe,
                trace_gen_script=trace_gen_script,
                ramulator_bin=ramulator_bin,
                tmp_trace_dir=tmp_trace_dir,
                tmp_yaml_dir=tmp_yaml_dir,
                cleanup_temp=not keep_temp,
                lock=lock,
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            done += 1
            try:
                row = future.result()
                new_rows.append(row)
            except Exception as exc:
                failures.append(f"{task.basename}: {exc}")
                if len(failures) <= max(0, print_failures):
                    print(f"[PREGEN][FAIL] {failures[-1]}", flush=True)

            if done % max(1, flush_every) == 0 or done == len(tasks):
                merged = merge_rows(existing_rows, new_rows)
                write_cache_rows_atomic(ramulator_out, merged)
                elapsed = max(time.time() - start_ts, 1e-6)
                print(
                    "[PREGEN] progress done={}/{} success={} fail={} elapsed_s={:.1f} rate={:.2f}task/s".format(
                        done,
                        len(tasks),
                        len(new_rows),
                        len(failures),
                        elapsed,
                        done / elapsed,
                    )
                )

    # Final flush.
    merged = merge_rows(existing_rows, new_rows)
    write_cache_rows_atomic(ramulator_out, merged)
    elapsed = max(time.time() - start_ts, 1e-6)
    print(
        "[PREGEN] completed success={} fail={} total={} elapsed_s={:.1f}".format(
            len(new_rows), len(failures), len(tasks), elapsed
        )
    )
    if failures:
        fail_log = tmp_dir / "pregen_failures.log"
        fail_log.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"[PREGEN] failures logged to {fail_log}")


if __name__ == "__main__":
    main()
