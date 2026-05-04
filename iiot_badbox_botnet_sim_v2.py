"""
IIoT BadBox-Style Botnet Simulation v2
========================================
Realistic C2-driven, multi-stage botnet simulation with:
  - Persistent C2 commands (not random each step)
  - Per-device state history (for real visualization)
  - Per-device propagation (not aggregated)
  - Attack target (victim) concept
  - Device roles: scanner / ddos / exfil
  - Stealth (long dormant behavior)
  - Full 5-visualization suite + phase explanations

Lifecycle phases:
  0–40   : Normal baseline
  40–70  : Initial compromise (patient-zero seeding)
  70–110 : Beaconing + stealth (periodic C2 check-ins)
  110–140: Scanning & lateral spread
  140–170: Attack (DDoS) + exfiltration
  170–200: Recovery / remediation

Device states:
  clean → compromised → beaconing → dormant → active → exfiltrating

Device roles (assigned at compromise):
  scanner  – focuses on lateral movement
  ddos     – contributes to flood attacks
  exfil    – prioritizes data theft

Dependencies: pandas, matplotlib, scikit-learn, networkx
Install:      pip install pandas matplotlib scikit-learn networkx
"""

from __future__ import annotations

import math
import random
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — single source of truth
# ─────────────────────────────────────────────────────────────────────────────
CONFIG: Dict = {
    # Simulation
    "duration":          200,
    "num_devices":       30,
    "seed":              42,

    # Baseline device metrics
    "baseline_traffic":     5.0,       # MB per device per step
    "baseline_connections": 12,
    "noise_factor":         0.10,

    # Propagation
    "initial_compromise_frac": 0.10,   # fraction seeded at t=40
    "badbox_preinfect_frac":   0.10,   # BadBox supply-chain pre-infection fraction
    "scan_success_prob":       0.35,
    "spread_prob_per_scan":    0.22,
    "beacon_interval":         3,      # steps between beacons

    # Roles (proportional weights among compromised bots)
    "role_weights": {"scanner": 0.30, "ddos": 0.45, "exfil": 0.25},

    # Stealth: dormant bots have reduced footprint; they reactivate after a delay
    "dormant_min_steps": 8,
    "dormant_max_steps": 20,

    # Exfiltration
    "exfil_volume_per_device": 50.0,   # MB per exfiltrating device per step

    # C2 command persistence: each command lasts this many steps before rotating
    "c2_command_persistence": 6,

    # Isolation Forest
    "if_n_estimators":  150,
    "if_contamination": 0.35,

    # Threat thresholds (normalised anomaly score 0–1)
    "threat_medium": 0.40,
    "threat_high":   0.70,

    # Phase boundaries
    "phase_normal_end":     40,
    "phase_compromise_end": 70,
    "phase_beacon_end":    110,
    "phase_scan_end":      140,
    "phase_attack_end":    170,
    # 170+ → recovery
}

# ML features fed into Isolation Forest
FEATURES = [
    "traffic_mb",
    "connections",
    "c2_beacons",
    "scan_attempts",
    "failed_logins",
    "outbound_traffic_ratio",
    "unique_dst_ips",
    "exfiltration_mb",
]

# ─────────────────────────────────────────────────────────────────────────────
# PHASE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
PHASE_COLORS = {
    "normal":     "#d4edda",
    "compromise": "#fff3cd",
    "beaconing":  "#fde8d8",
    "scanning":   "#fddde6",
    "attack":     "#f5c6cb",
    "recovery":   "#d1ecf1",
}

PHASE_LABELS = {
    "normal":     "① Normal baseline",
    "compromise": "② Initial compromise",
    "beaconing":  "③ Beaconing / stealth",
    "scanning":   "④ Scanning & lateral spread",
    "attack":     "⑤ Attack + exfiltration",
    "recovery":   "⑥ Recovery",
}

PHASE_STARTS = {
    "normal": 0, "compromise": 40, "beaconing": 70,
    "scanning": 110, "attack": 140, "recovery": 170,
}
PHASE_ENDS = {
    "normal": 40, "compromise": 70, "beaconing": 110,
    "scanning": 140, "attack": 170, "recovery": CONFIG["duration"],
}


def get_phase(t: int) -> str:
    p = CONFIG
    if t < p["phase_normal_end"]:      return "normal"
    elif t < p["phase_compromise_end"]: return "compromise"
    elif t < p["phase_beacon_end"]:     return "beaconing"
    elif t < p["phase_scan_end"]:       return "scanning"
    elif t < p["phase_attack_end"]:     return "attack"
    else:                               return "recovery"


def phase_elapsed_fraction(t: int) -> float:
    phase = get_phase(t)
    s, e = PHASE_STARTS[phase], PHASE_ENDS[phase]
    return (t - s) / max(e - s, 1)


# ─────────────────────────────────────────────────────────────────────────────
# C2 SERVER  (persistent commands)
# ─────────────────────────────────────────────────────────────────────────────
# Each command profile: (traffic_mult, conn_mult, scan_rate, exfil_flag)
C2_COMMAND_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "idle":              (1.0,  1.0,  0.0, 0.0),
    "beacon":            (1.1,  1.05, 0.0, 0.0),
    "low_rate_ddos":     (2.5,  1.8,  0.0, 0.0),
    "burst_ddos":        (6.0,  3.5,  0.0, 0.0),
    "data_exfiltration": (4.0,  1.2,  0.0, 1.0),
}

# Commands available per phase (ordered by likelihood)
PHASE_COMMAND_POOL: Dict[str, List[str]] = {
    "normal":     ["idle"],
    "compromise": ["idle", "idle", "beacon"],
    "beaconing":  ["beacon", "beacon", "idle"],
    "scanning":   ["beacon", "idle"],
    "attack":     ["low_rate_ddos", "burst_ddos", "data_exfiltration",
                   "low_rate_ddos", "burst_ddos"],
    "recovery":   ["idle"],
}


class C2Server:
    """
    Central command-and-control entity.

    Commands PERSIST for multiple timesteps (c2_command_persistence steps)
    before the C2 server rotates to a new command.  This matches real
    botnet behaviour where operators issue sustained campaigns.
    """

    def __init__(self) -> None:
        self._bots: Dict[int, str] = {}       # device_id → role
        self._current_command: str = "idle"
        self._command_age: int = 0            # steps since last command change
        self._command_history: List[str] = [] # for reporting
        self.beacon_count: int = 0

    def register_bot(self, device_id: int, role: str) -> None:
        if device_id not in self._bots:
            self._bots[device_id] = role

    def issue_command(self, phase: str, t: int) -> str:
        """
        Return the current persistent command.
        Rotate only when the persistence window expires or the phase changes.
        """
        persistence = CONFIG["c2_command_persistence"]
        pool = PHASE_COMMAND_POOL[phase]

        if self._command_age >= persistence or self._current_command not in pool:
            self._current_command = random.choice(pool)
            self._command_age = 0
        else:
            self._command_age += 1

        self._command_history.append(self._current_command)
        return self._current_command

    def receive_beacon(self) -> None:
        self.beacon_count += 1

    def reset_step(self) -> None:
        self.beacon_count = 0

    def get_role(self, device_id: int) -> Optional[str]:
        return self._bots.get(device_id)

    @property
    def bot_count(self) -> int:
        return len(self._bots)


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE  (per-device state history + role + dormancy timer)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_STATES = ["clean", "compromised", "beaconing", "dormant", "active", "exfiltrating"]

_STATE_COLOR = {
    "clean":        "#2ecc71",
    "compromised":  "#f39c12",
    "beaconing":    "#f1c40f",
    "dormant":      "#95a5a6",
    "active":       "#e74c3c",
    "exfiltrating": "#8e44ad",
    "badbox":       "#1abc9c",   # teal — pre-infected supply-chain device
}


# Botnet type → allowed roles mapping
BOTNET_ROLES: Dict[str, List[str]] = {
    "mirai":    ["scanner", "ddos"],
    "satori":   ["exploit"],
    "persirai": ["iot_camera", "exfil"],
    "badbox":   ["proxy", "fraud"],       # supply-chain pre-infected, abuse-focused
}

# Updated role weights per botnet type
BOTNET_ROLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "mirai":    {"scanner": 0.45, "ddos": 0.55},
    "satori":   {"exploit": 1.0},
    "persirai": {"iot_camera": 0.50, "exfil": 0.50},
    "badbox":   {"proxy": 0.60, "fraud": 0.40},
}

# Spread probability multiplier per botnet type (Satori spreads faster)
BOTNET_SPREAD_MULT: Dict[str, float] = {
    "mirai":    1.0,
    "satori":   1.80,   # fast-spreading exploit botnet
    "persirai": 0.65,   # low scan, stealthy
    "badbox":   0.0,    # supply-chain only — never spreads via scanning
}

# Botnet type weights for random assignment at compromise
# BadBox is NOT included here: it is pre-seeded at device init, not spread
BOTNET_TYPE_WEIGHTS = {"mirai": 0.45, "satori": 0.30, "persirai": 0.25}


@dataclass
class Device:
    id: int
    state: str = "clean"
    role: Optional[str] = None          # scanner | ddos | exfil | exploit | iot_camera
    botnet_type: Optional[str] = None   # mirai | satori | persirai

    # Per-device baseline (heterogeneous fleet)
    base_traffic:     float = field(default=0.0)
    base_connections: int   = field(default=0)

    # Per-device state history (list of state at each timestep)
    state_history: List[str] = field(default_factory=list)

    # Dormancy management
    _dormant_steps_remaining: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.base_traffic     = random.uniform(3.5, 6.5)
        self.base_connections = random.randint(8, 16)

    # ── helpers ──────────────────────────────────────────────────────────────

    def is_bot(self) -> bool:
        return self.state != "clean"

    def noise(self, value: float, factor: float = CONFIG["noise_factor"]) -> float:
        return value * (1.0 + random.uniform(-factor, factor))

    def assign_role(self, forced_botnet: Optional[str] = None) -> None:
        """Legacy shim — delegates to assign_botnet_and_role."""
        self.assign_botnet_and_role(forced_botnet=forced_botnet)

    def assign_botnet_and_role(self, forced_botnet: Optional[str] = None) -> None:
        """
        Assign botnet type and a role consistent with that type when first compromised.
          Mirai    -> scanner, ddos
          Satori   -> exploit  (fast spread, high scans)
          Persirai -> iot_camera, exfil  (low scan, high outbound)

        forced_botnet: when provided (e.g. "mirai"), skips random selection and
        locks this compromise to that single botnet type.  Driven by the
        simulation mode ("Mirai Only", "Satori Only", "Persirai Only").
        """
        if forced_botnet and forced_botnet in BOTNET_ROLE_WEIGHTS:
            self.botnet_type = forced_botnet
        else:
            bt_types = list(BOTNET_TYPE_WEIGHTS.keys())
            bt_probs = [BOTNET_TYPE_WEIGHTS[b] for b in bt_types]
            self.botnet_type = random.choices(bt_types, weights=bt_probs)[0]

        role_map = BOTNET_ROLE_WEIGHTS[self.botnet_type]
        roles = list(role_map.keys())
        probs = [role_map[r] for r in roles]
        self.role = random.choices(roles, weights=probs)[0]

    # ── state machine ─────────────────────────────────────────────────────────

    def advance_state(self, c2_command: str, phase: str, t: int) -> None:
        """
        Progress device through the botnet lifecycle.

        Key improvements over v1:
          - Dormancy is timer-based (reactivates after dormant_min/max steps)
          - Role influences which command the device acts on
          - Recovery is probabilistic but accelerates over time
          - BadBox devices stay active permanently (firmware-level — not remediable)
        """
        # BadBox: firmware-level implant — immune to standard recovery,
        # always stays active, never enters normal botnet lifecycle.
        if self.botnet_type == "badbox":
            self.state = "active"
            self.state_history.append(self.state)
            return
        if self.state == "compromised":
            if phase in ("beaconing", "scanning", "attack"):
                if random.random() < 0.60:
                    self.state = "beaconing"

        elif self.state == "beaconing":
            if phase == "scanning" and random.random() < 0.18:
                # Enter dormancy — stealth period
                self.state = "dormant"
                self._dormant_steps_remaining = random.randint(
                    CONFIG["dormant_min_steps"], CONFIG["dormant_max_steps"]
                )
            elif phase == "attack" and random.random() < 0.75:
                self.state = "active"

        elif self.state == "dormant":
            # Count down dormancy timer
            if self._dormant_steps_remaining > 0:
                self._dormant_steps_remaining -= 1
            else:
                # Reactivate when timer expires
                if phase == "attack":
                    self.state = "active"
                elif phase in ("scanning", "beaconing"):
                    self.state = "beaconing"

        elif self.state == "active":
            # Role-gated exfiltration: only exfil-role bots exfiltrate
            if (c2_command == "data_exfiltration"
                    and self.role == "exfil"
                    and random.random() < 0.80):
                self.state = "exfiltrating"

        elif self.state == "exfiltrating":
            if phase == "recovery" and random.random() < 0.22:
                self.state = "clean"
                self.role = None

        # Recovery: gradual remediation across all bot states
        if phase == "recovery" and self.state != "clean":
            if random.random() < 0.16:
                self.state = "clean"
                self.role = None

        self.state_history.append(self.state)

    # ── metrics ───────────────────────────────────────────────────────────────

    def generate_metrics(
        self, c2_command: str, phase: str, t: int
    ) -> Dict[str, float]:
        """
        Return per-device network metrics shaped by state, role, and C2 command.
        Role matters:
          scanner  → higher scan_attempts
          ddos     → higher traffic multiplier on active
          exfil    → higher exfil_mb, lower dst_ips
        """
        profile = C2_COMMAND_PROFILES[c2_command]
        tm, cm, _, er = profile

        traffic      = self.noise(self.base_traffic)
        connections  = int(self.noise(self.base_connections))
        scan_attempts = 0
        failed_logins = 0
        dst_ips       = random.randint(1, 4)
        exfil_mb      = 0.0
        outbound      = random.uniform(0.3, 0.5)

        frac = phase_elapsed_fraction(t)

        if self.state == "clean":
            pass

        elif self.state == "compromised":
            traffic *= 1.05  # subtle foothold

        elif self.state == "beaconing":
            # Periodic spikes simulating check-in intervals
            beacon_spike = 1.0 + 0.5 * math.sin(frac * 2 * math.pi)
            traffic  *= 1.1 * beacon_spike
            outbound  = random.uniform(0.50, 0.65)

        elif self.state == "dormant":
            traffic  *= 0.92  # minimal footprint — stealth
            outbound  = random.uniform(0.25, 0.40)

        elif self.state == "active":
            # ── Traffic shaped by BOTH role AND botnet_type ──────────────────
            traffic *= tm
            connections = int(connections * cm)

            if c2_command in ("low_rate_ddos", "burst_ddos"):
                if self.botnet_type == "mirai":
                    # Mirai: strongest DDoS — dominant volumetric attacker
                    dst_ips  = random.randint(30, 80)
                    outbound = random.uniform(0.80, 0.95)
                    if self.role == "ddos":
                        traffic *= random.uniform(1.4, 1.8)      # DDoS role amplifier
                    if c2_command == "burst_ddos":
                        traffic *= random.uniform(1.5, 3.2)      # Mirai burst spike

                elif self.botnet_type == "satori":
                    # Satori: moderate traffic — spread-focused, not a raw DDoS botnet
                    dst_ips  = random.randint(40, 120)            # high dst_ips (worm reach)
                    outbound = random.uniform(0.60, 0.78)
                    traffic *= random.uniform(0.7, 1.2)           # reduced vs Mirai
                    if c2_command == "burst_ddos":
                        traffic *= random.uniform(0.8, 1.3)       # clearly weaker than Mirai

                elif self.botnet_type == "persirai":
                    # Persirai: very weak DDoS — primarily an exfil / camera botnet
                    dst_ips  = random.randint(3, 15)
                    outbound = random.uniform(0.55, 0.70)
                    traffic *= random.uniform(0.5, 0.85)          # significantly weaker
                    if c2_command == "burst_ddos":
                        traffic *= random.uniform(0.6, 0.95)      # near-negligible burst

                elif self.botnet_type == "badbox":
                    # BadBox: proxy/ad-fraud bot — weak DDoS, mostly passthrough traffic
                    dst_ips  = random.randint(5, 20)
                    outbound = random.uniform(0.70, 0.85)
                    traffic *= random.uniform(0.4, 0.75)   # low footprint, no DDoS spike
                    if c2_command == "burst_ddos":
                        traffic *= random.uniform(0.5, 0.80)  # almost invisible in floods

                else:
                    dst_ips  = random.randint(20, 60)
                    outbound = random.uniform(0.70, 0.92)

            else:
                # Non-DDoS command — Satori still probes many IPs; Persirai stays quiet
                if self.botnet_type == "satori":
                    dst_ips  = random.randint(40, 120)
                    outbound = random.uniform(0.55, 0.70)
                elif self.botnet_type == "persirai":
                    dst_ips  = 1                                   # only C2 endpoint
                    outbound = random.uniform(0.70, 0.85)          # consistently high out
                elif self.botnet_type == "badbox":
                    # BadBox idle: many small connections for proxy/fraud routing
                    dst_ips  = random.randint(5, 20)
                    outbound = random.uniform(0.82, 0.95)          # very high outbound (proxy)
                    connections = int(connections * random.uniform(1.5, 2.5))  # many small conns

        elif self.state == "exfiltrating":
            # Persirai: continuous smooth exfil, single C2, stealthy low-noise traffic
            if self.botnet_type == "persirai":
                exfil_mb    = self.noise(CONFIG["exfil_volume_per_device"] * 1.5, 0.08)
                outbound    = random.uniform(0.85, 0.98)           # 85–98% outbound
                dst_ips     = 1                                     # single C2 endpoint
                traffic    += exfil_mb * 0.85                      # subtle — hard to detect
            elif self.botnet_type == "badbox":
                # BadBox: continuous low-volume exfil — ad fraud data, device fingerprints
                exfil_mb    = random.uniform(2.0, 8.0)             # small but constant
                outbound    = random.uniform(0.85, 0.98)           # very high outbound
                dst_ips     = random.randint(5, 20)                # multiple fraud endpoints
                traffic     = random.uniform(1.0, 3.0)             # very flat — hides in baseline
                connections = int(connections * random.uniform(1.8, 3.0))  # many tiny conns
            else:
                exfil_mb    = self.noise(CONFIG["exfil_volume_per_device"], 0.15)
                outbound    = random.uniform(0.82, 0.96)
                dst_ips     = random.randint(1, 2)
                traffic    += exfil_mb
            connections = int(connections * 0.55)                  # fewer but persistent

        # ── Botnet-specific scanning behavior ────────────────────────────────
        if phase in ("scanning", "attack") and self.state in (
            "beaconing", "active", "exfiltrating"
        ):
            if self.botnet_type == "badbox":
                # BadBox: supply-chain pre-infected — almost zero active scanning
                scan_attempts = random.randint(0, 2)
            elif self.botnet_type == "satori" and self.role == "exploit":
                # Satori: very high scan rate — fast worm-like spread signature
                scan_attempts = random.randint(20, 50)
            elif self.botnet_type == "mirai" and self.role == "scanner":
                # Mirai scanner: moderate telnet-style scan rate
                scan_attempts = random.randint(10, 25)
            elif self.botnet_type == "persirai":
                # Persirai: minimal scanning — stealthy, almost no probing
                scan_attempts = random.randint(0, 5)
            elif self.role in ("scanner", "exploit"):
                scan_attempts = random.randint(8, 20)
            else:
                scan_attempts = random.randint(2, 10)

            failed_logins = sum(
                1 for _ in range(scan_attempts)
                if random.random() > CONFIG["scan_success_prob"]
            )

        return {
            "traffic":      max(traffic, 0.1),
            "connections":  max(connections, 1),
            "scan_attempts": scan_attempts,
            "failed_logins": failed_logins,
            "outbound":     outbound,
            "dst_ips":      dst_ips,
            "exfil_mb":     exfil_mb,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ATTACK TARGET (victim system)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AttackTarget:
    """
    Represents a victim web server / infrastructure being targeted.
    Tracks cumulative DDoS load and exfil data received.
    """
    name: str = "VictimServer"
    ddos_load: float = 0.0        # cumulative DDoS load (MB)
    exfil_received: float = 0.0   # cumulative exfil data sent to C2 (MB)
    degraded: bool = False        # True when under heavy DDoS

    DDOS_THRESHOLD = 500.0        # MB before service degrades

    def receive_ddos(self, traffic_mb: float) -> None:
        self.ddos_load += traffic_mb
        if self.ddos_load > self.DDOS_THRESHOLD:
            self.degraded = True

    def receive_exfil(self, exfil_mb: float) -> None:
        self.exfil_received += exfil_mb

    def reset_step(self) -> None:
        """Partial recovery between steps if not under sustained attack."""
        self.ddos_load = max(0.0, self.ddos_load * 0.85)
        if self.ddos_load < self.DDOS_THRESHOLD * 0.5:
            self.degraded = False


# ─────────────────────────────────────────────────────────────────────────────
# PROPAGATION (per-device, not aggregated)
# ─────────────────────────────────────────────────────────────────────────────
def seed_initial_compromise(
    devices: List[Device], c2: C2Server, forced_botnet: Optional[str] = None
) -> None:
    n = max(1, int(len(devices) * CONFIG["initial_compromise_frac"]))
    chosen = random.sample(devices, n)
    for d in chosen:
        d.state = "compromised"
        d.assign_botnet_and_role(forced_botnet=forced_botnet)
        c2.register_bot(d.id, d.role)


def lateral_spread(
    devices: List[Device], c2: C2Server, phase: str, forced_botnet: Optional[str] = None
) -> None:
    """
    Per-device scanning: each active scanner tries to compromise a clean neighbor.

    Spreading roles (ONLY these can spread):
      - "scanner"  (Mirai)   -- moderate spread, 3 attempts
      - "exploit"  (Satori)  -- fast spread, 4-6 attempts, higher probability

    Non-spreading roles (ddos, exfil, iot_camera) do NOT spread.

    Persirai bots: very limited spread regardless of role (1 attempt, ~50% prob).

    Botnet competition: if a target is already infected, a small probability
    (~20%) allows the attacker to take it over, replacing its botnet_type and role.
    Competition is disabled in single-botnet modes (all devices share same type).

    forced_botnet: when set, every newly infected device is locked to that type.
    """
    clean_devices = [d for d in devices if d.state == "clean"]
    all_targets   = [d for d in devices if d.state != "clean"]  # for takeover

    # ONLY "scanner" (Mirai) and "exploit" (Satori) roles can propagate.
    # "ddos", "exfil", and "iot_camera" roles never spread.
    scanners = [
        d for d in devices
        if d.state in ("beaconing", "active") and d.role in ("scanner", "exploit")
    ]

    if not scanners:
        return

    for scanner in scanners:
        botnet      = scanner.botnet_type or "mirai"

        # BadBox never spreads via scanning — supply-chain infection only
        if botnet == "badbox":
            continue

        spread_mult = BOTNET_SPREAD_MULT.get(botnet, 1.0)

        # Attempt count + probability differentiated by botnet type
        if botnet == "satori":
            n_attempts  = random.randint(4, 6)
            spread_prob = min(1.0, CONFIG["spread_prob_per_scan"] * spread_mult)
        elif botnet == "persirai":
            n_attempts  = 1
            spread_prob = min(1.0, CONFIG["spread_prob_per_scan"] * 0.50)
        else:
            n_attempts  = 3
            spread_prob = min(1.0, CONFIG["spread_prob_per_scan"] * spread_mult)

        # Infect clean devices
        if clean_devices:
            targets = random.sample(clean_devices, min(n_attempts, len(clean_devices)))
            for target in targets:
                if random.random() < spread_prob:
                    target.state = "compromised"
                    target.assign_botnet_and_role(forced_botnet=forced_botnet)
                    c2.register_bot(target.id, target.role)
                    if target in clean_devices:
                        clean_devices.remove(target)

        # Botnet competition: disabled in single-botnet modes (nothing to compete with)
        if forced_botnet is None and all_targets and random.random() < 0.15:
            victim = random.choice(all_targets)
            if victim.botnet_type != botnet and random.random() < 0.20:
                victim.botnet_type = botnet
                role_map   = BOTNET_ROLE_WEIGHTS[botnet]
                roles      = list(role_map.keys())
                probs      = [role_map[r] for r in roles]
                victim.role = random.choices(roles, weights=probs)[0]
                c2.register_bot(victim.id, victim.role)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
class Simulation:
    """
    Orchestrates the multi-stage botnet lifecycle.

    Improvements over v1:
      - Per-device state history stored in Device.state_history
      - Persistent C2 commands via C2Server
      - Per-device lateral spread (not aggregated)
      - AttackTarget tracks DDoS load and exfil received
      - Device roles drive differentiated behavior
    """
    def __init__(
        self,
        seed: int = CONFIG["seed"],
        selected_botnet: str = "All (Realistic)",
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)

        # Resolve the forced botnet string once — None means "All (Realistic)"
        _MODE_TO_TYPE = {
            "Mirai Only":    "mirai",
            "Satori Only":   "satori",
            "Persirai Only": "persirai",
            "BadBox Only":   "badbox",
        }
        self.selected_botnet = selected_botnet
        self._forced_botnet: Optional[str] = _MODE_TO_TYPE.get(selected_botnet, None)

        self.devices  = [Device(i) for i in range(CONFIG["num_devices"])]

        # ── BadBox pre-infection: supply-chain style ──────────────────────────
        # In "All (Realistic)" or "BadBox Only" modes, ~10% of devices (or 100%
        # in BadBox Only) are pre-compromised as supply-chain firmware implants.
        if self._forced_botnet in (None, "badbox"):
            frac = 1.0 if self._forced_botnet == "badbox" else CONFIG.get("badbox_preinfect_frac", 0.10)
            for d in self.devices:
                if random.random() < frac:
                    d.botnet_type = "badbox"
                    d.role        = random.choice(["proxy", "fraud"])
                    d.state       = "active"   # already operational from day 0

        self.c2       = C2Server()
        self.target   = AttackTarget()
        self._seeded  = False

    def step(self, t: int):
        return self._step(t)

    def reset(self, selected_botnet: Optional[str] = None) -> None:
        """Re-initialise the simulation, optionally with a new botnet mode."""
        mode = selected_botnet if selected_botnet is not None else self.selected_botnet
        self.__init__(seed=CONFIG["seed"], selected_botnet=mode)


    def run(self) -> pd.DataFrame:
        records = []
        for t in range(CONFIG["duration"]):
            records.append(self._step(t))
        return pd.DataFrame(records)

    def _step(self, t: int) -> Dict:
        phase      = get_phase(t)
        prev_phase = get_phase(t - 1) if t > 0 else "normal"

        # ── Phase transitions ──────────────────────────────────────────────
        if phase == "compromise" and not self._seeded:
            seed_initial_compromise(self.devices, self.c2, forced_botnet=self._forced_botnet)
            self._seeded = True

        # ── Persistent C2 command ──────────────────────────────────────────
        command = self.c2.issue_command(phase, t)
        self.c2.reset_step()
        self.target.reset_step()

        # ── Advance each device state ──────────────────────────────────────
        for d in self.devices:
            d.advance_state(command, phase, t)

        # ── Beaconing (staggered by device ID for realism) ─────────────────
        # ── Beaconing (probabilistic with jitter for realism) ──────────────
        beacon_interval = CONFIG["beacon_interval"]

        beaconing_bots = [
            d for d in self.devices
            if d.state in ("beaconing", "active", "exfiltrating")
        ]

        beacons = 0
        for d in beaconing_bots:
            # Base probability (avg 1 beacon every interval)
            prob = 1.0 / beacon_interval

            # Add jitter (random variation)
            jitter = random.uniform(0.7, 1.3)

            final_prob = min(1.0, prob * jitter)

            if random.random() < final_prob:
                beacons += 1

        # During attack → near-synchronized beaconing (but not perfect)
        if phase == "attack":
            beacons = sum(
                1 for _ in beaconing_bots
                if random.random() < 0.85
            )

        for _ in range(beacons):
            self.c2.receive_beacon()         

        # ── Per-device lateral spread ──────────────────────────────────────
        if phase in ("scanning", "attack"):
            lateral_spread(self.devices, self.c2, phase, forced_botnet=self._forced_botnet)

        # ── Aggregate metrics ──────────────────────────────────────────────
        total_traffic = total_exfil = total_outbound = 0.0
        total_connections = total_scans = total_failed = total_dst_ips = 0

        for d in self.devices:
            m = d.generate_metrics(command, phase, t)
            total_traffic     += m["traffic"]
            total_connections += m["connections"]
            total_scans       += m["scan_attempts"]
            total_failed      += m["failed_logins"]
            total_exfil       += m["exfil_mb"]
            total_dst_ips     += m["dst_ips"]
            total_outbound    += m["outbound"]

        # Feed into attack target
        if phase == "attack":
            if command in ("low_rate_ddos", "burst_ddos"):
                self.target.receive_ddos(total_traffic * 0.5)
            if command == "data_exfiltration":
                self.target.receive_exfil(total_exfil)

        n = CONFIG["num_devices"]
        outbound_ratio = round(total_outbound / n, 4)

        # ── State census ──────────────────────────────────────────────────
        state_counts: Dict[str, int] = {s: 0 for s in DEVICE_STATES}
        role_counts:  Dict[str, int] = {"scanner": 0, "ddos": 0, "exfil": 0, "exploit": 0, "iot_camera": 0, "proxy": 0, "fraud": 0}
        botnet_counts: Dict[str, int] = {"mirai": 0, "satori": 0, "persirai": 0, "badbox": 0}
        for d in self.devices:
            state_counts[d.state] += 1
            if d.role:
                role_counts[d.role] = role_counts.get(d.role, 0) + 1
            if d.botnet_type:
                botnet_counts[d.botnet_type] = botnet_counts.get(d.botnet_type, 0) + 1

        pct_compromised = round(100.0 * (n - state_counts["clean"]) / n, 2)

        return {
            "time":                   t,
            "phase":                  phase,
            "c2_command":             command,
            # Network metrics
            "traffic_mb":             round(total_traffic, 2),
            "connections":            int(total_connections),
            "c2_beacons":             self.c2.beacon_count,
            "scan_attempts":          int(total_scans),
            "failed_logins":          int(total_failed),
            "outbound_traffic_ratio": outbound_ratio,
            "unique_dst_ips":         int(total_dst_ips),
            "exfiltration_mb":        round(total_exfil, 2),
            # Device census
            "n_clean":         state_counts["clean"],
            "n_compromised":   state_counts["compromised"],
            "n_beaconing":     state_counts["beaconing"],
            "n_dormant":       state_counts["dormant"],
            "n_active":        state_counts["active"],
            "n_exfiltrating":  state_counts["exfiltrating"],
            "pct_compromised": pct_compromised,
            # Role census
            "r_scanner":    role_counts["scanner"],
            "r_ddos":       role_counts["ddos"],
            "r_exfil":      role_counts["exfil"],
            "r_exploit":    role_counts["exploit"],
            "r_iot_camera": role_counts["iot_camera"],
            # Botnet type census
            "b_mirai":    botnet_counts["mirai"],
            "b_satori":   botnet_counts["satori"],
            "b_persirai": botnet_counts["persirai"],
            "b_badbox":   botnet_counts["badbox"],
            # Role census (extended)
            "r_proxy":    role_counts.get("proxy", 0),
            "r_fraud":    role_counts.get("fraud", 0),
            # Target state
            "target_ddos_load":     round(self.target.ddos_load, 2),
            "target_exfil_recv":    round(self.target.exfil_received, 2),
            "target_degraded":      int(self.target.degraded),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION — Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
def assign_threat_level(score: float) -> str:
    if score < CONFIG["threat_medium"]:  return "low"
    elif score < CONFIG["threat_high"]:  return "medium"
    return "high"


def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train Isolation Forest on the clean normal-phase baseline,
    then score all timesteps to detect anomalous behavior.
    """
    X = df[FEATURES].values
    normal_mask = df["phase"] == "normal"
    X_train = X[normal_mask]

    scaler = MinMaxScaler()

    # Fit ONLY on normal (training) data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform full dataset using same scaler
    X_scaled = scaler.transform(X)

    model = IsolationForest(
        n_estimators=CONFIG["if_n_estimators"],
        contamination=CONFIG["if_contamination"],
        random_state=42,
    )
    model.fit(X_train_scaled)

    df = df.copy()
    df["if_flag"]   = model.predict(X_scaled)
    raw_scores      = model.score_samples(X_scaled)
    df["if_anomaly"] = df["if_flag"] == -1

    s_min, s_max = raw_scores.min(), raw_scores.max()
    df["anomaly_score"] = ((raw_scores - s_max) / (s_min - s_max)).round(3)
    df["threat_level"]  = df["anomaly_score"].apply(assign_threat_level)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK GRAPH
# ─────────────────────────────────────────────────────────────────────────────
def build_network_graph(
    df: pd.DataFrame, sim: Simulation, t_snapshot: int
) -> Tuple[nx.DiGraph, Dict]:
    """
    Build a directed communication graph for a given timestep snapshot.
    Uses actual per-device state history (Device.state_history) for accuracy.
    """
    n = CONFIG["num_devices"]

    # Use real per-device state at t_snapshot
    states_at_t: List[str] = []
    for d in sim.devices:
        if t_snapshot < len(d.state_history):
            states_at_t.append(d.state_history[t_snapshot])
        else:
            states_at_t.append(d.state)

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, state=states_at_t[i], role=sim.devices[i].role, node_type="device")

    # Special nodes
    c2_node     = n
    victim_node = n + 1
    G.add_node(c2_node,     state="c2",     node_type="c2")
    G.add_node(victim_node, state="victim", node_type="victim")

    node_colors = []
    for i in range(n):
        state = G.nodes[i]["state"]
        bt    = sim.devices[i].botnet_type
        # BadBox devices get their own teal colour regardless of state
        if bt == "badbox" and state != "clean":
            node_colors.append(_STATE_COLOR["badbox"])
        else:
            node_colors.append(_STATE_COLOR.get(state, "#bdc3c7"))
    node_colors.append("#2c3e50")   # C2: dark navy
    node_colors.append("#c0392b")   # Victim: red

    row = df[df["time"] == t_snapshot].iloc[0]

    for i, s in enumerate(states_at_t):
        # Bot → C2 beaconing / exfil
        if s in ("beaconing", "active", "exfiltrating"):
            G.add_edge(i, c2_node, edge_type="c2_comm")
        # Active DDoS bots → Victim
        if s == "active" and row["c2_command"] in ("low_rate_ddos", "burst_ddos"):
            G.add_edge(i, victim_node, edge_type="ddos")
        # Scanning edges: scanner-role bots → clean devices
        if s in ("active", "beaconing") and sim.devices[i].role == "scanner":
            rng = random.Random(t_snapshot)  # deterministic per timestep
            targets = [j for j, st in enumerate(states_at_t) if st == "clean"]
            for tgt in random.sample(targets, min(2, len(targets))):
                G.add_edge(i, tgt, edge_type="scan")

    return G, {
        "node_colors": node_colors,
        "c2_node":     c2_node,
        "victim_node": victim_node,
        "states":      states_at_t,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _shade_phases(ax: plt.Axes, df: pd.DataFrame) -> None:
    df_r = df.reset_index(drop=True)
    prev = df_r["phase"].iloc[0]
    start = 0
    for i, row in df_r.iterrows():
        if row["phase"] != prev:
            ax.axvspan(start, i, alpha=0.17, color=PHASE_COLORS[prev], linewidth=0)
            start = i
            prev = row["phase"]
    ax.axvspan(start, len(df_r), alpha=0.17, color=PHASE_COLORS[prev], linewidth=0)


def _phase_legend() -> List[mpatches.Patch]:
    return [
        mpatches.Patch(color=PHASE_COLORS[p], alpha=0.6, label=PHASE_LABELS[p])
        for p in PHASE_COLORS
    ]


def _annotate_phases(ax: plt.Axes, df: pd.DataFrame, y_frac: float = 0.96) -> None:
    """Add small phase label text above the top of each shaded region."""
    phase_order = ["normal", "compromise", "beaconing", "scanning", "attack", "recovery"]
    short = {
        "normal": "①Normal", "compromise": "②Compromise", "beaconing": "③Beacon",
        "scanning": "④Scan", "attack": "⑤Attack", "recovery": "⑥Recovery",
    }
    ylim = ax.get_ylim()
    y = ylim[0] + (ylim[1] - ylim[0]) * y_frac
    for phase in phase_order:
        rows = df[df["phase"] == phase]["time"]
        if rows.empty:
            continue
        mid = (rows.min() + rows.max()) / 2
        ax.text(mid, y, short[phase], ha="center", va="top",
                fontsize=6.5, color="#444", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))


def _save(filename: str, fig: plt.Figure) -> None:
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {filename}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1 — MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame) -> None:
    """
    5-panel main dashboard:
      1. Network traffic  — shows DDoS spikes + exfil surges
      2. Connections      — scanning causes spikes; exfil reduces count
      3. Exfiltration     — data theft volume
      4. C2 beacons       — periodic check-ins
      5. Anomaly score    — Isolation Forest output
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
    fig.suptitle(
        "IIoT BadBox-Style Botnet — Multi-Stage Lifecycle Dashboard",
        fontsize=15, fontweight="bold", y=0.995,
    )

    anomalies = df[df["if_anomaly"]]

    def _scatter_anom(ax: plt.Axes, col: str) -> None:
        if not anomalies.empty:
            ax.scatter(anomalies["time"], anomalies[col],
                       color="#c0392b", s=20, zorder=5, alpha=0.65,
                       label="IF anomaly detected")

    # ── Panel 1: Traffic ──────────────────────────────────────────────────
    ax = axes[0]
    _shade_phases(ax, df)
    ax.plot(df["time"], df["traffic_mb"], color="#2980b9", lw=1.5, label="Total traffic (MB)")
    ax.fill_between(df["time"], df["traffic_mb"], alpha=0.12, color="#2980b9")
    _scatter_anom(ax, "traffic_mb")
    ax.set_ylabel("Traffic (MB)", fontsize=9)
    ax.set_title("① Network Traffic — DDoS spikes emerge in attack phase; exfil adds sustained load",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.20)
    _annotate_phases(ax, df)

    # ── Panel 2: Connections ──────────────────────────────────────────────
    ax = axes[1]
    _shade_phases(ax, df)
    ax.plot(df["time"], df["connections"], color="#27ae60", lw=1.4, label="Connections")
    _scatter_anom(ax, "connections")
    ax.set_ylabel("Connections", fontsize=9)
    ax.set_title("② Connections — scanning causes spikes; exfiltration reduces (fewer but persistent)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.20)

    # ── Panel 3: Exfiltration ─────────────────────────────────────────────
    ax = axes[2]
    _shade_phases(ax, df)
    ax.fill_between(df["time"], df["exfiltration_mb"], alpha=0.45, color="#8e44ad")
    ax.plot(df["time"], df["exfiltration_mb"], color="#8e44ad", lw=1.3, label="Exfiltration (MB)")
    _scatter_anom(ax, "exfiltration_mb")
    ax.set_ylabel("Exfil (MB)", fontsize=9)
    ax.set_title("③ Data Exfiltration Volume — only exfil-role bots activate; BadBox-style data theft",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.20)

    # ── Panel 4: C2 beacons ───────────────────────────────────────────────
    ax = axes[3]
    _shade_phases(ax, df)
    ax.bar(df["time"], df["c2_beacons"], color="#e67e22", alpha=0.75, label="C2 beacons/step",
           width=0.9)
    ax.set_ylabel("Beacons", fontsize=9)
    ax.set_title("④ C2 Beacon Frequency — staggered check-ins; all bots synchronize during attack",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.20)

    # ── Panel 5: Anomaly score ────────────────────────────────────────────
    ax = axes[4]
    _shade_phases(ax, df)
    ax.fill_between(df["time"], df["anomaly_score"], alpha=0.35, color="#e74c3c")
    ax.plot(df["time"], df["anomaly_score"], color="#e74c3c", lw=1.3, label="Anomaly score")
    ax.axhline(CONFIG["threat_medium"], color="orange", ls="--", lw=1.1,
               label=f"Medium threshold ({CONFIG['threat_medium']})")
    ax.axhline(CONFIG["threat_high"], color="red", ls="--", lw=1.1,
               label=f"High threshold ({CONFIG['threat_high']})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Anomaly Score", fontsize=9)
    ax.set_xlabel("Simulation Timestep", fontsize=10)
    ax.set_title("⑤ Isolation Forest Anomaly Score — 0 = normal, 1 = most anomalous",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.20)

    fig.legend(handles=_phase_legend(), loc="lower center", ncol=6,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.002), frameon=True)
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    _save("iiot_botnet_dashboard.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2 — INFECTION CURVE (% devices per state)
# ─────────────────────────────────────────────────────────────────────────────
def plot_infection_curve(df: pd.DataFrame) -> None:
    """
    Stacked area chart of % devices in each state over time.
    Shows clean shrinking, botnet states growing, then recovery.
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        "Botnet Infection Curve — Device State Evolution & Role Distribution",
        fontsize=13, fontweight="bold",
    )

    _shade_phases(ax, df)
    _shade_phases(ax2, df)

    state_style = {
        "n_compromised":  ("#f39c12", "Compromised"),
        "n_beaconing":    ("#f1c40f", "Beaconing"),
        "n_dormant":      ("#95a5a6", "Dormant (stealth)"),
        "n_active":       ("#e74c3c", "Active (DDoS)"),
        "n_exfiltrating": ("#8e44ad", "Exfiltrating"),
    }

    n = CONFIG["num_devices"]
    bottom = np.zeros(len(df))
    for col, (color, label) in state_style.items():
        vals = df[col].values / n * 100
        ax.fill_between(df["time"], bottom, bottom + vals,
                        alpha=0.78, color=color, label=label)
        bottom += vals

    # Clean devices = remainder
    ax.fill_between(df["time"], bottom, 100,
                    alpha=0.35, color="#2ecc71", label="Clean")

    ax.set_ylabel("% of Devices", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", fontsize=9, bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, alpha=0.20)
    _annotate_phases(ax, df)

    # Role breakdown (bottom panel)
    ax2.stackplot(
        df["time"],
        df["r_scanner"], df["r_ddos"], df["r_exfil"],
        labels=["Scanner role", "DDoS role", "Exfil role"],
        colors=["#3498db", "#e74c3c", "#8e44ad"],
        alpha=0.75,
    )
    ax2.set_ylabel("# Bots by Role", fontsize=9)
    ax2.set_xlabel("Simulation Timestep", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.20)

    fig.legend(handles=_phase_legend(), loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.005), frameon=True)
    plt.tight_layout(rect=[0, 0.03, 0.88, 1])
    _save("iiot_botnet_infection_curve.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3 — C2 COMMUNICATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_c2_comms(df: pd.DataFrame) -> None:
    """
    C2 beacon frequency + scan attempts + C2 command timeline.
    Shows the persistent command windows (not random per step).
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("C2 Communication Activity", fontsize=13, fontweight="bold")

    for ax in axes:
        _shade_phases(ax, df)

    # Panel A: Beacons
    ax = axes[0]
    ax.bar(df["time"], df["c2_beacons"], color="#e67e22", alpha=0.80,
           label="C2 beacons/step", width=0.9)
    ax.set_ylabel("Beacons / step", fontsize=9)
    ax.set_title("C2 Beacon Frequency — bots checking in every beacon_interval steps",
                 fontsize=9, loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.20)

    # Panel B: Scan + Failed logins
    ax = axes[1]
    ax.plot(df["time"], df["scan_attempts"], color="#2980b9", lw=1.4, label="Scan attempts")
    ax.plot(df["time"], df["failed_logins"], color="#c0392b", lw=1.1,
            ls="--", label="Failed logins")
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Scanning Activity — scanner-role bots probe clean devices; failures = brute-force noise",
                 fontsize=9, loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.20)

    # Panel C: Persistent C2 command timeline
    ax = axes[2]
    command_map = {
        "idle": 0, "beacon": 1,
        "low_rate_ddos": 2, "burst_ddos": 3, "data_exfiltration": 4,
    }
    cmd_colors = {
        "idle": "#bdc3c7", "beacon": "#f1c40f",
        "low_rate_ddos": "#e67e22", "burst_ddos": "#e74c3c",
        "data_exfiltration": "#8e44ad",
    }
    prev_cmd = df["c2_command"].iloc[0]
    seg_start = 0
    for i, row in df.iterrows():
        if row["c2_command"] != prev_cmd or i == len(df) - 1:
            ax.barh(
                0, i - seg_start, left=seg_start, height=0.6,
                color=cmd_colors[prev_cmd], alpha=0.85,
                label=prev_cmd if i <= 30 else "_",
            )
            ax.text(seg_start + (i - seg_start) / 2, 0, prev_cmd.replace("_", "\n"),
                    ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold")
            seg_start = i
            prev_cmd = row["c2_command"]
    ax.set_yticks([])
    ax.set_xlabel("Simulation Timestep", fontsize=10)
    ax.set_title("Persistent C2 Command Windows — commands last multiple steps (realistic)",
                 fontsize=9, loc="left")
    ax.set_xlim(0, CONFIG["duration"])

    # Legend for command colors
    handles = [mpatches.Patch(color=v, label=k) for k, v in cmd_colors.items()]
    ax.legend(handles=handles, fontsize=7, loc="upper left", ncol=5)
    ax.grid(True, alpha=0.15)

    fig.legend(handles=_phase_legend(), loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.005), frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save("iiot_botnet_c2_comms.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 4 — NETWORK GRAPH (real per-device state)
# ─────────────────────────────────────────────────────────────────────────────
def plot_network_graph(
    df: pd.DataFrame, sim: Simulation, t_snapshot: int = 160
) -> None:
    """
    Directed network graph at peak attack timestep.
    Uses REAL per-device state history (not approximated distributions).
    Nodes = devices + C2 + Victim.
    Edges = C2 beacons, DDoS flows, scan attempts.
    """
    G, meta = build_network_graph(df, sim, t_snapshot)
    n        = CONFIG["num_devices"]
    c2_node  = meta["c2_node"]
    vic_node = meta["victim_node"]

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(
        f"Network Communication Graph — Timestep {t_snapshot}  (Peak Attack Phase)\n"
        f"Real per-device state history · Nodes colored by state · "
        f"Edges show communication flow",
        fontsize=11, fontweight="bold",
    )

    # Layout: circle for devices, C2 top-centre, Victim bottom-centre
    pos = nx.circular_layout(list(range(n)))
    pos[c2_node]  = np.array([0.0,  0.85])
    pos[vic_node] = np.array([0.0, -0.85])

    # Node sizes: C2 and Victim are larger
    node_sizes = [320] * n + [900, 900]

    # Separate edge types
    c2_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "c2_comm"]
    ddos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "ddos"]
    scan_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "scan"]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=meta["node_colors"],
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: str(i) for i in range(n)},
                            font_size=6.5, font_color="white", font_weight="bold")
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={c2_node: "C2\nServer", vic_node: "Victim\nServer"},
                            font_size=8.5, font_color="white", font_weight="bold")

    # Edges
    nx.draw_networkx_edges(G, pos, edgelist=c2_edges, ax=ax,
                           edge_color="#e67e22", arrows=True, arrowsize=11,
                           width=1.3, alpha=0.55, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edges(G, pos, edgelist=ddos_edges, ax=ax,
                           edge_color="#e74c3c", arrows=True, arrowsize=14,
                           width=2.0, alpha=0.65, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(G, pos, edgelist=scan_edges, ax=ax,
                           edge_color="#3498db", arrows=True, arrowsize=9,
                           width=0.9, alpha=0.45, connectionstyle="arc3,rad=0.15",
                           style="dashed")

    # Legend — states
    state_handles = [
        mpatches.Patch(color=c, label=s.replace("_", " ").title())
        for s, c in _STATE_COLOR.items()
    ]
    state_handles += [
        mpatches.Patch(color="#2c3e50", label="C2 Server"),
        mpatches.Patch(color="#c0392b", label="Victim Server"),
    ]
    edge_handles = [
        mpatches.Patch(color="#e67e22", alpha=0.6, label="→ C2 beacon"),
        mpatches.Patch(color="#e74c3c", alpha=0.7, label="→ DDoS flood"),
        mpatches.Patch(color="#3498db", alpha=0.5, label="→ Scan attempt"),
    ]
    leg1 = ax.legend(handles=state_handles, loc="lower left",  fontsize=8, title="Device State")
    ax.add_artist(leg1)
    ax.legend(handles=edge_handles, loc="lower right", fontsize=8, title="Edge Type")

    ax.axis("off")
    plt.tight_layout()
    _save("iiot_botnet_network_graph.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 5 — OUTBOUND + EXFILTRATION + TARGET IMPACT
# ─────────────────────────────────────────────────────────────────────────────
def plot_outbound_exfil(df: pd.DataFrame) -> None:
    """
    3-panel outbound behavior + victim impact:
      1. Outbound traffic ratio
      2. Unique destination IPs
      3. Attack target: DDoS load + degraded indicator
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Outbound Behaviour · Exfiltration Fingerprints · Victim Impact",
                 fontsize=12, fontweight="bold")

    for ax in axes:
        _shade_phases(ax, df)

    # Panel 1: Outbound ratio
    ax = axes[0]
    ax.plot(df["time"], df["outbound_traffic_ratio"],
            color="#16a085", lw=1.5, label="Outbound traffic ratio")
    ax.fill_between(df["time"], df["outbound_traffic_ratio"], alpha=0.18, color="#16a085")
    ax.axhline(0.7, color="#c0392b", ls=":", lw=1.0, label="Exfil threshold (0.70)")
    ax.set_ylabel("Outbound ratio (0–1)", fontsize=9)
    ax.set_title("Outbound Traffic Ratio — exfiltration pushes this toward 1.0 (all traffic leaving)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.20)
    _annotate_phases(ax, df)

    # Panel 2: Unique destination IPs
    ax = axes[1]
    ax.plot(df["time"], df["unique_dst_ips"],
            color="#8e44ad", lw=1.4, label="Unique destination IPs")
    ax.fill_between(df["time"], df["unique_dst_ips"], alpha=0.18, color="#8e44ad")
    ax.set_ylabel("Unique IPs", fontsize=9)
    ax.set_title("Unique Destination IPs — many during DDoS (fan-out), very few during exfil (one C2)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.20)

    # Panel 3: Victim target impact
    ax = axes[2]
    ax2b = ax.twinx()
    l1, = ax.plot(df["time"], df["target_ddos_load"],
                  color="#e74c3c", lw=1.5, label="DDoS load on victim (MB)")
    l2, = ax2b.plot(df["time"], df["target_exfil_recv"],
                    color="#8e44ad", lw=1.4, ls="--", label="Exfil received by C2 (MB)")
    # Shade when target is degraded
    degraded = df[df["target_degraded"] == 1]
    if not degraded.empty:
        ax.fill_between(df["time"], 0, df["target_ddos_load"],
                        where=(df["target_degraded"] == 1),
                        alpha=0.30, color="#e74c3c", label="Victim degraded ⚠️")
    ax.set_ylabel("DDoS Load (MB)", fontsize=9, color="#e74c3c")
    ax2b.set_ylabel("Exfil (MB)", fontsize=9, color="#8e44ad")
    ax.set_xlabel("Simulation Timestep", fontsize=10)
    ax.set_title("Attack Target Impact — DDoS load exceeding threshold causes service degradation",
                 fontsize=9, loc="left")
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.20)

    fig.legend(handles=_phase_legend(), loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.005), frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save("iiot_botnet_outbound_exfil.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# STEP-BY-STEP PHASE EXPLANATIONS
# ─────────────────────────────────────────────────────────────────────────────
PHASE_EXPLANATIONS = {
"normal": """
━━━ PHASE 1 · Normal Baseline (t=0–40) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT HAPPENS:
  All 30 IIoT devices operate normally. No botnet activity.
  C2 command: "idle" — the attacker has not yet taken action.

DEVICE BEHAVIOR:
  Every device generates baseline traffic (~5 MB/step), a moderate number of
  connections, and low outbound ratios. Traffic is noisy but stable.

METRICS CHANGES:
  • Traffic:  flat ~150 MB/step total (30 devices × 5 MB)
  • Beacons:  zero (no bots yet)
  • Anomaly:  score stays near 0 — Isolation Forest trained on this window

REAL-WORLD ANALOGY:
  A factory floor where all sensors, PLCs, and edge devices are reporting
  normally. Security teams see clean dashboards.
""",

"compromise": """
━━━ PHASE 2 · Initial Compromise (t=40–70) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT HAPPENS:
  At t=40, the attacker exploits a vulnerability in ~3 devices (10% of fleet).
  These become "patient zero" — the seed infection. Each compromised device
  is assigned a ROLE: scanner / ddos / exfil.
  C2 command rotates between "idle" and "beacon" — establishing foothold quietly.

DEVICE BEHAVIOR:
  Compromised devices increase traffic only slightly (~5%). They haven't started
  beaconing yet — the attacker is still establishing the C2 channel.

METRICS CHANGES:
  • n_compromised: spikes from 0 → ~3
  • Traffic:  barely detectable (+5%)
  • Anomaly:  may start rising slightly

REAL-WORLD ANALOGY:
  BadBox malware embeds itself in Android firmware before the device boots.
  The user has no idea. The malware sits silently.
""",

"beaconing": """
━━━ PHASE 3 · Beaconing / Stealth (t=70–110) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT HAPPENS:
  Compromised devices begin regular C2 check-ins (beacons) every 3 steps.
  Some devices intentionally go DORMANT to evade detection — a stealth tactic.
  C2 command: persistent "beacon" blocks lasting 6 steps each.

DEVICE BEHAVIOR:
  Beaconing devices generate small periodic traffic spikes (sine-wave pattern).
  Dormant devices reduce their footprint to nearly baseline — they look clean.
  Dormant devices have a countdown timer; they reactivate automatically.

METRICS CHANGES:
  • c2_beacons: rises to 2–6 per step (staggered by device ID)
  • n_dormant:  some devices enter dormancy
  • Traffic:  slight periodic bumps
  • Anomaly:  starts rising as beacon pattern emerges

REAL-WORLD ANALOGY:
  Mirai bots check in with C2 every few minutes using tiny HTTP requests.
  Dormancy mimics real APT behavior where bots "sleep" to evade endpoint EDR.
""",

"scanning": """
━━━ PHASE 4 · Scanning & Lateral Spread (t=110–140) ━━━━━━━━━━━━━━━━━━━━━━━
WHAT HAPPENS:
  Per-device scanning begins. SCANNER-role bots try to compromise clean devices.
  Each scanner independently probes 1–3 clean targets per step.
  Success probability = 22% per probe. Failed probes log as "failed_logins".
  C2 command: "beacon" (maintaining spread while staying quiet).

DEVICE BEHAVIOR:
  Scanner-role bots: 10–25 scan attempts/step
  DDoS/Exfil-role bots: 2–10 scan attempts/step (opportunistic)
  Newly compromised devices immediately receive a role assignment.

METRICS CHANGES:
  • scan_attempts: spikes sharply (100–400/step)
  • failed_logins: high (65% of scans fail = brute-force noise)
  • n_compromised: grows as spread propagates
  • Anomaly:  sharply elevated

REAL-WORLD ANALOGY:
  Mirai's telnet scanner tries default credentials on every reachable IP.
  Failed login logs flood SIEM systems — a key detection indicator.
""",

"attack": """
━━━ PHASE 5 · Attack + Exfiltration (t=140–170) ━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT HAPPENS:
  The botnet activates for its primary mission. C2 rotates through three
  persistent commands over 6-step windows:
    • low_rate_ddos    — sustained low-volume flood (harder to filter)
    • burst_ddos       — Mirai-style volumetric spike (×2.8 traffic multiplier)
    • data_exfiltration — only exfil-role bots activate; data sent to C2

  DDoS traffic floods the Victim Server. When cumulative load >500 MB,
  the victim service is marked "degraded" (simulating real service disruption).

DEVICE BEHAVIOR:
  DDoS-role active bots: fan out to 20–60 unique destination IPs; 
                         high outbound ratio (70–92%)
  Exfil-role bots:       connect to just 1–2 C2 IPs; send 50 MB/step
  All bots beacon EVERY step (synchronized attack coordination)
  Dormant bots reactivate when their timer expires

METRICS CHANGES:
  • traffic_mb:       spikes ×3–6 during burst DDoS
  • exfiltration_mb:  rises sharply during exfil windows
  • unique_dst_ips:   high during DDoS; low during exfil
  • target_degraded:  1 when victim is overwhelmed
  • Anomaly:          near maximum (0.9–1.0)

REAL-WORLD ANALOGY:
  2016 Mirai botnet generated 620 Gbps DDoS against Krebs on Security.
  BadBox 2.0 simultaneously ran residential proxy fraud + data collection.
""",

"recovery": """
━━━ PHASE 6 · Recovery / Remediation (t=170–200) ━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT HAPPENS:
  Security teams begin isolating infected devices. Each step, bots have a 16%
  chance of being cleaned (firmware reflash / network isolation / reboot).
  C2 reverts to "idle" — the attacker pulls back.

DEVICE BEHAVIOR:
  Bots in all states (beaconing, active, exfiltrating, dormant) probabilistically
  return to "clean". Exfiltrating bots also have a separate 22% clean probability.
  Fleet gradually returns to clean state by t=200.

METRICS CHANGES:
  • n_clean: rises back toward 30
  • Traffic, beacons, scans: all fall
  • Anomaly:  decreases as behavior normalizes

REAL-WORLD ANALOGY:
  ISPs and CERTs notify device owners; devices are patched or factory-reset.
  Some bots survive by hiding (dormancy) — real botnets persist for months.
""",
}


def print_phase_explanations() -> None:
    print("\n" + "═" * 70)
    print("  STEP-BY-STEP PHASE EXPLANATIONS")
    print("═" * 70)
    for phase, text in PHASE_EXPLANATIONS.items():
        print(text)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
VIZ_GUIDE = """
═══════════════════════════════════════════════════════════════════════════════
  VISUALIZATION GUIDE
═══════════════════════════════════════════════════════════════════════════════

① iiot_botnet_dashboard.png  — MAIN DASHBOARD
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Panel 1 · Network Traffic (MB)                                          │
  │   Flat in normal. Slight rise in beaconing. Massive spikes during DDoS. │
  │   IF anomalies (red dots) mark detected outliers.                        │
  │ Panel 2 · Connections                                                    │
  │   Rises during scanning. Falls during exfiltration (fewer, persistent). │
  │ Panel 3 · Exfiltration (MB)                                              │
  │   Zero until attack phase. Surges when data_exfiltration command fires. │
  │ Panel 4 · C2 Beacons/step                                               │
  │   Staggered in beaconing. Synchronizes (all bots) during attack.        │
  │ Panel 5 · Anomaly Score                                                  │
  │   Near 0 in normal. Rises with beaconing. Peaks at 0.9+ during attack. │
  └─────────────────────────────────────────────────────────────────────────┘

② iiot_botnet_infection_curve.png  — INFECTION CURVE
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Top: Stacked area — % of devices in each state over time.               │
  │   Green (clean) shrinks. Yellow→Red→Purple grows as infection spreads.  │
  │   Grey (dormant) shows stealth periods mid-infection.                   │
  │ Bottom: Role breakdown — scanner / ddos / exfil bots over time.         │
  └─────────────────────────────────────────────────────────────────────────┘

③ iiot_botnet_c2_comms.png  — C2 COMMUNICATION
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Panel 1 · Beacon frequency — bar chart of check-ins per step.           │
  │ Panel 2 · Scan attempts vs failed logins.                               │
  │ Panel 3 · Persistent C2 command timeline (color-coded blocks).          │
  │   CRITICAL: commands hold for 6 steps — realistic operator behavior.    │
  └─────────────────────────────────────────────────────────────────────────┘

④ iiot_botnet_network_graph.png  — NETWORK GRAPH (REAL states)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Nodes = 30 devices (colored by ACTUAL per-device state at t=160)        │
  │       + C2 Server (dark navy, centre-top)                               │
  │       + Victim Server (red, centre-bottom)                              │
  │ Edges:                                                                  │
  │   Orange  → C2 beacons (bots checking in)                              │
  │   Red     → DDoS flood toward Victim                                   │
  │   Blue    → Scan attempts (scanner-role bots probing clean devices)     │
  └─────────────────────────────────────────────────────────────────────────┘

⑤ iiot_botnet_outbound_exfil.png  — OUTBOUND + VICTIM IMPACT
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Panel 1 · Outbound traffic ratio — exfil pushes toward 1.0             │
  │ Panel 2 · Unique destination IPs — many (DDoS), few (exfil to C2)      │
  │ Panel 3 · Victim impact — DDoS load + degraded marker + exfil received │
  └─────────────────────────────────────────────────────────────────────────┘
"""


# ─────────────────────────────────────────────────────────────────────────────
# REAL-BOTNET COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
BOTNET_COMPARISON = """
═══════════════════════════════════════════════════════════════════════════════
  HOW THIS RESEMBLES REAL BOTNETS
═══════════════════════════════════════════════════════════════════════════════

MIRAI BOTNET (2016)
  ✓ Default-credential telnet scanning → simulated as scan_attempts + failed_logins
  ✓ Rapid lateral spread to IoT devices → lateral_spread() per device each step
  ✓ Massive volumetric DDoS (620 Gbps) → burst_ddos command (×6 traffic multiplier)
  ✓ C2-synchronized attack onset → all bots beacon every step during attack phase

BADBOX / BADBOX 2.0 (2022–2024)
  ✓ Firmware-level persistence (pre-compromise) → patient-zero seeding at t=40
  ✓ Residential proxy abuse → exfil-role devices mimic data tunneling
  ✓ Mixed payload (ad fraud + data theft) → separate ddos + exfil roles
  ✓ Silent dormancy periods → dormant state with countdown timer

GENERAL BOTNET BEHAVIORS
  ✓ Persistent C2 commands (not random per step) → command_persistence=6 steps
  ✓ Staggered beacons to avoid synchronized bursts → stagger by device_id % interval
  ✓ Low-rate → burst DDoS escalation → low_rate_ddos before burst_ddos
  ✓ Data exfiltration uses few destination IPs → 1–2 dst_ips in exfil state
  ✓ Anomaly detection (Isolation Forest) → trained on clean baseline, scores all steps

KEY DIFFERENCES FROM REALITY (SIMULATION CONSTRAINTS)
  ✗ No real exploit code or CVE usage
  ✗ Propagation is probabilistic, not protocol-accurate
  ✗ Network topology is flat (no subnets, firewalls, or VLANs)
  ✗ Recovery is instantaneous per-device, not incremental patch deployment
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  IIoT BadBox-Style Botnet Simulation v2")
    print("  C2-persistent · Per-device state history · Role-based · Victim target")
    print("=" * 70)

    print("\n[1/5] Running simulation...")
    sim = Simulation(seed=CONFIG["seed"])
    df  = sim.run()

    print("[2/5] Running Isolation Forest anomaly detection...")
    df = run_isolation_forest(df)

    df.to_csv("iiot_botnet_simulation_data.csv", index=False)
    print("      Data saved → iiot_botnet_simulation_data.csv")

    # ── Summary statistics ────────────────────────────────────────────────
    print("\n── Threat Level Distribution ────────────────────────────────────")
    print(df["threat_level"].value_counts().to_string())

    print("\n── Anomalies Detected by Phase ──────────────────────────────────")
    print(df.groupby("phase")["if_anomaly"].sum().to_string())

    print("\n── Peak Metrics by Phase ────────────────────────────────────────")
    cols = ["traffic_mb", "exfiltration_mb", "c2_beacons", "scan_attempts", "failed_logins"]
    print(df.groupby("phase")[cols].max().to_string())

    print("\n── Device State at Peak Attack (t=160) ──────────────────────────")
    row160 = df[df["time"] == 160].iloc[0]
    for s in DEVICE_STATES:
        print(f"  {s:>14}: {int(row160[f'n_{s}']):3d}")
    print(f"  {'target_degraded':>14}: {int(row160['target_degraded'])}")

    print("\n── Bot Role Distribution at Peak ────────────────────────────────")
    for r in ["r_scanner", "r_ddos", "r_exfil"]:
        print(f"  {r:>12}: {int(row160[r])}")

    # ── Plots ────────────────────────────────────────────────────────────
    print("\n[3/5] Rendering 5 visualizations...")
    plot_dashboard(df)
    plot_infection_curve(df)
    plot_c2_comms(df)
    plot_network_graph(df, sim, t_snapshot=160)
    plot_outbound_exfil(df)

    # ── Explanations ──────────────────────────────────────────────────────
    print("\n[4/5] Phase explanations:")
    print_phase_explanations()

    print(VIZ_GUIDE)
    print(BOTNET_COMPARISON)

    # ── Sample records ────────────────────────────────────────────────────
    print("[5/5] Sample records (every 20 steps):")
    sample_cols = [
        "time", "phase", "c2_command",
        "traffic_mb", "exfiltration_mb", "c2_beacons",
        "n_clean", "n_active", "n_exfiltrating",
        "anomaly_score", "threat_level", "target_degraded",
    ]
    print(df[sample_cols].iloc[::20].to_string(index=False))

    print("\n✓ All outputs written to working directory.")
    print("  Files: iiot_botnet_dashboard.png | iiot_botnet_infection_curve.png")
    print("         iiot_botnet_c2_comms.png  | iiot_botnet_network_graph.png")
    print("         iiot_botnet_outbound_exfil.png | iiot_botnet_simulation_data.csv")
