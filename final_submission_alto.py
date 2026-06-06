"""
AutoTiering Simulation – v5  (All-Algorithm Edition)
======================================================

Algorithms compared
────────────────────
  baseline   Fault-driven: direct NUMA-fault-triggered migration to preferred DRAM.
  cpm        Fault-driven: tiered-fallback placement (local DRAM → remote DRAM → DCPMM).
  opm        Scan-driven:  LAP-threshold promotion/demotion, no migration cap.
  opmx       Scan-driven:  throttled OPM with stability guards.
  tpp_alto   Scan-driven:  Linux tpp.patch + tpp-alto.patch kernel stack simulation.
  alto_opm   Scan-driven:  Kleio v2 ML-gated promotion/demotion (v4 fixes applied).

Efficiency definitions
──────────────────────
  Fault-driven (baseline, cpm):
      efficiency = fault_migrated / fault_eligible
  Scan-driven  (opm, opmx, tpp_alto, alto_opm):
      efficiency = promotions / (promotions + demotions)

Changes in v5 vs v4
────────────────────
  • tpp_alto added as a 6th algorithm (from final_submission_tpp_alto.py).
  • Page class extended with is_active and pg_demoted fields for TPP.
  • MigrationStats.alto_* rejection counters preserved (only used by alto_opm).
  • All plotting extended to 6 algorithms; new TPP-specific graphs added:
      – Graph 8 : TPP-Alto vs ALTO-OPM head-to-head efficiency comparison.
      – Graph 9 : TPP-Alto active vs total page promotion analysis.
  • Output filenames updated to _v5.
"""

from collections import deque
import random
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  Shared constants
# ══════════════════════════════════════════════════════════════════════════════
HOT_THRESHOLD           = 6
COLD_THRESHOLD          = 2
MIGRATION_THRESHOLD     = 2
MAX_MIGRATIONS_PER_SCAN = 50

# Kleio v2 weights  [lap_norm, recency, freq_norm, locality,
#                    hot_streak_norm, consistency]
KLEIO_WEIGHTS = np.array([0.30, 0.20, 0.15, 0.10, 0.15, 0.10])

# ALTO gate parameters
ALTO_BASE_PROMOTE_THRESH = 0.50
ALTO_PROMOTE_SLOPE       = 0.28
ALTO_ML_DEMOTE_THRESH    = 0.22
ALTO_MIN_SCORE_GAP       = 0.18
ALTO_HIGH_PRESSURE_GAP   = 0.32
ALTO_PRESSURE_LIMIT      = 0.80
HOT_STREAK_MIN           = 2
ALTO_MAX_MIGRATIONS      = 50

# TPP-Alto knobs (mirrors kernel sysctl tunables)
TPP_ALTO_CONFIG = {
    "disable_migrate"    : False,   # sysctl_numa_balancing_disable_migrate
    "page_promote_scale" : 5,       # promotions allowed per 10 decisions
    "skip_node_one"      : True,    # numa_skip_node_one
}


# ══════════════════════════════════════════════════════════════════════════════
#  Page
# ══════════════════════════════════════════════════════════════════════════════
class Page:
    def __init__(self, page_id):
        self.size              = 4096
        self.page_id           = page_id
        self.current_node      = None
        self.last_accessed_cpu = None
        self.access_history    = 0b00000000   # 8-bit sliding window
        self.accessed_scan     = False
        self.access_frequency  = 0
        self.lap_level         = 0
        self.prot_none         = False
        self.fault_count       = {}
        self.scan_generation   = 0

        # Kleio v2 extra fields
        self.total_faults_local = 0
        self.total_faults_all   = 0
        self.hot_streak         = 0   # consecutive scans above HOT_THRESHOLD

        # TPP-Alto extra fields
        self.is_active  = False   # mirrors PageActive(): set on access, cleared on cold scan
        self.pg_demoted = False   # mirrors PG_demoted: set when page is demoted via MR_DEMOTION

    def mark_for_numa_scan(self):
        self.prot_none = True

    def numa_fault(self, cpu_id, current_generation):
        if self.scan_generation != current_generation:
            self.fault_count     = {}
            self.scan_generation = current_generation
        self.last_accessed_cpu = cpu_id
        self.accessed_scan     = True
        self.access_frequency += 1
        self.is_active         = True   # TPP: access → active
        self.fault_count[cpu_id] = self.fault_count.get(cpu_id, 0) + 1
        self.total_faults_all   += 1
        if cpu_id == self.last_accessed_cpu:
            self.total_faults_local += 1
        return self.fault_count[cpu_id]

    def record_access(self, cpu_id):
        self.last_accessed_cpu = cpu_id
        self.accessed_scan     = True
        self.access_frequency += 1
        self.is_active         = True   # TPP: access → active

    def update_history_on_scan(self):
        """Update 8-bit sliding history, lap_level, hot_streak, and TPP is_active."""
        bit = 1 if self.accessed_scan else 0
        self.access_history = ((self.access_history << 1) | bit) & 0xFF
        if not self.accessed_scan:
            self.is_active = False   # TPP: not accessed this window → inactive
        self.accessed_scan    = False
        self.access_frequency = 0

        old_level      = self.lap_level
        self.lap_level = bin(self.access_history).count('1')

        # Update hot_streak: increment if still hot, reset if not
        if self.lap_level >= HOT_THRESHOLD:
            self.hot_streak += 1
        else:
            self.hot_streak = 0

        if self.current_node and old_level != self.lap_level:
            self.current_node.update(self, old_level)

    def aol_vector(self):
        """
        6-feature AOL vector for Kleio v2.

        Feature          Range   Description
        ───────────────  ──────  ─────────────────────────────────────
        lap_norm         [0,1]   lap_level / 8
        recency          {0,1}   accessed in last scan window?
        freq_norm        [0,1]   access_frequency / 10, clipped
        locality         [0,1]   fraction of faults from same CPU
        hot_streak_norm  [0,1]   consecutive hot scans / 8  (v2 NEW)
        consistency      [0,1]   popcount(history bits 1-4) / 4  (v2 NEW)
        """
        lap_norm        = self.lap_level / 8.0
        recency         = float((self.access_history & 0b00000001) != 0)
        freq_norm       = min(self.access_frequency / 10.0, 1.0)
        locality        = (self.total_faults_local / self.total_faults_all
                           if self.total_faults_all > 0 else 0.5)
        hot_streak_norm = min(self.hot_streak / 8.0, 1.0)
        mid4            = (self.access_history >> 1) & 0b00001111
        consistency     = bin(mid4).count('1') / 4.0
        return np.array([lap_norm, recency, freq_norm,
                         locality, hot_streak_norm, consistency])

    def __repr__(self):
        return (f"Page(id={self.page_id}, "
                f"node={self.current_node.name if self.current_node else None}, "
                f"lap={self.lap_level}, streak={self.hot_streak}, active={self.is_active})")


# ══════════════════════════════════════════════════════════════════════════════
#  KleioHotnessModel  (Kleio v2, 6-feature)
# ══════════════════════════════════════════════════════════════════════════════
class KleioHotnessModel:
    def __init__(self, weights=KLEIO_WEIGHTS):
        assert len(weights) == 6,              "Need 6 weights for v2 AOL vector"
        assert abs(weights.sum() - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.weights = weights

    def score(self, page: Page) -> float:
        return float(np.clip(np.dot(self.weights, page.aol_vector()), 0.0, 1.0))

    def score_all(self, pages) -> dict:
        return {p.page_id: self.score(p) for p in pages}


# ══════════════════════════════════════════════════════════════════════════════
#  MemoryNode
# ══════════════════════════════════════════════════════════════════════════════
class MemoryNode:
    def __init__(self, node_id, name, tier, capacity, cpu_socket):
        self.node_id    = node_id
        self.name       = name
        self.tier       = tier
        self.capacity   = capacity
        self.cpu_socket = cpu_socket
        self.pages      = deque()
        self.lap_lists  = {i: [] for i in range(9)}

    def get_latency(self, cpu_id):
        is_local = (self.cpu_socket == cpu_id)
        if self.tier.upper() == 'UPPER':
            return 100 if is_local else 150
        else:
            return 300 if is_local else 350

    def is_full(self):     return len(self.pages) >= self.capacity
    def free_slots(self):  return self.capacity - len(self.pages)
    def utilization(self): return len(self.pages) / self.capacity if self.capacity else 0

    def add_page(self, page):
        self.pages.append(page)
        page.current_node = self
        self.lap_lists[page.lap_level].append(page)

    def remove_page(self, page):
        try:    self.pages.remove(page)
        except ValueError: pass
        if page in self.lap_lists[page.lap_level]:
            self.lap_lists[page.lap_level].remove(page)
        page.current_node = None

    def get_least_accessed_page(self):
        for level in range(9):
            if self.lap_lists[level]:
                return self.lap_lists[level][0]
        return None

    def update(self, page, old_level):
        if page in self.lap_lists[old_level]:
            self.lap_lists[old_level].remove(page)
        self.lap_lists[page.lap_level].append(page)

    def __repr__(self):
        return f"MemoryNode({self.name}, {len(self.pages)}/{self.capacity}, tier={self.tier})"


# ══════════════════════════════════════════════════════════════════════════════
#  CPU
# ══════════════════════════════════════════════════════════════════════════════
class CPU:
    def __init__(self, cpu_id):
        self.cpu_id         = cpu_id
        self.total_accesses = 0
        self.total_latency  = 0

    def access(self, page, scheduler=None):
        if page.current_node is None:
            raise ValueError("Page not mapped to any node")
        if page.prot_none and scheduler:
            fault_count = page.numa_fault(self.cpu_id, scheduler.current_generation)
            result = ("fault", fault_count)
        else:
            page.record_access(self.cpu_id)
            result = ("normal", None)
        latency              = page.current_node.get_latency(self.cpu_id)
        self.total_latency  += latency
        self.total_accesses += 1
        return result

    def avg_latency(self):
        return self.total_latency / self.total_accesses if self.total_accesses else 0


# ══════════════════════════════════════════════════════════════════════════════
#  AutoNUMAScheduler
# ══════════════════════════════════════════════════════════════════════════════
class AutoNUMAScheduler:
    SCAN_PERIOD_MIN      = 50
    SCAN_PERIOD_MAX      = 200
    FAULT_THRESHOLD      = 3
    MIGRATION_RATE_LIMIT = 256

    def __init__(self):
        self.scan_period            = 100
        self.access_count           = 0
        self.migrations_this_period = 0
        self.current_generation     = 0
        self.total_scans            = 0

    def tick(self, memory_system):
        self.access_count += 1
        if self.access_count >= self.scan_period:
            self._end_of_period(memory_system)

    def _end_of_period(self, memory_system):
        if self.migrations_this_period > 0:
            self.scan_period = max(self.SCAN_PERIOD_MIN, self.scan_period // 2)
        else:
            self.scan_period = min(self.SCAN_PERIOD_MAX, self.scan_period * 2)
        self.migrations_this_period = 0
        self.access_count           = 0
        self.current_generation    += 1
        self.total_scans           += 1
        memory_system.SystemScan()

    def is_throttled(self):
        return self.migrations_this_period >= self.MIGRATION_RATE_LIMIT

    def record_migration(self):
        self.migrations_this_period += 1


# ══════════════════════════════════════════════════════════════════════════════
#  MigrationStats
# ══════════════════════════════════════════════════════════════════════════════
class MigrationStats:
    def __init__(self):
        self.promotions         = 0
        self.demotions          = 0
        self.lateral            = 0
        self.fault_eligible     = 0
        self.fault_migrated     = 0
        # ALTO rejection counters (only populated by alto_opm)
        self.alto_ml_rejected     = 0
        self.alto_gap_rejected    = 0
        self.alto_streak_rejected = 0
        self.alto_stab_rejected   = 0

    def record_move(self, src_node, dst_node):
        src_upper = src_node.tier.upper() == 'UPPER'
        dst_upper = dst_node.tier.upper() == 'UPPER'
        if   not src_upper and dst_upper:  self.promotions += 1
        elif src_upper and not dst_upper:  self.demotions  += 1
        else:                              self.lateral    += 1

    def total(self):
        return self.promotions + self.demotions + self.lateral

    def fault_driven_efficiency(self):
        if self.fault_eligible == 0: return 0.0
        return self.fault_migrated / self.fault_eligible

    def scan_driven_efficiency(self):
        total_scan = self.promotions + self.demotions
        if total_scan == 0: return 0.0
        return self.promotions / total_scan

    def alto_total_rejected(self):
        return (self.alto_ml_rejected + self.alto_gap_rejected
                + self.alto_streak_rejected + self.alto_stab_rejected)


# ══════════════════════════════════════════════════════════════════════════════
#  MemorySystem
# ══════════════════════════════════════════════════════════════════════════════
class MemorySystem:
    def __init__(self, stats: MigrationStats):
        self.stats   = stats
        self.dram_0  = MemoryNode(0, "DRAM_node0",  "upper", capacity=20, cpu_socket=0)
        self.dram_1  = MemoryNode(1, "DRAM_node1",  "upper", capacity=20, cpu_socket=1)
        self.dcpmm_2 = MemoryNode(2, "DCPMM_node2", "lower", capacity=80, cpu_socket=0)
        self.dcpmm_3 = MemoryNode(3, "DCPMM_node3", "lower", capacity=80, cpu_socket=1)
        self.cpu0    = CPU(0)
        self.cpu1    = CPU(1)
        self.page_table = {}
        self.all_nodes  = [self.dram_0, self.dram_1, self.dcpmm_2, self.dcpmm_3]
        self.all_cpu    = [self.cpu0, self.cpu1]

    def initialize_pages(self, num_pages):
        for i in range(num_pages):
            page   = Page(i)
            target = self.dcpmm_2 if i % 2 == 0 else self.dcpmm_3
            if target.is_full():
                alt    = self.dcpmm_3 if i % 2 == 0 else self.dcpmm_2
                target = alt if not alt.is_full() else None
            if target:
                target.add_page(page)
            self.page_table[i] = page

    def SystemScan(self):
        for node in self.all_nodes:
            for page in list(node.pages):
                page.update_history_on_scan()
                page.mark_for_numa_scan()

    def get_preferred_node(self, cpu_id):
        return self.dram_0 if cpu_id == 0 else self.dram_1

    def get_cpm_fallback_order(self, cpu_id):
        preferred = [self.dram_0, self.dram_1] if cpu_id == 0 else [self.dram_1, self.dram_0]
        return preferred + [self.dcpmm_2, self.dcpmm_3]

    def _dram_pressure(self):
        total_cap  = self.dram_0.capacity + self.dram_1.capacity
        total_used = len(self.dram_0.pages) + len(self.dram_1.pages)
        return total_used / total_cap if total_cap else 0.0

    def _alto_promote_thresh(self):
        pressure = self._dram_pressure()
        return min(ALTO_BASE_PROMOTE_THRESH + ALTO_PROMOTE_SLOPE * pressure, 0.95)

    def move_logic(self, page, destination):
        if page.current_node is destination:
            return
        src = page.current_node
        if src:
            src.remove_page(page)
        destination.add_page(page)
        page.prot_none = False
        if src:
            self.stats.record_move(src, destination)

    def get_victim(self, node1, node2):
        v0 = node1.get_least_accessed_page()
        v1 = node2.get_least_accessed_page()
        if v0 is None: return v1
        if v1 is None: return v0
        return v0 if v0.lap_level <= v1.lap_level else v1

    # ── Fault-driven: baseline ─────────────────────────────────────────────
    def migrate_baseline(self, page, cpu_id, scheduler):
        target = self.get_preferred_node(cpu_id)
        if page.current_node is target:    return "already_local"
        if scheduler.is_throttled():       return "throttled"
        if target.is_full():               return "failed_full"
        self.move_logic(page, target)
        scheduler.record_migration()
        self.stats.fault_migrated += 1
        return "migrated"

    # ── Fault-driven: CPM ─────────────────────────────────────────────────
    def migrate_cpm(self, page, cpu_id, scheduler):
        if scheduler.is_throttled(): return "throttled"
        for node in self.get_cpm_fallback_order(cpu_id):
            if page.current_node is node: return "already_at_best"
            if not node.is_full():
                self.move_logic(page, node)
                scheduler.record_migration()
                self.stats.fault_migrated += 1
                return f"migrated_to_{node.name}"
        return "failed_all_full"

    # ── Fault dispatcher ──────────────────────────────────────────────────
    def handle_numa_fault(self, page, cpu_id, fault_count, scheduler, mode):
        preferred = self.get_preferred_node(cpu_id)
        if page.current_node is preferred:          return "local_already"
        if fault_count < scheduler.FAULT_THRESHOLD: return "insufficient_faults"
        self.stats.fault_eligible += 1
        if   mode == "baseline": return self.migrate_baseline(page, cpu_id, scheduler)
        elif mode == "cpm":      return self.migrate_cpm(page, cpu_id, scheduler)
        else:                    return "opm_demand_noted"

    # ── Scan-driven: OPM ──────────────────────────────────────────────────
    def opm(self):
        for node in [self.dcpmm_2, self.dcpmm_3]:
            for level in range(HOT_THRESHOLD, 9):
                for page in list(node.lap_lists[level]):
                    cpu_id = page.last_accessed_cpu
                    if cpu_id is None: continue
                    target_dram = self.dram_0 if cpu_id == 0 else self.dram_1
                    if page.current_node is target_dram: continue
                    if not target_dram.is_full():
                        self.move_logic(page, target_dram)
                    else:
                        victim = self.get_victim(self.dram_0, self.dram_1)
                        if victim and page.lap_level > victim.lap_level:
                            v_cpu   = victim.last_accessed_cpu
                            v_dcpmm = self.dcpmm_2 if (v_cpu == 0 or v_cpu is None) else self.dcpmm_3
                            self.move_logic(victim, v_dcpmm)
                            self.move_logic(page,   target_dram)
        for node in [self.dram_0, self.dram_1]:
            for level in range(0, COLD_THRESHOLD + 1):
                for page in list(node.lap_lists[level]):
                    cpu_id  = page.last_accessed_cpu
                    t_dcpmm = self.dcpmm_2 if (cpu_id == 0 or cpu_id is None) else self.dcpmm_3
                    self.move_logic(page, t_dcpmm)

    # ── Scan-driven: OPMX ─────────────────────────────────────────────────
    def opmx(self):
        done = 0
        for node in [self.dcpmm_2, self.dcpmm_3]:
            for level in range(HOT_THRESHOLD, 9):
                for page in list(node.lap_lists[level]):
                    if done >= MAX_MIGRATIONS_PER_SCAN: return
                    cpu_id = page.last_accessed_cpu
                    if cpu_id is None: continue
                    target_dram = self.dram_0 if cpu_id == 0 else self.dram_1
                    if page.current_node is target_dram: continue
                    if (page.access_history & 0b1) == 0: continue
                    if not target_dram.is_full():
                        if page.lap_level < (HOT_THRESHOLD + 1): continue
                        self.move_logic(page, target_dram)
                        done += 1
                    else:
                        victim = self.get_victim(self.dram_0, self.dram_1)
                        if victim is None: continue
                        if (page.lap_level - victim.lap_level) < MIGRATION_THRESHOLD: continue
                        v_cpu   = victim.last_accessed_cpu
                        v_dcpmm = self.dcpmm_2 if (v_cpu == 0 or v_cpu is None) else self.dcpmm_3
                        self.move_logic(victim, v_dcpmm)
                        self.move_logic(page,   target_dram)
                        done += 2
        for node in [self.dram_0, self.dram_1]:
            for level in range(0, COLD_THRESHOLD + 1):
                for page in list(node.lap_lists[level]):
                    if done >= MAX_MIGRATIONS_PER_SCAN: return
                    cpu_id  = page.last_accessed_cpu
                    t_dcpmm = self.dcpmm_2 if (cpu_id == 0 or cpu_id is None) else self.dcpmm_3
                    if (page.access_history & 0b11) == 0:
                        self.move_logic(page, t_dcpmm)
                        done += 1

    # ── Scan-driven: TPP-Alto ─────────────────────────────────────────────
    def tpp_alto(self, tpp_state):
        """
        Full tpp + tpp-alto kernel stack simulation.

        Topology: dram_0 = toptier, dram_1 = non-toptier (skip_node_one).
        Promotion: only ACTIVE pages pass; quota gate limits promotion rate.
        Demotion: only dram_0 cold pages are demoted; dram_1 is skipped.
        """
        DISABLE_MIGRATE    = tpp_state.get("disable_migrate", False)
        PAGE_PROMOTE_SCALE = tpp_state.get("page_promote_scale", 5)
        PAGE_CNT_LIMIT     = 10
        SKIP_NODE_ONE      = tpp_state.get("skip_node_one", True)

        def _is_toptier(node):
            return node is self.dram_0

        def _page_counters(pid):
            if pid not in tpp_state["page_cntrs"]:
                tpp_state["page_cntrs"][pid] = {"page_cnt": 0, "promote_cnt": 0}
            return tpp_state["page_cntrs"][pid]

        def _should_promote(page):
            if DISABLE_MIGRATE:
                return False
            if PAGE_PROMOTE_SCALE > PAGE_CNT_LIMIT:
                return True
            cntrs = _page_counters(page.page_id)
            if cntrs["promote_cnt"] < PAGE_PROMOTE_SCALE:
                cntrs["page_cnt"]    += 1
                cntrs["promote_cnt"] += 1
                if cntrs["page_cnt"] == PAGE_CNT_LIMIT:
                    cntrs["page_cnt"] = cntrs["promote_cnt"] = 0
                return True
            else:
                cntrs["page_cnt"] += 1
                if cntrs["page_cnt"] == PAGE_CNT_LIMIT:
                    cntrs["page_cnt"] = cntrs["promote_cnt"] = 0
                return False

        def _preferred_dram(cpu_id):
            if SKIP_NODE_ONE: return self.dram_0
            return self.dram_0 if cpu_id == 0 else self.dram_1

        def _demotion_target(cpu_id):
            if SKIP_NODE_ONE: return self.dcpmm_2
            return self.dcpmm_2 if (cpu_id == 0 or cpu_id is None) else self.dcpmm_3

        done = 0

        # Step 1: Un-scan toptier and skip-node-one pages
        for node in self.all_nodes:
            for page in list(node.pages):
                if _is_toptier(node):
                    page.prot_none = False
                elif SKIP_NODE_ONE and node is self.dram_1:
                    page.prot_none = False

        # Step 2: Promote hot ACTIVE pages from non-toptier → toptier
        for node in [self.dcpmm_2, self.dcpmm_3, self.dram_1]:
            for level in range(HOT_THRESHOLD, 9):
                for page in list(node.lap_lists[level]):
                    if done >= MAX_MIGRATIONS_PER_SCAN: break
                    if not page.is_active:              continue
                    cpu_id = page.last_accessed_cpu
                    if cpu_id is None:                  continue
                    target_dram = _preferred_dram(cpu_id)
                    if page.current_node is target_dram: continue
                    if (page.access_history & 0b1) == 0: continue
                    if not _should_promote(page):        continue
                    if not target_dram.is_full():
                        if page.lap_level < (HOT_THRESHOLD + 1): continue
                        page.pg_demoted = False
                        self.move_logic(page, target_dram)
                        done += 1
                    else:
                        victim = self.get_victim(self.dram_0, self.dram_1)
                        if victim is None: continue
                        if (page.lap_level - victim.lap_level) < MIGRATION_THRESHOLD: continue
                        v_cpu   = victim.last_accessed_cpu
                        v_dcpmm = _demotion_target(v_cpu)
                        victim.pg_demoted = True
                        self.move_logic(victim, v_dcpmm)
                        page.pg_demoted = False
                        self.move_logic(page, target_dram)
                        done += 2

        # Step 3: Demote cold pages (dram_0 only; dram_1 skipped)
        for node in [self.dram_0, self.dram_1]:
            for level in range(0, COLD_THRESHOLD + 1):
                for page in list(node.lap_lists[level]):
                    if done >= MAX_MIGRATIONS_PER_SCAN: return
                    if SKIP_NODE_ONE and node is self.dram_1: continue
                    cpu_id  = page.last_accessed_cpu
                    t_dcpmm = _demotion_target(cpu_id if cpu_id is not None else 0)
                    if (page.access_history & 0b11) == 0:
                        page.pg_demoted = True
                        self.move_logic(page, t_dcpmm)
                        done += 1

    # ── Scan-driven: ALTO-OPM v4 ─────────────────────────────────────────
    def alto_opm(self, kleio: KleioHotnessModel):
        done = 0
        all_pages      = [p for node in self.all_nodes for p in node.pages]
        ml_scores      = kleio.score_all(all_pages)
        pressure       = self._dram_pressure()
        promote_thresh = self._alto_promote_thresh()
        gap_thresh     = (ALTO_HIGH_PRESSURE_GAP
                          if pressure > ALTO_PRESSURE_LIMIT
                          else ALTO_MIN_SCORE_GAP)

        # ─────────────────────────────────────────────────────────────
# Promotion pass (DCPMM → DRAM) — "triple gate" filtering
# Goal: Only move truly important (hot + ML-approved) pages to fast memory
# ─────────────────────────────────────────────────────────────
        for node in [self.dcpmm_2, self.dcpmm_3]:   # iterate over slow memory (DCPMM nodes)
            for level in range(8, HOT_THRESHOLD - 1, -1):   # check hottest pages first (high LAP → low)
                for page in list(node.lap_lists[level]):    # iterate over pages in this hotness level

                    # Stop if migration budget for this scan is exhausted
                    if done >= ALTO_MAX_MIGRATIONS: break

                    cpu_id = page.last_accessed_cpu
                    if cpu_id is None: continue   # skip if no access history

                    # Find the DRAM node closest to the CPU that accessed this page
                    target_dram = self.dram_0 if cpu_id == 0 else self.dram_1

                    # Skip if page is already in correct DRAM
                    if page.current_node is target_dram: continue

                    # Get ML (Kleio) hotness score for this page
                    score = ml_scores.get(page.page_id, 0.0)

                    # ── GATE 1: ML threshold check ─────────────────────
                    # Only promote if ML says page is important enough
                    if score < promote_thresh:
                        self.stats.alto_ml_rejected += 1   # track rejection reason
                        continue

                    # ── GATE 2: Stability check (hot streak) ───────────
                    # Avoid promoting pages that are only temporarily hot
                    if page.hot_streak < HOT_STREAK_MIN:
                        self.stats.alto_streak_rejected += 1
                        continue

                    # ── If DRAM has space → direct promotion ───────────
                    if not target_dram.is_full():
                        self.move_logic(page, target_dram)  # move page to DRAM
                        done += 1                           # count migration

                    else:
                        # ── DRAM full → need to evict a victim ─────────
                        victim = self.get_victim(self.dram_0, self.dram_1)
                        if victim is None: continue

                        # ── GATE 3: Score gap check ───────────────────
                        # Only replace victim if new page is significantly better
                        if (score - ml_scores.get(victim.page_id, 0.0)) < gap_thresh:
                            self.stats.alto_gap_rejected += 1
                            continue

                        # Decide where to demote victim (back to slow memory)
                        v_cpu   = victim.last_accessed_cpu
                        v_dcpmm = self.dcpmm_2 if (v_cpu == 0 or v_cpu is None) else self.dcpmm_3

                        # Perform swap: victim → DCPMM, new page → DRAM
                        self.move_logic(victim, v_dcpmm)
                        self.move_logic(page, target_dram)
                        done += 2   # two migrations (one demotion + one promotion)


# ─────────────────────────────────────────────────────────────
# Demotion pass (DRAM → DCPMM) — "single loose gate"
# Goal: Remove cold/unimportant pages from DRAM
# ─────────────────────────────────────────────────────────────
        for node in [self.dram_0, self.dram_1]:   # iterate over fast memory (DRAM)
            for level in range(0, COLD_THRESHOLD + 1):   # check coldest pages first
                for page in list(node.lap_lists[level]):

                    # Stop if migration budget exceeded
                    if done >= ALTO_MAX_MIGRATIONS: break

                    # Get ML score
                    score = ml_scores.get(page.page_id, 0.0)

                    # ── Demotion gate (loose) ─────────────────────────
                    # Only keep page in DRAM if ML says it's still useful
                    if score > ALTO_ML_DEMOTE_THRESH:
                        self.stats.alto_ml_rejected += 1   # reject demotion
                        continue

                    # Choose correct DCPMM node based on CPU locality
                    cpu_id  = page.last_accessed_cpu
                    t_dcpmm = self.dcpmm_2 if (cpu_id == 0 or cpu_id is None) else self.dcpmm_3

                    # Move cold page from DRAM → DCPMM
                    self.move_logic(page, t_dcpmm)
                    done += 1


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation runner
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_STEPS = 20_000
MODES       = ["baseline", "cpm", "opm", "opmx", "tpp_alto", "alto_opm"]
PAGE_COUNTS = [100, 200, 300]
TOTAL_CAP   = 200

SCAN_DRIVEN  = {"opm", "opmx", "tpp_alto", "alto_opm"}
FAULT_DRIVEN = {"baseline", "cpm"}

results = {}
kleio   = KleioHotnessModel()

for total_pages in PAGE_COUNTS:
    results[total_pages] = {}
    half = total_pages // 2

    for mode in MODES:
        st  = MigrationStats()
        mem = MemorySystem(st)
        sch = AutoNUMAScheduler()
        mem.initialize_pages(total_pages)

        tpp_state = dict(TPP_ALTO_CONFIG)
        tpp_state["page_cntrs"] = {}

        outcomes    = {}
        latency_log = []
        dram_hits   = 0
        # TPP-Alto specific counters
        tpp_active_promotions = 0
        tpp_total_candidates  = 0

        random.seed(42)

        for step in range(TOTAL_STEPS):
            number  = random.randint(0, 99)
            cpu_sel = random.choice(mem.all_cpu)

            if number < 80:
                page_id = (random.randint(0, half - 1)
                           if cpu_sel.cpu_id == 0
                           else random.randint(half, total_pages - 1))
            else:
                page_id = (random.randint(half, total_pages - 1)
                           if cpu_sel.cpu_id == 0
                           else random.randint(0, half - 1))

            page = mem.page_table[page_id]
            if page.current_node is None:
                outcomes["unmapped"] = outcomes.get("unmapped", 0) + 1
                continue

            result = cpu_sel.access(page, sch)

            if page.current_node and page.current_node.tier.upper() == 'UPPER':
                dram_hits += 1

            if result[0] == "fault":
                outcome = mem.handle_numa_fault(
                    page, cpu_sel.cpu_id, result[1], sch, mode=mode)
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

                if mode in SCAN_DRIVEN and sch.access_count == 0:
                    promo_before = st.promotions
                    if   mode == "opm":      mem.opm()
                    elif mode == "opmx":     mem.opmx()
                    elif mode == "tpp_alto": mem.tpp_alto(tpp_state)
                    elif mode == "alto_opm": mem.alto_opm(kleio)
                    if mode == "tpp_alto":
                        tpp_active_promotions += (st.promotions - promo_before)
                        tpp_total_candidates  += sum(
                            1 for n in [mem.dcpmm_2, mem.dcpmm_3, mem.dram_1]
                            for lvl in range(HOT_THRESHOLD, 9)
                            for p in n.lap_lists[lvl])
            else:
                outcomes["normal"] = outcomes.get("normal", 0) + 1

            sch.tick(mem)

            if (step + 1) % 500 == 0:
                avg_lat = sum(c.avg_latency() for c in mem.all_cpu) / len(mem.all_cpu)
                latency_log.append(avg_lat)

        total_accesses = sum(c.total_accesses for c in mem.all_cpu)
        avg_lat_final  = sum(c.avg_latency()  for c in mem.all_cpu) / len(mem.all_cpu)
        dram_hit_ratio = dram_hits / total_accesses if total_accesses else 0

        if mode in FAULT_DRIVEN:
            mig_eff   = st.fault_driven_efficiency()
            eff_label = "fault_migrated/fault_eligible"
        else:
            mig_eff   = st.scan_driven_efficiency()
            eff_label = "promotions/(promotions+demotions)"

        results[total_pages][mode] = {
            "avg_latency"          : avg_lat_final,
            "total_migrations"     : st.total(),
            "promotions"           : st.promotions,
            "demotions"            : st.demotions,
            "lateral"              : st.lateral,
            "fault_count"          : st.fault_eligible,
            "fault_migrated"       : st.fault_migrated,
            "migration_efficiency" : mig_eff,
            "eff_label"            : eff_label,
            "dram_hit_ratio"       : dram_hit_ratio,
            "latency_log"          : latency_log,
            "outcomes"             : outcomes,
            "scan_generations"     : sch.current_generation,
            "alto_ml_rejected"     : st.alto_ml_rejected,
            "alto_gap_rejected"    : st.alto_gap_rejected,
            "alto_streak_rejected" : st.alto_streak_rejected,
            "alto_stab_rejected"   : st.alto_stab_rejected,
            "alto_total_rejected"  : st.alto_total_rejected(),
            "tpp_active_promotions": tpp_active_promotions,
            "tpp_total_candidates" : tpp_total_candidates,
        }

        alto_str = (f"  rejected=(ml:{st.alto_ml_rejected} "
                    f"gap:{st.alto_gap_rejected} streak:{st.alto_streak_rejected})"
                    if mode == "alto_opm" else "")
        print(f"[pages={total_pages:3d} | mode={mode:10s}] "
              f"mig={st.total():5d} (↑{st.promotions} ↓{st.demotions} ↔{st.lateral})  "
              f"avg_lat={avg_lat_final:.1f}ns  "
              f"dram={dram_hit_ratio:.3f}  "
              f"eff={mig_eff:.3f} [{eff_label}]{alto_str}")

os.makedirs("outputs", exist_ok=True)
with open("outputs/sim_results_v5.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nJSON saved → outputs/sim_results_v5.json")


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "baseline" : "#e15759",
    "cpm"      : "#f28e2b",
    "opm"      : "#4e79a7",
    "opmx"     : "#59a14f",
    "tpp_alto" : "#76b7b2",   # teal — new
    "alto_opm" : "#b07aa1",
}
LABELS = {
    "baseline" : "BASELINE",
    "cpm"      : "CPM",
    "opm"      : "OPM",
    "opmx"     : "OPMX",
    "tpp_alto" : "TPP-ALTO",
    "alto_opm" : "ALTO-OPM",
}
PAGE_LABELS = {
    100: "Under-pressure\n(100 pages)",
    200: "At-capacity\n(200 pages)",
    300: "Over-pressure\n(300 pages)",
}

x       = np.arange(len(PAGE_COUNTS))
width   = 0.12
offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * width
ls_map  = {
    "baseline" : "-",
    "cpm"      : "--",
    "opm"      : "-.",
    "opmx"     : ":",
    "tpp_alto" : (0, (5, 1)),
    "alto_opm" : (0, (3, 1, 1, 1)),
}


def grouped_bar(ax, metric_key, title, ylabel, fmt=".1f"):
    for mode, off in zip(MODES, offsets):
        vals = [results[p][mode][metric_key] for p in PAGE_COUNTS]
        bars = ax.bar(x + off, vals, width, label=LABELS[mode],
                      color=COLORS[mode], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{v:{fmt}}", ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(axis='y', alpha=0.3)


# ── Graph 1: Main 3×3 overview ───────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(26, 20))
fig.suptitle(
    "AutoTiering Simulation v5 — All 6 Algorithms\n"
    "BASELINE · CPM · OPM · OPMX · TPP-ALTO · ALTO-OPM",
    fontsize=11, fontweight='bold')

# 1 · Avg Latency
grouped_bar(axes[0][0], "avg_latency", "1 · Average Latency (ns)", "Latency (ns)")

# 2 · Migration breakdown
ax2   = axes[0][1]
bar_w = width * 0.45
for mode, off in zip(MODES, offsets):
    proms = [results[p][mode]["promotions"] for p in PAGE_COUNTS]
    dems  = [results[p][mode]["demotions"]  for p in PAGE_COUNTS]
    ax2.bar(x + off - bar_w/2, proms, bar_w, color=COLORS[mode], alpha=0.9)
    ax2.bar(x + off + bar_w/2, dems,  bar_w, color=COLORS[mode], alpha=0.45, hatch='//')
ax2.set_xticks(x)
ax2.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax2.set_title("2 · Total Migrations\n(solid=promotions, hatched=demotions)", fontweight='bold', pad=8)
ax2.set_ylabel("# Migrations")
legend_els  = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in MODES]
legend_els += [Patch(facecolor='grey', label='↑ Promotions'),
               Patch(facecolor='grey', hatch='//', alpha=0.4, label='↓ Demotions')]
ax2.legend(handles=legend_els, fontsize=5.5, ncol=2)
ax2.grid(axis='y', alpha=0.3)

# 3 · Eligible NUMA Faults
grouped_bar(axes[0][2], "fault_count",
            "3 · Eligible NUMA Faults\n(above threshold, non-local)", "# Faults", fmt=".0f")

# 4A · Fault-driven Efficiency
ax4a = axes[1][0]
fw = width * 1.2
for mode, off in zip(["baseline", "cpm"], [-fw/2, fw/2]):
    vals = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    bars = ax4a.bar(x + off, vals, fw, label=LABELS[mode],
                    color=COLORS[mode], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax4a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.005, f"{v:.3f}", ha='center', fontsize=8)
ax4a.set_xticks(x)
ax4a.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax4a.set_title("4A · Fault-driven Efficiency\n(migrated / eligible faults)", fontweight='bold')
ax4a.set_ylabel("Efficiency (0–1)")
ax4a.legend()
ax4a.grid(axis='y', alpha=0.3)

# 4B · Scan-driven Efficiency — OPM vs OPMX vs TPP-ALTO vs ALTO-OPM
ax4b = axes[1][1]
sw   = width
scan_modes = ["opm", "opmx", "tpp_alto", "alto_opm"]
for mode, off in zip(scan_modes, np.array([-1.5, -0.5, 0.5, 1.5]) * sw):
    vals = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    bars = ax4b.bar(x + off, vals, sw, label=LABELS[mode],
                    color=COLORS[mode], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax4b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.01, f"{v:.3f}", ha='center', fontsize=7)
ax4b.set_xticks(x)
ax4b.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax4b.set_ylim(0, 1.15)
ax4b.axhline(0.5, color='red', linestyle='--', linewidth=1, label='0.5 baseline')
ax4b.set_title("4B · Scan-driven Efficiency ★\n(promotions / (promo + demo))", fontweight='bold')
ax4b.set_ylabel("Efficiency (0–1)")
ax4b.legend(fontsize=7)
ax4b.grid(axis='y', alpha=0.3)

# 4C · ALTO gate rejection breakdown (stacked)
ax4c = axes[1][2]
alto_keys   = ["alto_ml_rejected", "alto_gap_rejected", "alto_streak_rejected"]
alto_labels = ["ML threshold", "Score gap", "Hot-streak"]
alto_colors = ["#d4a373", "#ccd5ae", "#a2d2ff"]
bar_bottom  = np.zeros(len(PAGE_COUNTS))
for key, lbl, col in zip(alto_keys, alto_labels, alto_colors):
    vals = np.array([results[p]["alto_opm"][key] for p in PAGE_COUNTS], dtype=float)
    ax4c.bar(x, vals, 0.4, bottom=bar_bottom, label=lbl, color=col, edgecolor='white')
    bar_bottom += vals
ax4c.set_xticks(x)
ax4c.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax4c.set_title("4C · ALTO-OPM Gate Rejections\n(stacked by reason)", fontweight='bold')
ax4c.set_ylabel("# Moves Blocked")
ax4c.legend(fontsize=8)
ax4c.grid(axis='y', alpha=0.3)

# 5 · Latency over time
ax5 = axes[2][0]
for pi, pages in enumerate(PAGE_COUNTS):
    alpha = 0.5 + 0.25 * pi
    for mode in MODES:
        log   = results[pages][mode]["latency_log"]
        steps = [(i + 1) * 500 for i in range(len(log))]
        ax5.plot(steps, log, linestyle=ls_map[mode], color=COLORS[mode],
                 alpha=alpha, linewidth=1.3,
                 label=f"{pages}p {LABELS[mode]}" if pi == 0 else "_")
mode_handles = [plt.Line2D([0],[0], color=COLORS[m], linewidth=2,
                            linestyle=ls_map[m], label=LABELS[m]) for m in MODES]
pg_handles   = [plt.Line2D([0],[0], color='grey', linewidth=1+pi,
                            alpha=0.5+0.25*pi, label=f"{p} pages")
                for pi, p in enumerate(PAGE_COUNTS)]
ax5.legend(handles=mode_handles + pg_handles, fontsize=5.5, ncol=2)
ax5.set_xlabel("Simulation Step")
ax5.set_ylabel("Avg Latency (ns)")
ax5.set_title("5 · Latency over Time", fontweight='bold', pad=8)
ax5.grid(alpha=0.3)

# 6 · DRAM Hit Ratio
grouped_bar(axes[2][1], "dram_hit_ratio",
            "6 · DRAM Hit Ratio\n(accesses served from upper tier)",
            "Ratio (0–1)", fmt=".3f")

# 7 · Efficiency vs Migrations scatter (scan-driven only)
ax7 = axes[2][2]
for mode in ["opm", "opmx", "tpp_alto", "alto_opm"]:
    xs = [results[p][mode]["total_migrations"]     for p in PAGE_COUNTS]
    ys = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    ax7.scatter(xs, ys, color=COLORS[mode], s=100, label=LABELS[mode], zorder=5)
    for xi, yi, p in zip(xs, ys, PAGE_COUNTS):
        ax7.annotate(f"{p}p", (xi, yi),
                     textcoords="offset points", xytext=(4, 3), fontsize=7)
ax7.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax7.set_xlabel("Total Migrations")
ax7.set_ylabel("Scan-driven Efficiency")
ax7.set_title("7 · Efficiency vs Migrations\n(ideal: upper-left = high eff, low churn)",
              fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(alpha=0.3)

plt.tight_layout()
out1 = "outputs/autoTiering_metrics_v5.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Graph 1 (main overview) → {out1}")
plt.close()


# ── Graph 2: Promotion vs Demotion line chart (all 6 modes) ─────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle("Scan-driven Move Breakdown: Promotions vs Demotions — all 6 modes",
              fontweight='bold')
for ax, metric, label in zip(axes2,
                              ["promotions", "demotions"],
                              ["↑ Promotions (DCPMM→DRAM)", "↓ Demotions (DRAM→DCPMM)"]):
    for mode in MODES:
        vals = [results[p][mode][metric] for p in PAGE_COUNTS]
        ax.plot(PAGE_COUNTS, vals, marker='o', linewidth=2,
                linestyle=ls_map[mode], color=COLORS[mode], label=LABELS[mode])
        for px, vy in zip(PAGE_COUNTS, vals):
            ax.annotate(str(vy), (px, vy),
                        textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8)
    ax.axvline(TOTAL_CAP, color='red', linestyle='--', linewidth=1.1,
               label=f"Total cap ({TOTAL_CAP})")
    ax.set_xticks(PAGE_COUNTS)
    ax.set_xticklabels(["100\n(under)", "200\n(at cap)", "300\n(over)"])
    ax.set_xlabel("Total Pages")
    ax.set_ylabel("Count")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
out2 = "outputs/promo_vs_demo_v5.png"
plt.savefig(out2, dpi=150)
print(f"Graph 2 (promo/demo) → {out2}")
plt.close()


# ── Graph 3: ALTO-OPM rejection pie charts ───────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle("ALTO-OPM v4: Gate Rejection Breakdown (ML / Gap / Hot-streak)",
              fontweight='bold')
for ax, p in zip(axes3, PAGE_COUNTS):
    r      = results[p]["alto_opm"]
    sizes  = [r["alto_ml_rejected"], r["alto_gap_rejected"], r["alto_streak_rejected"]]
    labels = [f"ML thresh\n({r['alto_ml_rejected']})",
              f"Score gap\n({r['alto_gap_rejected']})",
              f"Hot-streak\n({r['alto_streak_rejected']})"]
    total_rej = sum(sizes)
    if total_rej == 0:
        ax.text(0.5, 0.5, "No rejections", ha='center', va='center',
                transform=ax.transAxes)
    else:
        ax.pie(sizes, labels=labels, colors=["#d4a373", "#ccd5ae", "#a2d2ff"],
               autopct='%1.1f%%', startangle=90)
    ax.set_title(f"{PAGE_LABELS[p]}\n(total rejected: {total_rej})")
plt.tight_layout()
out3 = "outputs/alto_rejection_breakdown_v5.png"
plt.savefig(out3, dpi=150)
print(f"Graph 3 (ALTO rejection) → {out3}")
plt.close()


# ── Graph 4: TPP-Alto vs ALTO-OPM head-to-head ───────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
fig4.suptitle("Graph 4 · TPP-ALTO vs ALTO-OPM: Head-to-Head Comparison",
              fontweight='bold')
metrics_hth = [
    ("avg_latency",          "Average Latency (ns)",           ".1f"),
    ("migration_efficiency", "Scan-driven Efficiency",         ".3f"),
    ("dram_hit_ratio",       "DRAM Hit Ratio",                 ".3f"),
]
for ax, (mkey, mtitle, mfmt) in zip(axes4, metrics_hth):
    bw  = 0.28
    for mode, off in zip(["tpp_alto", "alto_opm"], [-bw/2, bw/2]):
        vals = [results[p][mode][mkey] for p in PAGE_COUNTS]
        bars = ax.bar(x + off, vals, bw, label=LABELS[mode],
                      color=COLORS[mode], edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02, f"{v:{mfmt}}",
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
    ax.set_title(mtitle, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out4 = "outputs/tpp_vs_alto_headtohead_v5.png"
plt.savefig(out4, dpi=150)
print(f"Graph 4 (TPP vs ALTO head-to-head) → {out4}")
plt.close()


# ── Graph 5: TPP-Alto active-page promotion analysis ─────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
fig5.suptitle("Graph 5 · TPP-ALTO: Active-page Filtering & Promotion Quota Analysis",
              fontweight='bold')

# 5a: Promotions vs total migrations (TPP-Alto)
ax5a = axes5[0]
tpp_proms  = [results[p]["tpp_alto"]["promotions"]       for p in PAGE_COUNTS]
tpp_total  = [results[p]["tpp_alto"]["total_migrations"] for p in PAGE_COUNTS]
tpp_dems   = [results[p]["tpp_alto"]["demotions"]        for p in PAGE_COUNTS]
bw = 0.25
bars_p = ax5a.bar(x - bw, tpp_proms, bw, label="Promotions", color=COLORS["tpp_alto"], alpha=0.9)
bars_d = ax5a.bar(x,       tpp_dems,  bw, label="Demotions",  color=COLORS["tpp_alto"], alpha=0.5, hatch="//")
bars_t = ax5a.bar(x + bw,  tpp_total, bw, label="Total mig.",  color="#555555", alpha=0.6)
for bars, vals in [(bars_p, tpp_proms), (bars_d, tpp_dems), (bars_t, tpp_total)]:
    for bar, v in zip(bars, vals):
        ax5a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() * 1.02, str(v),
                  ha='center', va='bottom', fontsize=8)
ax5a.set_xticks(x)
ax5a.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax5a.set_title("TPP-ALTO: Promotion / Demotion / Total", fontweight='bold')
ax5a.set_ylabel("# Migrations")
ax5a.legend(fontsize=8)
ax5a.grid(axis='y', alpha=0.3)

# 5b: TPP-Alto efficiency vs all scan-driven algorithms
ax5b = axes5[1]
for mode in ["opm", "opmx", "tpp_alto", "alto_opm"]:
    vals = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    ax5b.plot(PAGE_COUNTS, vals, marker='o', linewidth=2,
              linestyle=ls_map[mode], color=COLORS[mode], label=LABELS[mode])
    for px, vy in zip(PAGE_COUNTS, vals):
        ax5b.annotate(f"{vy:.3f}", (px, vy),
                      textcoords="offset points", xytext=(0, 8),
                      ha='center', fontsize=7.5)
ax5b.axhline(0.5, color='red', linestyle='--', linewidth=1, label='0.5 baseline')
ax5b.set_xticks(PAGE_COUNTS)
ax5b.set_xticklabels(["100\n(under)", "200\n(at cap)", "300\n(over)"])
ax5b.set_xlabel("Total Pages")
ax5b.set_ylabel("Scan-driven Efficiency")
ax5b.set_title("Scan-driven Efficiency: All 4 Scan Algorithms", fontweight='bold')
ax5b.legend(fontsize=8)
ax5b.grid(alpha=0.3)
ax5b.set_ylim(0, 1.1)

plt.tight_layout()
out5 = "outputs/tpp_alto_analysis_v5.png"
plt.savefig(out5, dpi=150)
print(f"Graph 5 (TPP-ALTO analysis) → {out5}")
plt.close()


# ── Graph 6: Six-algorithm radar / spider chart ───────────────────────────────
metrics_radar = ["avg_latency", "migration_efficiency", "dram_hit_ratio",
                 "total_migrations", "fault_count"]
metric_labels = ["Avg Latency\n(lower=better)", "Mig Efficiency\n(higher=better)",
                 "DRAM Hit Ratio\n(higher=better)", "Total Mig\n(lower=better)",
                 "Fault Count\n(lower=better)"]

fig6 = plt.figure(figsize=(19, 6))
fig6.suptitle("Graph 6 · All Algorithms — Metric Radar (normalised per pressure scenario)",
              fontweight='bold')

for idx, pc in enumerate(PAGE_COUNTS):
    ax = fig6.add_subplot(1, 3, idx + 1, projection='polar')

    raw = {m: {mode: results[pc][mode][m] for mode in MODES} for m in metrics_radar}
    invert = {"avg_latency", "total_migrations", "fault_count"}
    norm = {}
    for m in metrics_radar:
        vals = list(raw[m].values())
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1
        for mode in MODES:
            v = (raw[m][mode] - lo) / span
            norm.setdefault(mode, {})[m] = (1 - v) if m in invert else v

    n_vars = len(metrics_radar)
    angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=6.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=5.5)
    ax.set_ylim(0, 1)

    for mode in MODES:
        values = [norm[mode][m] for m in metrics_radar]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid',
                color=COLORS[mode], label=LABELS[mode])
        ax.fill(angles, values, color=COLORS[mode], alpha=0.08)

    ax.set_title(PAGE_LABELS[pc], fontweight='bold', size=9, pad=14)
    if idx == 2:
        ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=7)

plt.tight_layout()
out6 = "outputs/radar_all_algorithms_v5.png"
plt.savefig(out6, dpi=150, bbox_inches='tight')
print(f"Graph 6 (radar chart) → {out6}")
plt.close()

print("\nAll done. Outputs in ./outputs/")
print("Files generated:")
for f in [out1, out2, out3, out4, out5, out6]:
    print(f"  {f}")                    self.stats.alto_ml_rejected += 1   # track rejection reason
                    continue
    
                # ── GATE 2: Stability check (hot streak) ───────────
                # Avoid promoting pages that are only temporarily hot
                if page.hot_streak < HOT_STREAK_MIN:
                    self.stats.alto_streak_rejected += 1
                    continue
    
                # ── If DRAM has space → direct promotion ───────────
                if not target_dram.is_full():
                    self.move_logic(page, target_dram)  # move page to DRAM
                    done += 1                           # count migration
    
                else:
                    # ── DRAM full → need to evict a victim ─────────
                    victim = self.get_victim(self.dram_0, self.dram_1)
                    if victim is None: continue
    
                    # ── GATE 3: Score gap check ───────────────────
                    # Only replace victim if new page is significantly better
                    if (score - ml_scores.get(victim.page_id, 0.0)) < gap_thresh:
                        self.stats.alto_gap_rejected += 1
                        continue
    
                    # Decide where to demote victim (back to slow memory)
                    v_cpu   = victim.last_accessed_cpu
                    v_dcpmm = self.dcpmm_2 if (v_cpu == 0 or v_cpu is None) else self.dcpmm_3
    
                    # Perform swap: victim → DCPMM, new page → DRAM
                    self.move_logic(victim, v_dcpmm)
                    self.move_logic(page, target_dram)
                    done += 2   # two migrations (one demotion + one promotion)
    
    
    # ─────────────────────────────────────────────────────────────
    # Demotion pass (DRAM → DCPMM) — "single loose gate"
    # Goal: Remove cold/unimportant pages from DRAM
    # ─────────────────────────────────────────────────────────────
    for node in [self.dram_0, self.dram_1]:   # iterate over fast memory (DRAM)
        for level in range(0, COLD_THRESHOLD + 1):   # check coldest pages first
            for page in list(node.lap_lists[level]):
    
                # Stop if migration budget exceeded
                if done >= ALTO_MAX_MIGRATIONS: break
    
                # Get ML score
                score = ml_scores.get(page.page_id, 0.0)
    
                # ── Demotion gate (loose) ─────────────────────────
                # Only keep page in DRAM if ML says it's still useful
                if score > ALTO_ML_DEMOTE_THRESH:
                    self.stats.alto_ml_rejected += 1   # reject demotion
                    continue
    
                # Choose correct DCPMM node based on CPU locality
                cpu_id  = page.last_accessed_cpu
                t_dcpmm = self.dcpmm_2 if (cpu_id == 0 or cpu_id is None) else self.dcpmm_3
    
                # Move cold page from DRAM → DCPMM
                self.move_logic(page, t_dcpmm)
                done += 1


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation runner
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_STEPS = 20_000
MODES       = ["baseline", "cpm", "opm", "opmx", "tpp_alto", "alto_opm"]
PAGE_COUNTS = [100, 200, 300]
TOTAL_CAP   = 200

SCAN_DRIVEN  = {"opm", "opmx", "tpp_alto", "alto_opm"}
FAULT_DRIVEN = {"baseline", "cpm"}

results = {}
kleio   = KleioHotnessModel()

for total_pages in PAGE_COUNTS:
    results[total_pages] = {}
    half = total_pages // 2

    for mode in MODES:
        st  = MigrationStats()
        mem = MemorySystem(st)
        sch = AutoNUMAScheduler()
        mem.initialize_pages(total_pages)

        tpp_state = dict(TPP_ALTO_CONFIG)
        tpp_state["page_cntrs"] = {}

        outcomes    = {}
        latency_log = []
        dram_hits   = 0
        # TPP-Alto specific counters
        tpp_active_promotions = 0
        tpp_total_candidates  = 0

        random.seed(42)

        for step in range(TOTAL_STEPS):
            number  = random.randint(0, 99)
            cpu_sel = random.choice(mem.all_cpu)

            if number < 80:
                page_id = (random.randint(0, half - 1)
                           if cpu_sel.cpu_id == 0
                           else random.randint(half, total_pages - 1))
            else:
                page_id = (random.randint(half, total_pages - 1)
                           if cpu_sel.cpu_id == 0
                           else random.randint(0, half - 1))

            page = mem.page_table[page_id]
            if page.current_node is None:
                outcomes["unmapped"] = outcomes.get("unmapped", 0) + 1
                continue

            result = cpu_sel.access(page, sch)

            if page.current_node and page.current_node.tier.upper() == 'UPPER':
                dram_hits += 1

            if result[0] == "fault":
                outcome = mem.handle_numa_fault(
                    page, cpu_sel.cpu_id, result[1], sch, mode=mode)
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

                if mode in SCAN_DRIVEN and sch.access_count == 0:
                    promo_before = st.promotions
                    if   mode == "opm":      mem.opm()
                    elif mode == "opmx":     mem.opmx()
                    elif mode == "tpp_alto": mem.tpp_alto(tpp_state)
                    elif mode == "alto_opm": mem.alto_opm(kleio)
                    if mode == "tpp_alto":
                        tpp_active_promotions += (st.promotions - promo_before)
                        tpp_total_candidates  += sum(
                            1 for n in [mem.dcpmm_2, mem.dcpmm_3, mem.dram_1]
                            for lvl in range(HOT_THRESHOLD, 9)
                            for p in n.lap_lists[lvl])
            else:
                outcomes["normal"] = outcomes.get("normal", 0) + 1

            sch.tick(mem)

            if (step + 1) % 500 == 0:
                avg_lat = sum(c.avg_latency() for c in mem.all_cpu) / len(mem.all_cpu)
                latency_log.append(avg_lat)

        total_accesses = sum(c.total_accesses for c in mem.all_cpu)
        avg_lat_final  = sum(c.avg_latency()  for c in mem.all_cpu) / len(mem.all_cpu)
        dram_hit_ratio = dram_hits / total_accesses if total_accesses else 0

        if mode in FAULT_DRIVEN:
            mig_eff   = st.fault_driven_efficiency()
            eff_label = "fault_migrated/fault_eligible"
        else:
            mig_eff   = st.scan_driven_efficiency()
            eff_label = "promotions/(promotions+demotions)"

        results[total_pages][mode] = {
            "avg_latency"          : avg_lat_final,
            "total_migrations"     : st.total(),
            "promotions"           : st.promotions,
            "demotions"            : st.demotions,
            "lateral"              : st.lateral,
            "fault_count"          : st.fault_eligible,
            "fault_migrated"       : st.fault_migrated,
            "migration_efficiency" : mig_eff,
            "eff_label"            : eff_label,
            "dram_hit_ratio"       : dram_hit_ratio,
            "latency_log"          : latency_log,
            "outcomes"             : outcomes,
            "scan_generations"     : sch.current_generation,
            "alto_ml_rejected"     : st.alto_ml_rejected,
            "alto_gap_rejected"    : st.alto_gap_rejected,
            "alto_streak_rejected" : st.alto_streak_rejected,
            "alto_stab_rejected"   : st.alto_stab_rejected,
            "alto_total_rejected"  : st.alto_total_rejected(),
            "tpp_active_promotions": tpp_active_promotions,
            "tpp_total_candidates" : tpp_total_candidates,
        }

        alto_str = (f"  rejected=(ml:{st.alto_ml_rejected} "
                    f"gap:{st.alto_gap_rejected} streak:{st.alto_streak_rejected})"
                    if mode == "alto_opm" else "")
        print(f"[pages={total_pages:3d} | mode={mode:10s}] "
              f"mig={st.total():5d} (↑{st.promotions} ↓{st.demotions} ↔{st.lateral})  "
              f"avg_lat={avg_lat_final:.1f}ns  "
              f"dram={dram_hit_ratio:.3f}  "
              f"eff={mig_eff:.3f} [{eff_label}]{alto_str}")

os.makedirs("outputs", exist_ok=True)
with open("outputs/sim_results_v5.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nJSON saved → outputs/sim_results_v5.json")


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "baseline" : "#e15759",
    "cpm"      : "#f28e2b",
    "opm"      : "#4e79a7",
    "opmx"     : "#59a14f",
    "tpp_alto" : "#76b7b2",   # teal — new
    "alto_opm" : "#b07aa1",
}
LABELS = {
    "baseline" : "BASELINE",
    "cpm"      : "CPM",
    "opm"      : "OPM",
    "opmx"     : "OPMX",
    "tpp_alto" : "TPP-ALTO",
    "alto_opm" : "ALTO-OPM",
}
PAGE_LABELS = {
    100: "Under-pressure\n(100 pages)",
    200: "At-capacity\n(200 pages)",
    300: "Over-pressure\n(300 pages)",
}

x       = np.arange(len(PAGE_COUNTS))
width   = 0.12
offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * width
ls_map  = {
    "baseline" : "-",
    "cpm"      : "--",
    "opm"      : "-.",
    "opmx"     : ":",
    "tpp_alto" : (0, (5, 1)),
    "alto_opm" : (0, (3, 1, 1, 1)),
}


def grouped_bar(ax, metric_key, title, ylabel, fmt=".1f"):
    for mode, off in zip(MODES, offsets):
        vals = [results[p][mode][metric_key] for p in PAGE_COUNTS]
        bars = ax.bar(x + off, vals, width, label=LABELS[mode],
                      color=COLORS[mode], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{v:{fmt}}", ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(axis='y', alpha=0.3)


# ── Graph 1: Main 3×3 overview ───────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(26, 20))
fig.suptitle(
    "AutoTiering Simulation v5 — All 6 Algorithms\n"
    "BASELINE · CPM · OPM · OPMX · TPP-ALTO · ALTO-OPM",
    fontsize=11, fontweight='bold')

# 1 · Avg Latency
grouped_bar(axes[0][0], "avg_latency", "1 · Average Latency (ns)", "Latency (ns)")

# 2 · Migration breakdown
ax2   = axes[0][1]
bar_w = width * 0.45
for mode, off in zip(MODES, offsets):
    proms = [results[p][mode]["promotions"] for p in PAGE_COUNTS]
    dems  = [results[p][mode]["demotions"]  for p in PAGE_COUNTS]
    ax2.bar(x + off - bar_w/2, proms, bar_w, color=COLORS[mode], alpha=0.9)
    ax2.bar(x + off + bar_w/2, dems,  bar_w, color=COLORS[mode], alpha=0.45, hatch='//')
ax2.set_xticks(x)
ax2.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax2.set_title("2 · Total Migrations\n(solid=promotions, hatched=demotions)", fontweight='bold', pad=8)
ax2.set_ylabel("# Migrations")
legend_els  = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in MODES]
legend_els += [Patch(facecolor='grey', label='↑ Promotions'),
               Patch(facecolor='grey', hatch='//', alpha=0.4, label='↓ Demotions')]
ax2.legend(handles=legend_els, fontsize=5.5, ncol=2)
ax2.grid(axis='y', alpha=0.3)

# 3 · Eligible NUMA Faults
grouped_bar(axes[0][2], "fault_count",
            "3 · Eligible NUMA Faults\n(above threshold, non-local)", "# Faults", fmt=".0f")

# 4A · Fault-driven Efficiency
ax4a = axes[1][0]
fw = width * 1.2
for mode, off in zip(["baseline", "cpm"], [-fw/2, fw/2]):
    vals = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    bars = ax4a.bar(x + off, vals, fw, label=LABELS[mode],
                    color=COLORS[mode], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax4a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.005, f"{v:.3f}", ha='center', fontsize=8)
ax4a.set_xticks(x)
ax4a.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax4a.set_title("4A · Fault-driven Efficiency\n(migrated / eligible faults)", fontweight='bold')
ax4a.set_ylabel("Efficiency (0–1)")
ax4a.legend()
ax4a.grid(axis='y', alpha=0.3)

# 4B · Scan-driven Efficiency — OPM vs OPMX vs TPP-ALTO vs ALTO-OPM
ax4b = axes[1][1]
sw   = width
scan_modes = ["opm", "opmx", "tpp_alto", "alto_opm"]
for mode, off in zip(scan_modes, np.array([-1.5, -0.5, 0.5, 1.5]) * sw):
    vals = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    bars = ax4b.bar(x + off, vals, sw, label=LABELS[mode],
                    color=COLORS[mode], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax4b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.01, f"{v:.3f}", ha='center', fontsize=7)
ax4b.set_xticks(x)
ax4b.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax4b.set_ylim(0, 1.15)
ax4b.axhline(0.5, color='red', linestyle='--', linewidth=1, label='0.5 baseline')
ax4b.set_title("4B · Scan-driven Efficiency ★\n(promotions / (promo + demo))", fontweight='bold')
ax4b.set_ylabel("Efficiency (0–1)")
ax4b.legend(fontsize=7)
ax4b.grid(axis='y', alpha=0.3)

# 4C · ALTO gate rejection breakdown (stacked)
ax4c = axes[1][2]
alto_keys   = ["alto_ml_rejected", "alto_gap_rejected", "alto_streak_rejected"]
alto_labels = ["ML threshold", "Score gap", "Hot-streak"]
alto_colors = ["#d4a373", "#ccd5ae", "#a2d2ff"]
bar_bottom  = np.zeros(len(PAGE_COUNTS))
for key, lbl, col in zip(alto_keys, alto_labels, alto_colors):
    vals = np.array([results[p]["alto_opm"][key] for p in PAGE_COUNTS], dtype=float)
    ax4c.bar(x, vals, 0.4, bottom=bar_bottom, label=lbl, color=col, edgecolor='white')
    bar_bottom += vals
ax4c.set_xticks(x)
ax4c.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax4c.set_title("4C · ALTO-OPM Gate Rejections\n(stacked by reason)", fontweight='bold')
ax4c.set_ylabel("# Moves Blocked")
ax4c.legend(fontsize=8)
ax4c.grid(axis='y', alpha=0.3)

# 5 · Latency over time
ax5 = axes[2][0]
for pi, pages in enumerate(PAGE_COUNTS):
    alpha = 0.5 + 0.25 * pi
    for mode in MODES:
        log   = results[pages][mode]["latency_log"]
        steps = [(i + 1) * 500 for i in range(len(log))]
        ax5.plot(steps, log, linestyle=ls_map[mode], color=COLORS[mode],
                 alpha=alpha, linewidth=1.3,
                 label=f"{pages}p {LABELS[mode]}" if pi == 0 else "_")
mode_handles = [plt.Line2D([0],[0], color=COLORS[m], linewidth=2,
                            linestyle=ls_map[m], label=LABELS[m]) for m in MODES]
pg_handles   = [plt.Line2D([0],[0], color='grey', linewidth=1+pi,
                            alpha=0.5+0.25*pi, label=f"{p} pages")
                for pi, p in enumerate(PAGE_COUNTS)]
ax5.legend(handles=mode_handles + pg_handles, fontsize=5.5, ncol=2)
ax5.set_xlabel("Simulation Step")
ax5.set_ylabel("Avg Latency (ns)")
ax5.set_title("5 · Latency over Time", fontweight='bold', pad=8)
ax5.grid(alpha=0.3)

# 6 · DRAM Hit Ratio
grouped_bar(axes[2][1], "dram_hit_ratio",
            "6 · DRAM Hit Ratio\n(accesses served from upper tier)",
            "Ratio (0–1)", fmt=".3f")

# 7 · Efficiency vs Migrations scatter (scan-driven only)
ax7 = axes[2][2]
for mode in ["opm", "opmx", "tpp_alto", "alto_opm"]:
    xs = [results[p][mode]["total_migrations"]     for p in PAGE_COUNTS]
    ys = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    ax7.scatter(xs, ys, color=COLORS[mode], s=100, label=LABELS[mode], zorder=5)
    for xi, yi, p in zip(xs, ys, PAGE_COUNTS):
        ax7.annotate(f"{p}p", (xi, yi),
                     textcoords="offset points", xytext=(4, 3), fontsize=7)
ax7.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax7.set_xlabel("Total Migrations")
ax7.set_ylabel("Scan-driven Efficiency")
ax7.set_title("7 · Efficiency vs Migrations\n(ideal: upper-left = high eff, low churn)",
              fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(alpha=0.3)

plt.tight_layout()
out1 = "outputs/autoTiering_metrics_v5.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Graph 1 (main overview) → {out1}")
plt.close()


# ── Graph 2: Promotion vs Demotion line chart (all 6 modes) ─────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle("Scan-driven Move Breakdown: Promotions vs Demotions — all 6 modes",
              fontweight='bold')
for ax, metric, label in zip(axes2,
                              ["promotions", "demotions"],
                              ["↑ Promotions (DCPMM→DRAM)", "↓ Demotions (DRAM→DCPMM)"]):
    for mode in MODES:
        vals = [results[p][mode][metric] for p in PAGE_COUNTS]
        ax.plot(PAGE_COUNTS, vals, marker='o', linewidth=2,
                linestyle=ls_map[mode], color=COLORS[mode], label=LABELS[mode])
        for px, vy in zip(PAGE_COUNTS, vals):
            ax.annotate(str(vy), (px, vy),
                        textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8)
    ax.axvline(TOTAL_CAP, color='red', linestyle='--', linewidth=1.1,
               label=f"Total cap ({TOTAL_CAP})")
    ax.set_xticks(PAGE_COUNTS)
    ax.set_xticklabels(["100\n(under)", "200\n(at cap)", "300\n(over)"])
    ax.set_xlabel("Total Pages")
    ax.set_ylabel("Count")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
out2 = "outputs/promo_vs_demo_v5.png"
plt.savefig(out2, dpi=150)
print(f"Graph 2 (promo/demo) → {out2}")
plt.close()


# ── Graph 3: ALTO-OPM rejection pie charts ───────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle("ALTO-OPM v4: Gate Rejection Breakdown (ML / Gap / Hot-streak)",
              fontweight='bold')
for ax, p in zip(axes3, PAGE_COUNTS):
    r      = results[p]["alto_opm"]
    sizes  = [r["alto_ml_rejected"], r["alto_gap_rejected"], r["alto_streak_rejected"]]
    labels = [f"ML thresh\n({r['alto_ml_rejected']})",
              f"Score gap\n({r['alto_gap_rejected']})",
              f"Hot-streak\n({r['alto_streak_rejected']})"]
    total_rej = sum(sizes)
    if total_rej == 0:
        ax.text(0.5, 0.5, "No rejections", ha='center', va='center',
                transform=ax.transAxes)
    else:
        ax.pie(sizes, labels=labels, colors=["#d4a373", "#ccd5ae", "#a2d2ff"],
               autopct='%1.1f%%', startangle=90)
    ax.set_title(f"{PAGE_LABELS[p]}\n(total rejected: {total_rej})")
plt.tight_layout()
out3 = "outputs/alto_rejection_breakdown_v5.png"
plt.savefig(out3, dpi=150)
print(f"Graph 3 (ALTO rejection) → {out3}")
plt.close()


# ── Graph 4: TPP-Alto vs ALTO-OPM head-to-head ───────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
fig4.suptitle("Graph 4 · TPP-ALTO vs ALTO-OPM: Head-to-Head Comparison",
              fontweight='bold')
metrics_hth = [
    ("avg_latency",          "Average Latency (ns)",           ".1f"),
    ("migration_efficiency", "Scan-driven Efficiency",         ".3f"),
    ("dram_hit_ratio",       "DRAM Hit Ratio",                 ".3f"),
]
for ax, (mkey, mtitle, mfmt) in zip(axes4, metrics_hth):
    bw  = 0.28
    for mode, off in zip(["tpp_alto", "alto_opm"], [-bw/2, bw/2]):
        vals = [results[p][mode][mkey] for p in PAGE_COUNTS]
        bars = ax.bar(x + off, vals, bw, label=LABELS[mode],
                      color=COLORS[mode], edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02, f"{v:{mfmt}}",
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
    ax.set_title(mtitle, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out4 = "outputs/tpp_vs_alto_headtohead_v5.png"
plt.savefig(out4, dpi=150)
print(f"Graph 4 (TPP vs ALTO head-to-head) → {out4}")
plt.close()


# ── Graph 5: TPP-Alto active-page promotion analysis ─────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
fig5.suptitle("Graph 5 · TPP-ALTO: Active-page Filtering & Promotion Quota Analysis",
              fontweight='bold')

# 5a: Promotions vs total migrations (TPP-Alto)
ax5a = axes5[0]
tpp_proms  = [results[p]["tpp_alto"]["promotions"]       for p in PAGE_COUNTS]
tpp_total  = [results[p]["tpp_alto"]["total_migrations"] for p in PAGE_COUNTS]
tpp_dems   = [results[p]["tpp_alto"]["demotions"]        for p in PAGE_COUNTS]
bw = 0.25
bars_p = ax5a.bar(x - bw, tpp_proms, bw, label="Promotions", color=COLORS["tpp_alto"], alpha=0.9)
bars_d = ax5a.bar(x,       tpp_dems,  bw, label="Demotions",  color=COLORS["tpp_alto"], alpha=0.5, hatch="//")
bars_t = ax5a.bar(x + bw,  tpp_total, bw, label="Total mig.",  color="#555555", alpha=0.6)
for bars, vals in [(bars_p, tpp_proms), (bars_d, tpp_dems), (bars_t, tpp_total)]:
    for bar, v in zip(bars, vals):
        ax5a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() * 1.02, str(v),
                  ha='center', va='bottom', fontsize=8)
ax5a.set_xticks(x)
ax5a.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=8)
ax5a.set_title("TPP-ALTO: Promotion / Demotion / Total", fontweight='bold')
ax5a.set_ylabel("# Migrations")
ax5a.legend(fontsize=8)
ax5a.grid(axis='y', alpha=0.3)

# 5b: TPP-Alto efficiency vs all scan-driven algorithms
ax5b = axes5[1]
for mode in ["opm", "opmx", "tpp_alto", "alto_opm"]:
    vals = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    ax5b.plot(PAGE_COUNTS, vals, marker='o', linewidth=2,
              linestyle=ls_map[mode], color=COLORS[mode], label=LABELS[mode])
    for px, vy in zip(PAGE_COUNTS, vals):
        ax5b.annotate(f"{vy:.3f}", (px, vy),
                      textcoords="offset points", xytext=(0, 8),
                      ha='center', fontsize=7.5)
ax5b.axhline(0.5, color='red', linestyle='--', linewidth=1, label='0.5 baseline')
ax5b.set_xticks(PAGE_COUNTS)
ax5b.set_xticklabels(["100\n(under)", "200\n(at cap)", "300\n(over)"])
ax5b.set_xlabel("Total Pages")
ax5b.set_ylabel("Scan-driven Efficiency")
ax5b.set_title("Scan-driven Efficiency: All 4 Scan Algorithms", fontweight='bold')
ax5b.legend(fontsize=8)
ax5b.grid(alpha=0.3)
ax5b.set_ylim(0, 1.1)

plt.tight_layout()
out5 = "outputs/tpp_alto_analysis_v5.png"
plt.savefig(out5, dpi=150)
print(f"Graph 5 (TPP-ALTO analysis) → {out5}")
plt.close()


# ── Graph 6: Six-algorithm radar / spider chart ───────────────────────────────
metrics_radar = ["avg_latency", "migration_efficiency", "dram_hit_ratio",
                 "total_migrations", "fault_count"]
metric_labels = ["Avg Latency\n(lower=better)", "Mig Efficiency\n(higher=better)",
                 "DRAM Hit Ratio\n(higher=better)", "Total Mig\n(lower=better)",
                 "Fault Count\n(lower=better)"]

fig6 = plt.figure(figsize=(19, 6))
fig6.suptitle("Graph 6 · All Algorithms — Metric Radar (normalised per pressure scenario)",
              fontweight='bold')

for idx, pc in enumerate(PAGE_COUNTS):
    ax = fig6.add_subplot(1, 3, idx + 1, projection='polar')

    raw = {m: {mode: results[pc][mode][m] for mode in MODES} for m in metrics_radar}
    invert = {"avg_latency", "total_migrations", "fault_count"}
    norm = {}
    for m in metrics_radar:
        vals = list(raw[m].values())
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1
        for mode in MODES:
            v = (raw[m][mode] - lo) / span
            norm.setdefault(mode, {})[m] = (1 - v) if m in invert else v

    n_vars = len(metrics_radar)
    angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=6.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=5.5)
    ax.set_ylim(0, 1)

    for mode in MODES:
        values = [norm[mode][m] for m in metrics_radar]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid',
                color=COLORS[mode], label=LABELS[mode])
        ax.fill(angles, values, color=COLORS[mode], alpha=0.08)

    ax.set_title(PAGE_LABELS[pc], fontweight='bold', size=9, pad=14)
    if idx == 2:
        ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=7)

plt.tight_layout()
out6 = "outputs/radar_all_algorithms_v5.png"
plt.savefig(out6, dpi=150, bbox_inches='tight')
print(f"Graph 6 (radar chart) → {out6}")
plt.close()

print("\nAll done. Outputs in ./outputs/")
print("Files generated:")
for f in [out1, out2, out3, out4, out5, out6]:
    print(f"  {f}")
