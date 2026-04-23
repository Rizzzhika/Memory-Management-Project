
from collections import deque
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
HOT_THRESHOLD        = 6
COLD_THRESHOLD       = 2
MIGRATION_THRESHOLD  = 2
MAX_MIGRATIONS_PER_SCAN = 50


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
        self.lap_level         = 0            # 0 (coldest) … 8 (hottest)
        self.prot_none         = False
        self.fault_count       = {}
        self.scan_generation   = 0

    def mark_for_numa_scan(self):
        self.prot_none = True

    def numa_fault(self, cpu_id, current_generation):
        """Called on every re-access while prot_none=True.
        Accumulates faults within the same scan generation.
        prot_none stays True — cleared only on successful migration."""
        if self.scan_generation != current_generation:
            self.fault_count     = {}
            self.scan_generation = current_generation

        self.last_accessed_cpu = cpu_id
        self.accessed_scan     = True
        self.access_frequency += 1
        self.fault_count[cpu_id] = self.fault_count.get(cpu_id, 0) + 1
        return self.fault_count[cpu_id]

    def record_access(self, cpu_id):
        self.last_accessed_cpu = cpu_id
        self.accessed_scan     = True
        self.access_frequency += 1

    def update_history_on_scan(self):
        if self.accessed_scan:
            self.access_history = ((self.access_history << 1) | 1) & 0xFF
        else:
            self.access_history = (self.access_history << 1) & 0xFF
        self.accessed_scan    = False
        self.access_frequency = 0
        old_level  = self.lap_level
        self.lap_level = bin(self.access_history).count('1')
        if self.current_node and old_level != self.lap_level:
            self.current_node.update(self, old_level)

    def __repr__(self):
        return (f"Page(id={self.page_id}, "
                f"node={self.current_node.name if self.current_node else None}, "
                f"lap={self.lap_level})")


# ══════════════════════════════════════════════════════════════════════════════
#  MemoryNode
# ══════════════════════════════════════════════════════════════════════════════
class MemoryNode:
    def __init__(self, node_id, name, tier, capacity, cpu_socket):
        self.node_id    = node_id
        self.name       = name
        self.tier       = tier        # "upper" | "lower"
        self.capacity   = capacity
        self.cpu_socket = cpu_socket  # -1 → CPU-less
        self.pages      = deque()
        self.lap_lists  = {i: [] for i in range(9)}

    def get_latency(self, cpu_id):
        is_local = (self.cpu_socket == cpu_id)
        if self.tier.upper() == 'UPPER':
            return 100 if is_local else 150
        else:
            return 300 if is_local else 350

    def is_full(self):
        return len(self.pages) >= self.capacity

    def free_slots(self):
        return self.capacity - len(self.pages)

    def add_page(self, page):
        self.pages.append(page)
        page.current_node = self
        self.lap_lists[page.lap_level].append(page)

    def remove_page(self, page):
        try:
            self.pages.remove(page)
        except ValueError:
            pass
        if page in self.lap_lists[page.lap_level]:
            self.lap_lists[page.lap_level].remove(page)
        page.current_node = None

    def get_least_accessed_page(self):
        for level in range(9):
            if self.lap_lists[level]:
                return self.lap_lists[level][0]
        return None

    def utilization(self):
        return len(self.pages) / self.capacity if self.capacity else 0

    def update(self, page, old_level):
        if page in self.lap_lists[old_level]:
            self.lap_lists[old_level].remove(page)
        self.lap_lists[page.lap_level].append(page)

    def __repr__(self):
        return (f"MemoryNode({self.name}, "
                f"{len(self.pages)}/{self.capacity}, tier={self.tier})")


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
        latency = page.current_node.get_latency(self.cpu_id)
        self.total_latency  += latency
        self.total_accesses += 1
        return result

    def avg_latency(self):
        return self.total_latency / self.total_accesses if self.total_accesses else 0

    def __repr__(self):
        return f"CPU(id={self.cpu_id}, accesses={self.total_accesses})"


# ══════════════════════════════════════════════════════════════════════════════
#  AutoNUMAScheduler
# ══════════════════════════════════════════════════════════════════════════════
class AutoNUMAScheduler:
    SCAN_PERIOD_MIN      = 50
    SCAN_PERIOD_MAX      = 200
    FAULT_THRESHOLD      = 3      # faults per generation needed to trigger migration
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
        # Adaptive period: shrink when active, grow when quiet
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

    def __repr__(self):
        return (f"Scheduler(gen={self.current_generation}, "
                f"period={self.scan_period}, scans={self.total_scans})")


# ══════════════════════════════════════════════════════════════════════════════
#  MigrationStats  — tracks promotions vs demotions separately
# ══════════════════════════════════════════════════════════════════════════════
class MigrationStats:
    """
    Tracks every individual move with directional semantics:
      promotion  : lower-tier  → upper-tier   (beneficial, latency-reducing)
      demotion   : upper-tier  → lower-tier   (eviction, makes room)
      lateral    : same-tier move             (locality improvement)

    For fault-driven modes we also track:
      fault_eligible  : faults that passed threshold and were non-local
      fault_migrated  : of those, how many actually moved
    """
    def __init__(self):
        self.promotions     = 0
        self.demotions      = 0
        self.lateral        = 0
        self.fault_eligible = 0
        self.fault_migrated = 0

    def record_move(self, src_node, dst_node):
        src_upper = src_node.tier.upper() == 'UPPER'
        dst_upper = dst_node.tier.upper() == 'UPPER'
        if not src_upper and dst_upper:
            self.promotions += 1
        elif src_upper and not dst_upper:
            self.demotions  += 1
        else:
            self.lateral    += 1

    def total(self):
        return self.promotions + self.demotions + self.lateral

    # ── Efficiency definitions ──────────────────────────────────────────
    def fault_driven_efficiency(self):
        """Fraction of eligible faults that resulted in a migration.
        Valid for: baseline, cpm."""
        if self.fault_eligible == 0:
            return 0.0
        return self.fault_migrated / self.fault_eligible

    def scan_driven_efficiency(self):
        """Fraction of scan-moves that were promotions (upward = beneficial).
        Valid for: opm, opmx.
        Range: 0.0 (all demotions) → 1.0 (all promotions).
        0.5 means equal up/down churn."""
        total_scan = self.promotions + self.demotions
        if total_scan == 0:
            return 0.0
        return self.promotions / total_scan


# ══════════════════════════════════════════════════════════════════════════════
#  MemorySystem
# ══════════════════════════════════════════════════════════════════════════════
class MemorySystem:
    def __init__(self, stats: MigrationStats):
        self.stats = stats

        # Upper tier: 2 DRAM nodes × 20 pages each = 40 pages
        self.dram_0  = MemoryNode(0, "DRAM_node0",  "upper", capacity=20, cpu_socket=0)
        self.dram_1  = MemoryNode(1, "DRAM_node1",  "upper", capacity=20, cpu_socket=1)
        # Lower tier: 2 DCPMM nodes × 80 pages each = 160 pages
        self.dcpmm_2 = MemoryNode(2, "DCPMM_node2", "lower", capacity=80, cpu_socket=0)
        self.dcpmm_3 = MemoryNode(3, "DCPMM_node3", "lower", capacity=80, cpu_socket=1)

        self.cpu0 = CPU(0)
        self.cpu1 = CPU(1)

        self.page_table = {}
        self.all_nodes  = [self.dram_0, self.dram_1, self.dcpmm_2, self.dcpmm_3]
        self.all_cpu    = [self.cpu0, self.cpu1]

    def initialize_pages(self, num_pages):
        """Cold start: all pages in DCPMM"""
        for i in range(num_pages):
            page   = Page(i)
            target = self.dcpmm_2 if i % 2 == 0 else self.dcpmm_3
            if target.is_full():
                alt = self.dcpmm_3 if i % 2 == 0 else self.dcpmm_2
                target = alt if not alt.is_full() else None
            if target:
                target.add_page(page)
            self.page_table[i] = page

    def SystemScan(self):
        for node in self.all_nodes:
            for page in list(node.pages):          # copy → safe iteration
                page.update_history_on_scan()
                page.mark_for_numa_scan()

    # ── Placement helpers ──────────────────────────────────────────────
    def get_preferred_node(self, cpu_id):
        return self.dram_0 if cpu_id == 0 else self.dram_1

    def get_cpm_fallback_order(self, cpu_id):
        """Sort nodes: upper-tier first, then local-before-remote."""
        def rank(node):
            tier_score  = 0 if node.tier.upper() == 'UPPER' else 1
            local_score = 0 if node.cpu_socket == cpu_id    else 1
            return (tier_score, local_score)
        preferred = [self.dram_0, self.dram_1]
        fallback  = [self.dcpmm_2, self.dcpmm_3]

        # Prefer local DRAM first
        if cpu_id == 0:
            preferred = [self.dram_0, self.dram_1]
        else:
            preferred = [self.dram_1, self.dram_0]

        return preferred + fallback

    # ── Core move primitive ────────────────────────────────────────────
    def move_logic(self, page, destination):
        if page.current_node is destination:
            return
        src = page.current_node
        if src:
            src.remove_page(page)
        destination.add_page(page)
        page.prot_none = False          # re-enable normal access after move
        if src:
            self.stats.record_move(src, destination)

    def get_victim(self, node1, node2):
        v0 = node1.get_least_accessed_page()
        v1 = node2.get_least_accessed_page()
        if v0 is None: return v1
        if v1 is None: return v0
        return v0 if v0.lap_level <= v1.lap_level else v1

    # ── Fault-driven migration (baseline) ─────────────────────────────
    def migrate_baseline(self, page, cpu_id, scheduler):
        target = self.get_preferred_node(cpu_id)
        if page.current_node is target:
            return "already_local"
        if scheduler.is_throttled():
            return "throttled"
        if target.is_full():
            return "failed_full"
        self.move_logic(page, target)
        scheduler.record_migration()
        self.stats.fault_migrated += 1
        return "migrated"

    # ── Fault-driven migration (CPM) ──────────────────────────────────
    def migrate_cpm(self, page, cpu_id, scheduler):
        if scheduler.is_throttled():
            return "throttled"
        for node in self.get_cpm_fallback_order(cpu_id):
            if page.current_node is node:
                return "already_at_best"
            if not node.is_full():
                self.move_logic(page, node)
                scheduler.record_migration()
                self.stats.fault_migrated += 1
                return f"migrated_to_{node.name}"
        return "failed_all_full"

    # ── Fault dispatcher ──────────────────────────────────────────────
    def handle_numa_fault(self, page, cpu_id, fault_count, scheduler, mode):
        preferred = self.get_preferred_node(cpu_id)
        if page.current_node is preferred:
            return "local_already"
        if fault_count < scheduler.FAULT_THRESHOLD:
            return "insufficient_faults"

        # Page is eligible for migration
        self.stats.fault_eligible += 1

        if mode == "baseline":
            return self.migrate_baseline(page, cpu_id, scheduler)
        elif mode == "cpm":
            return self.migrate_cpm(page, cpu_id, scheduler)
        else:
            # opm / opmx are scan-driven; faults just record demand signal
            return "opm_demand_noted"

    # ── Scan-driven: OPM ─────────────────────────────────────────────
    def opm(self):
        # Step 1 – Promote hot pages from DCPMM → DRAM
        for node in [self.dcpmm_2, self.dcpmm_3]:
            for level in range(HOT_THRESHOLD, 9):
                for page in list(node.lap_lists[level]):
                    cpu_id = page.last_accessed_cpu
                    if cpu_id is None:
                        continue
                    target_dram = self.dram_0 if cpu_id == 0 else self.dram_1
                    if page.current_node is target_dram:
                        continue
                    if not target_dram.is_full():
                        self.move_logic(page, target_dram)
                    else:
                        victim = self.get_victim(self.dram_0, self.dram_1)
                        if victim and page.lap_level > victim.lap_level:
                            v_cpu  = victim.last_accessed_cpu
                            v_dcpmm = self.dcpmm_2 if (v_cpu == 0 or v_cpu is None) else self.dcpmm_3
                            self.move_logic(victim, v_dcpmm)   # demotion
                            self.move_logic(page, target_dram) # promotion

        # Step 2 – Demote cold pages from DRAM → DCPMM
        for node in [self.dram_0, self.dram_1]:
            for level in range(0, COLD_THRESHOLD + 1):
                for page in list(node.lap_lists[level]):
                    cpu_id  = page.last_accessed_cpu
                    t_dcpmm = self.dcpmm_2 if (cpu_id == 0 or cpu_id is None) else self.dcpmm_3
                    self.move_logic(page, t_dcpmm)

    # ── Scan-driven: OPMX ────────────────────────────────────────────
    def opmx(self):
        done = 0

        # Step 1 – Throttled promotion of hot pages
        for node in [self.dcpmm_2, self.dcpmm_3]:
            for level in range(HOT_THRESHOLD, 9):
                for page in list(node.lap_lists[level]):
                    if done >= MAX_MIGRATIONS_PER_SCAN:
                        return
                    cpu_id = page.last_accessed_cpu
                    if cpu_id is None:
                        continue
                    target_dram = self.dram_0 if cpu_id == 0 else self.dram_1
                    if page.current_node is target_dram:
                        continue
                    # Stability guard: accessed in last 2 consecutive windows
                    if (page.access_history & 0b1) == 0:
                        continue
                    if not target_dram.is_full():
                        if page.lap_level < (HOT_THRESHOLD + 1):
                            continue
                        self.move_logic(page, target_dram)
                        done += 1
                    else:
                        victim = self.get_victim(self.dram_0, self.dram_1)
                        if victim is None:
                            continue
                        if (page.lap_level - victim.lap_level) < MIGRATION_THRESHOLD:
                            continue
                        v_cpu   = victim.last_accessed_cpu
                        v_dcpmm = self.dcpmm_2 if (v_cpu == 0 or v_cpu is None) else self.dcpmm_3
                        self.move_logic(victim, v_dcpmm)   # demotion
                        self.move_logic(page, target_dram) # promotion
                        done += 2

        # Step 2 – Throttled cold demotion
        for node in [self.dram_0, self.dram_1]:
            for level in range(0, COLD_THRESHOLD + 1):
                for page in list(node.lap_lists[level]):
                    if done >= MAX_MIGRATIONS_PER_SCAN:
                        return
                    cpu_id  = page.last_accessed_cpu
                    t_dcpmm = self.dcpmm_2 if (cpu_id == 0 or cpu_id is None) else self.dcpmm_3
                    if (page.access_history & 0b11) == 0:
                        self.move_logic(page, t_dcpmm)
                        done += 1


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation runner
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_STEPS = 20_000
MODES       = ["baseline", "cpm", "opm", "opmx"]
PAGE_COUNTS = [100, 200, 300]   # under / at / over capacity (total cap = 200)
TOTAL_CAP   = 200               # 2×20 DRAM + 2×80 DCPMM

# Scan-driven modes: efficiency = promotions / (promotions + demotions)
# Fault-driven modes: efficiency = fault_migrated / fault_eligible
SCAN_DRIVEN  = {"opm", "opmx"}
FAULT_DRIVEN = {"baseline", "cpm"}

results = {}

for total_pages in PAGE_COUNTS:
    results[total_pages] = {}
    half = total_pages // 2

    for mode in MODES:
        st   = MigrationStats()
        mem  = MemorySystem(st)
        sch  = AutoNUMAScheduler()
        mem.initialize_pages(total_pages)

        outcomes    = {}
        latency_log = []
        dram_hits   = 0

        random.seed(42)

        for step in range(TOTAL_STEPS):
            number     = random.randint(0, 99)
            cpu_sel    = random.choice(mem.all_cpu)

            # 80 % local / 20 % remote access pattern
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

            # Track DRAM hits after (possible) fault handling
            if page.current_node and page.current_node.tier.upper() == 'UPPER':
                dram_hits += 1

            if result[0] == "fault":
                outcome = mem.handle_numa_fault(
                    page, cpu_sel.cpu_id, result[1], sch, mode=mode)
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

                # Trigger OPM/OPMX scan logic at the end of every period
                if mode in SCAN_DRIVEN and sch.access_count == 0:
                    if mode == "opm":
                        mem.opm()
                    else:
                        mem.opmx()
            else:
                outcomes["normal"] = outcomes.get("normal", 0) + 1

            sch.tick(mem)

            if (step + 1) % 500 == 0:
                avg_lat = sum(c.avg_latency() for c in mem.all_cpu) / len(mem.all_cpu)
                latency_log.append(avg_lat)

        # ── Compute metrics ────────────────────────────────────────────
        total_accesses = sum(c.total_accesses for c in mem.all_cpu)
        avg_lat_final  = sum(c.avg_latency()  for c in mem.all_cpu) / len(mem.all_cpu)
        dram_hit_ratio = dram_hits / total_accesses if total_accesses else 0

        # Conceptually correct efficiency
        if mode in FAULT_DRIVEN:
            mig_eff = st.fault_driven_efficiency()
            eff_label = "fault_migrated/fault_eligible"
        else:
            mig_eff = st.scan_driven_efficiency()
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
        }

        print(f"[pages={total_pages:3d} | mode={mode:8s}] "
              f"mig={st.total():5d} (↑{st.promotions} ↓{st.demotions} ↔{st.lateral})  "
              f"avg_lat={avg_lat_final:.1f}ns  "
              f"dram={dram_hit_ratio:.3f}  "
              f"eff={mig_eff:.3f} [{eff_label}]")

# Save JSON
import os
os.makedirs("outputs", exist_ok=True)

with open("outputs/sim_results_v2.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nJSON saved.")


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "baseline": "#e15759",
    "cpm"     : "#f28e2b",
    "opm"     : "#4e79a7",
    "opmx"    : "#59a14f",
}
PAGE_LABELS = {
    100: "Under-pressure\n(100 pages)",
    200: "At-capacity\n(200 pages)",
    300: "Over-pressure\n(300 pages)",
}

x       = np.arange(len(PAGE_COUNTS))
width   = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width


def grouped_bar(ax, metric_key, title, ylabel, log_scale=False, fmt=".1f"):
    for mode, off in zip(MODES, offsets):
        vals = [results[p][mode][metric_key] for p in PAGE_COUNTS]
        bars = ax.bar(x + off, vals, width, label=mode.upper(),
                      color=COLORS[mode], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            label = f"{v:{fmt}}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    label, ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=9)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    if log_scale:
        ax.set_yscale('symlog', linthresh=1)


fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle(
    "AutoTiering Simulation – Metrics Comparison\n"
    "Fault-driven efficiency = migrated/eligible faults  |  "
    "Scan-driven efficiency = promotions/(promotions+demotions)",
    fontsize=12, fontweight='bold')

# 1 · Avg Latency
grouped_bar(axes[0][0], "avg_latency", "1 · Average Latency (ns)", "Latency (ns)")

# 2 · Total Migrations (with promotion/demotion breakdown)
ax2 = axes[0][1]
bar_w = width * 0.45
for mode, off in zip(MODES, offsets):
    proms = [results[p][mode]["promotions"] for p in PAGE_COUNTS]
    dems  = [results[p][mode]["demotions"]  for p in PAGE_COUNTS]
    lats  = [results[p][mode]["lateral"]    for p in PAGE_COUNTS]
    ax2.bar(x + off - bar_w/2, proms, bar_w, color=COLORS[mode], alpha=0.9,
            label=f"{mode.upper()} ↑prom" if mode == "baseline" else "_")
    ax2.bar(x + off + bar_w/2, dems,  bar_w, color=COLORS[mode], alpha=0.45,
            hatch='//', label=f"{mode.upper()} ↓dem" if mode == "baseline" else "_")
ax2.set_xticks(x)
ax2.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=9)
ax2.set_title("2 · Total Migrations\n(solid=promotions, hatched=demotions)", fontweight='bold', pad=8)
ax2.set_ylabel("# Migrations")
# custom legend
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=COLORS[m], label=m.upper()) for m in MODES]
legend_els += [Patch(facecolor='grey', label='↑ Promotions'),
               Patch(facecolor='grey', hatch='//', alpha=0.4, label='↓ Demotions')]
ax2.legend(handles=legend_els, fontsize=7, ncol=2)
ax2.grid(axis='y', alpha=0.3)

# 3 · Fault Count
grouped_bar(axes[0][2], "fault_count",
            "3 · Eligible NUMA Faults\n(above threshold, non-local)", "# Faults", fmt=".0f")

# 4A · Fault-driven Efficiency
ax4a = axes[1][0]

FAULT_MODES = ["baseline", "cpm"]

for mode, off in zip(FAULT_MODES, [-width/2, width/2]):
    vals = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    bars = ax4a.bar(x + off, vals, width, label=mode.upper(),
                    color=COLORS[mode], edgecolor='white')

    for bar, v in zip(bars, vals):
        ax4a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.005,
                  f"{v:.3f}", ha='center', fontsize=8)

ax4a.set_xticks(x)
ax4a.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=9)
ax4a.set_title("4A · Fault-driven Efficiency\n(migrated / eligible faults)",
               fontweight='bold')
ax4a.set_ylabel("Efficiency (0–1)")
ax4a.legend()
ax4a.grid(axis='y', alpha=0.3)

# 4B · Scan-driven Efficiency
ax4b = axes[1][1]

SCAN_MODES = ["opm", "opmx"]

for mode, off in zip(SCAN_MODES, [-width/2, width/2]):
    vals = [results[p][mode]["migration_efficiency"] for p in PAGE_COUNTS]
    bars = ax4b.bar(x + off, vals, width, label=mode.upper(),
                    color=COLORS[mode], edgecolor='white')

    for bar, v in zip(bars, vals):
        ax4b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.01,
                  f"{v:.2f}", ha='center', fontsize=8)

ax4b.set_xticks(x)
ax4b.set_xticklabels([PAGE_LABELS[p] for p in PAGE_COUNTS], fontsize=9)
ax4b.set_ylim(0, 1.05)

ax4b.axhline(0.5, color='red', linestyle='--', linewidth=1,
             label='0.5 = equal promo/demotion')

ax4b.set_title("4B · Scan-driven Efficiency\n(promotions / (promo + demo))",
               fontweight='bold')
ax4b.set_ylabel("Efficiency (0–1)")
ax4b.legend()
ax4b.grid(axis='y', alpha=0.3)

# 5 · Latency over time
ax5 = axes[1][2]
ls_map = {"baseline": "-", "cpm": "--", "opm": "-.", "opmx": ":"}
for pi, pages in enumerate(PAGE_COUNTS):
    alpha = 0.5 + 0.25 * pi
    for mode in MODES:
        log    = results[pages][mode]["latency_log"]
        steps  = [(i + 1) * 500 for i in range(len(log))]
        ax5.plot(steps, log,
                 linestyle=ls_map[mode],
                 color=COLORS[mode],
                 alpha=alpha,
                 linewidth=1.3,
                 label=f"{pages}p {mode.upper()}" if pi == 0 else "_")
mode_handles = [plt.Line2D([0],[0], color=COLORS[m], linewidth=2,
                             linestyle=ls_map[m], label=m.upper()) for m in MODES]
pg_handles   = [plt.Line2D([0],[0], color='grey', linewidth=1+pi,
                             alpha=0.5+0.25*pi, label=f"{p} pages")
                for pi, p in enumerate(PAGE_COUNTS)]
ax5.legend(handles=mode_handles + pg_handles, fontsize=7, ncol=2)
ax5.set_xlabel("Simulation Step")
ax5.set_ylabel("Avg Latency (ns)")
ax5.set_title("5 · Latency vs Memory Pressure\n(over simulation time)", fontweight='bold', pad=8)
ax5.grid(alpha=0.3)

# 6 · DRAM Hit Ratio
grouped_bar(axes[1][3], "dram_hit_ratio",
            "6 · DRAM Hit Ratio\n(fraction of accesses served from upper tier)",
            "Ratio (0–1)", fmt=".3f")

plt.tight_layout()
out1 = "outputs/autoTiering_metrics_v2.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Main plot → {out1}")
plt.close()

# ── Bonus: promotion vs demotion ratio line chart ─────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Scan-driven Move Breakdown: Promotions vs Demotions",
              fontweight='bold')

for ax, metric, label in zip(axes2,
                              ["promotions", "demotions"],
                              ["↑ Promotions (DCPMM→DRAM)", "↓ Demotions (DRAM→DCPMM)"]):
    for mode in MODES:
        vals = [results[p][mode][metric] for p in PAGE_COUNTS]
        ax.plot(PAGE_COUNTS, vals, marker='o', linewidth=2,
                color=COLORS[mode], label=mode.upper())
        for px, vy in zip(PAGE_COUNTS, vals):
            ax.annotate(str(vy), (px, vy),
                        textcoords="offset points", xytext=(0, 7),
                        ha='center', fontsize=8)
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
out2 = "outputs/promo_vs_demo_v2.png"
plt.savefig(out2, dpi=150)
print(f"Promo/demo plot → {out2}")
plt.close()
