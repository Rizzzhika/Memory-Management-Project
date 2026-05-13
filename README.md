# ML-Augmented Multi-Tier Memory Management for NUMA Systems

A simulation-based Operating Systems project implementing and comparing multiple page-management and memory-tiering algorithms for heterogeneous NUMA architectures using DRAM and Persistent Memory (DCPMM). The project extends traditional AutoNUMA-style migration with scan-driven policies and machine-learning-assisted page placement.

---

# Project Overview

Modern servers increasingly use **multi-tier memory systems** where fast DRAM coexists with slower but larger persistent memory such as Intel Optane DCPMM. Efficient page placement across these tiers is critical for reducing memory-access latency and improving system performance.

This project simulates and evaluates:

- Fault-driven memory migration policies
- Scan-driven page promotion/demotion techniques
- ML-augmented page hotness prediction
- NUMA-aware memory tiering
- TPP-ALTO style Linux kernel behavior

The implementation is inspired by research on AutoTiering, Kleio, and multi-tier memory systems.

---

# Repository Structure

```text
├── final.py
├── final_submission.py
├── final_submission_ml.py
└── README.md
```

## 1. `final.py`

Contains the complete implementation written for:

- System initialization
- OPM (Opportunistic Page Migration)
- OPMX (Extended OPM)
- Core simulation framework
- Memory topology
- NUMA access handling
- Migration logic

---

## 2. `final_submission.py`

Integrated version containing:

- Baseline Linux AutoNUMA simulation
- CPM (Conservative Page Migration)
- AutoNUMA extensions
- OPM and OPMX integration
- Comparative evaluation framework

---

## 3. `final_submission_ml.py`

Final integrated implementation containing:

- All previous algorithms
- ML-augmented memory management
- Kleio v2-inspired hotness scoring
- ALTO-OPM
- TPP-ALTO simulation
- Adaptive promotion/demotion logic

---

# Algorithms Implemented

## Fault-Driven Policies

### Baseline Linux AutoNUMA

Traditional NUMA balancing approach where page migration is triggered only after NUMA faults occur.

### CPM (Conservative Page Migration)

Extends baseline migration with fallback placement:

```text
Local DRAM → Remote DRAM → DCPMM
```

Improves placement decisions under memory pressure.

---

## Scan-Driven Policies

### OPM (Opportunistic Page Migration)

Periodically scans memory pages and:

- Promotes hot pages to DRAM
- Demotes cold pages to DCPMM
- Uses lap-level based hotness estimation

### OPMX

Enhanced version of OPM with:

- Migration throttling
- Stability guards
- Reduced migration churn
- Per-scan migration caps

---

## ML-Augmented Policies

### ALTO-OPM

Machine-learning-assisted extension of OPM using a lightweight hotness prediction model inspired by the **Kleio** framework.

Features used for scoring:

- Lap-level normalization
- Access recency
- Access frequency
- Locality
- Hot streak history
- Access consistency

The model dynamically filters promotion and demotion candidates to improve migration efficiency.

### TPP-ALTO

Simulation of Linux kernel:

```text
tpp.patch + tpp-alto.patch
```

Includes:

- Active-page filtering
- Quota-gated promotion
- PG_demoted tracking
- Top-tier optimization
- NUMA-aware migration behavior

---

# Simulated Memory Topology

The simulator models a dual-socket NUMA system:

| Node | Memory Type | Capacity | Socket |
|------|-------------|----------|---------|
| DRAM_node0 | Upper Tier DRAM | 20 pages | CPU 0 |
| DRAM_node1 | Upper Tier DRAM | 20 pages | CPU 1 |
| DCPMM_node2 | Lower Tier PMEM | 80 pages | CPU 0 |
| DCPMM_node3 | Lower Tier PMEM | 80 pages | CPU 1 |

Latency characteristics:

- Local DRAM: ~100 ns
- Remote DRAM: ~150 ns
- Local DCPMM: ~300 ns
- Remote DCPMM: ~350 ns

---

# Workload Model

The simulator executes synthetic workloads with:

- 20,000 simulation steps
- Pareto-like hot/cold access distribution
- 80/20 hot-page locality
- NUMA-aware CPU access preference

Three pressure scenarios are evaluated:

| Scenario | Pages |
|----------|--------|
| Under Pressure | 100 |
| At Capacity | 200 |
| Over Pressure | 300 |

---

# Metrics Evaluated

The project evaluates:

- Average memory-access latency
- DRAM hit ratio
- Total page migrations
- Promotion/demotion counts
- Fault-driven efficiency
- Scan-driven efficiency
- Migration stability
- ML rejection statistics

---

# Technologies Used

- Python
- NumPy
- Matplotlib
- Object-Oriented Simulation Design

---

# Research Papers & Credits

This project was inspired by the following research works:

## 1. AutoTiering / OPM / OPMX

**Exploring the Design Space of Page Management for Multi-Tiered Memory Systems**  
Jonghyeon Kim, Wonkyo Choe, Jeongseob Ahn  
USENIX ATC 2021

Introduced:
- OPM
- Multi-tier page promotion/demotion
- AutoTiering concepts
- DRAM ↔ DCPMM memory management

Paper:
https://www.usenix.org/conference/atc21/presentation/kim-jonghyeon

---

## 2. TPP-ALTO

**TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory**  
A. Maruf et al.  
ASPLOS 2023

Inspired:
- TPP behavior
- Active-page filtering
- Quota-gated promotion
- Linux tiered memory patch behavior

---

## 3. Kleio ML Framework

**Kleio: A Hybrid Memory Page Scheduler with Machine Intelligence**  
J. Liu, H. Hadian, H. Xu, H. Li  
ACM SoCC 2023

Inspired:
- ML-based page scheduling
- AOL feature vectors
- Adaptive hotness scoring
- ML gating for migration decisions

Paper:
https://dl.acm.org/doi/10.1145/3698038.3698546

---

# Team Contributions

## [Rizzzhika](https://github.com/Rizzzhika)

Implemented:

- System initialization
- OPM
- OPMX
- Core simulation infrastructure
- Memory management framework
- Migration handling logic

## [kashish16official](https://github.com/kashish16official)

Implemented:

- Baseline Linux AutoNUMA
- CPM
- AutoNUMA extensions
- Comparative integration

## [anshikaui18-](https://github.com/anshika18-ui)

Implemented:

- ML integration
- Kleio-inspired hotness model
- ALTO-OPM enhancements
- ML-based migration gating

---

# Key Learning Outcomes

- NUMA-aware memory management
- Operating system page migration
- Multi-tier memory architectures
- Persistent memory systems
- Scan-driven vs fault-driven policies
- Machine learning in systems research
- Linux memory-management concepts
- Simulation-based systems evaluation

---

# How to Run

```bash
python final.py
```

or

```bash
python final_submission.py
```

or

```bash
python final_submission_ml.py
```

---

# Output

The simulator generates:

- Comparative performance metrics
- Migration statistics
- DRAM utilization analysis
- Efficiency graphs
- Algorithm comparison plots

---

# Academic Context

Course: **CSD204 – Operating Systems**  
Department of Computer Science and Engineering  
Academic Year: 2025–26

---

# Acknowledgement

We thank the authors and open-source contributors behind:

- Linux AutoNUMA
- TPP / TPP-ALTO kernel patches
- Kleio ML framework
- AutoTiering research

Their work provided the conceptual foundation for this academic simulation project.
