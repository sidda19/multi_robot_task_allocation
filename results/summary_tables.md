# Experimental Evaluation Summary

## Multi-Robot Task Allocation Under Limited Communication (MRTA-LC)

**Course Project — Group 45**

Raw observations: **380 runs** across **10 seeds**. All displayed values are mean ± sample standard deviation unless noted.

### Metric definitions

- **End-to-end total cost:** physical execution cost + allocation-stage conflict penalty. This is the `total_cost` column.
- **Physical execution cost:** travel distance + penalties for unassigned tasks, missed deadlines, and conflicts observed during physical execution.
- **Static assignment objective:** estimated assignment cost + unassigned-task penalty. It excludes conflict overhead and is the only objective compared with Exact ILP.
- **ILP gap (static):** static assignment objective relative to the assignment-model Exact ILP. Defined only for the static allocators; it is an optimistic reference because it ignores that a robot serves its tasks one after another.
- **Exact physical reference (sequence-aware ILP):** the exact optimum of the simulator's physical cost (travel + penalty for unassigned tasks); replaying its plan gives zero missed deadlines.
- **Gap to exact physical reference:** (end-to-end cost - reference) / reference, defined for every method that is physically executed, including the online policies.
- **Runtime:** static methods include allocator + physical replay; MARL includes the online policy and physical episode. Exact ILP reports solver time only and is not plotted.

### Table 1. Results by method and communication level

| Method | Comm | End-to-end cost | Gap to exact physical ref. | Static objective | Static ILP gap | Assignment | On-time completion | Conflicts | Messages | Runtime ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact ILP (assignment) | N/A (Offline) | N/A | N/A | 194.01 ± 16.38 | 0.00 ± 0.00% | 90.00 ± 0.00% | N/A | 0.00 ± 0.00 | 0.00 ± 0.00 | 11.99 ± 2.70 |
| Exact ILP (sequence-aware) | N/A (Offline) | 159.50 ± 41.45 | 0.00 ± 0.00% | N/A | N/A | 83.00 ± 9.49% | 83.00 ± 9.49% | 0.00 ± 0.00 | 0.00 ± 0.00 | 17.43 ± 4.70 |
| Centralized Greedy | Global (Full Info) | 272.60 ± 30.15 | 76.81 ± 30.76% | 302.64 ± 30.77 | 56.27 ± 13.15% | 60.00 ± 6.67% | 60.00 ± 6.67% | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.73 ± 0.62 |
| Distributed Greedy | L0_None | 287.00 ± 45.53 | 88.96 ± 48.48% | 290.62 ± 31.92 | 50.71 ± 20.85% | 57.00 ± 10.59% | 57.00 ± 10.59% | 2.20 ± 1.03 | 0.00 ± 0.00 | 2.89 ± 0.89 |
| Distributed Greedy | L1_Low | 225.30 ± 29.15 | 45.58 ± 23.75% | 252.53 ± 23.78 | 30.33 ± 8.88% | 70.00 ± 6.67% | 70.00 ± 6.67% | 1.00 ± 0.82 | 8.50 ± 1.27 | 2.90 ± 0.52 |
| Distributed Greedy | L2_Medium | 225.30 ± 29.15 | 45.58 ± 23.75% | 252.53 ± 23.78 | 30.33 ± 8.88% | 70.00 ± 6.67% | 70.00 ± 6.67% | 1.00 ± 0.82 | 8.50 ± 1.27 | 3.02 ± 0.80 |
| Distributed Greedy | L3_High | 227.90 ± 28.29 | 47.39 ± 23.90% | 252.78 ± 23.71 | 30.46 ± 8.86% | 70.00 ± 6.67% | 70.00 ± 6.67% | 1.20 ± 1.03 | 8.20 ± 1.32 | 3.20 ± 1.04 |
| Distributed Greedy | L4_Full | 227.90 ± 28.29 | 47.39 ± 23.90% | 252.78 ± 23.71 | 30.46 ± 8.86% | 70.00 ± 6.67% | 70.00 ± 6.67% | 1.20 ± 1.03 | 8.20 ± 1.32 | 2.81 ± 0.57 |
| SSI Auction | L0_None | 306.60 ± 28.93 | 99.24 ± 34.47% | 302.64 ± 30.77 | 56.27 ± 13.15% | 60.00 ± 6.67% | 60.00 ± 6.67% | 3.40 ± 1.07 | 0.00 ± 0.00 | 2.56 ± 0.41 |
| SSI Auction | L1_Low | 322.00 ± 27.20 | 112.05 ± 48.78% | 344.13 ± 24.48 | 77.97 ± 12.65% | 47.00 ± 6.75% | 47.00 ± 6.75% | 0.00 ± 0.00 | 7.80 ± 0.92 | 2.63 ± 0.75 |
| SSI Auction | L2_Medium | 272.60 ± 30.15 | 76.81 ± 30.76% | 302.64 ± 30.77 | 56.27 ± 13.15% | 60.00 ± 6.67% | 60.00 ± 6.67% | 0.00 ± 0.00 | 9.40 ± 1.35 | 2.95 ± 0.77 |
| SSI Auction | L3_High | 272.60 ± 30.15 | 76.81 ± 30.76% | 302.64 ± 30.77 | 56.27 ± 13.15% | 60.00 ± 6.67% | 60.00 ± 6.67% | 0.00 ± 0.00 | 9.40 ± 1.35 | 2.67 ± 0.54 |
| SSI Auction | L4_Full | 272.60 ± 30.15 | 76.81 ± 30.76% | 302.64 ± 30.77 | 56.27 ± 13.15% | 60.00 ± 6.67% | 60.00 ± 6.67% | 0.00 ± 0.00 | 9.40 ± 1.35 | 2.78 ± 0.52 |
| Cooperative MARL (IDQN) | L0_None | 357.40 ± 77.12 | 130.72 ± 53.28% | N/A | N/A | 50.00 ± 8.16% | 50.00 ± 8.16% | 4.90 ± 5.82 | 0.00 ± 0.00 | 20.66 ± 4.75 |
| Cooperative MARL (IDQN) | L1_Low | 289.30 ± 55.52 | 84.96 ± 32.02% | N/A | N/A | 55.00 ± 11.79% | 55.00 ± 11.79% | 0.00 ± 0.00 | 5.50 ± 1.18 | 18.29 ± 2.81 |
| Cooperative MARL (IDQN) | L2_Medium | 295.70 ± 57.72 | 89.47 ± 36.56% | N/A | N/A | 54.00 ± 11.74% | 54.00 ± 11.74% | 0.30 ± 0.95 | 5.70 ± 1.42 | 18.56 ± 4.09 |
| Cooperative MARL (IDQN) | L3_High | 290.10 ± 55.40 | 85.52 ± 32.28% | N/A | N/A | 55.00 ± 11.79% | 55.00 ± 11.79% | 0.10 ± 0.32 | 5.60 ± 1.26 | 19.00 ± 4.89 |
| Cooperative MARL (IDQN) | L4_Full | 290.10 ± 55.40 | 85.52 ± 32.28% | N/A | N/A | 55.00 ± 11.79% | 55.00 ± 11.79% | 0.10 ± 0.32 | 5.60 ± 1.26 | 18.50 ± 3.52 |
| Cooperative MARL (shared DQfD) | L0_None | 280.70 ± 54.66 | 80.56 ± 34.90% | N/A | N/A | 62.00 ± 7.89% | 62.00 ± 7.89% | 2.00 ± 3.13 | 0.00 ± 0.00 | 23.26 ± 6.09 |
| Cooperative MARL (shared DQfD) | L1_Low | 269.50 ± 39.17 | 73.59 ± 25.57% | N/A | N/A | 62.00 ± 7.89% | 62.00 ± 7.89% | 0.90 ± 1.91 | 6.90 ± 1.66 | 20.59 ± 5.45 |
| Cooperative MARL (shared DQfD) | L2_Medium | 263.20 ± 36.69 | 69.55 ± 24.69% | N/A | N/A | 62.00 ± 7.89% | 62.00 ± 7.89% | 0.30 ± 0.68 | 6.50 ± 1.08 | 19.11 ± 4.74 |
| Cooperative MARL (shared DQfD) | L3_High | 257.40 ± 37.27 | 66.14 ± 27.13% | N/A | N/A | 63.00 ± 8.23% | 63.00 ± 8.23% | 0.10 ± 0.32 | 6.40 ± 0.97 | 18.80 ± 3.75 |
| Cooperative MARL (shared DQfD) | L4_Full | 257.40 ± 37.27 | 66.14 ± 27.13% | N/A | N/A | 63.00 ± 8.23% | 63.00 ± 8.23% | 0.10 ± 0.32 | 6.40 ± 0.97 | 21.26 ± 5.76 |
| Online Distributed Greedy | L0_None | 511.60 ± 128.60 | 243.50 ± 126.15% | N/A | N/A | 58.00 ± 10.33% | 58.00 ± 10.33% | 25.30 ± 10.58 | 0.00 ± 0.00 | 3.65 ± 0.71 |
| Online Distributed Greedy | L1_Low | 283.70 ± 93.02 | 84.75 ± 65.24% | N/A | N/A | 68.00 ± 9.19% | 68.00 ± 9.19% | 6.50 ± 8.55 | 9.60 ± 3.34 | 3.31 ± 1.02 |
| Online Distributed Greedy | L2_Medium | 227.60 ± 38.20 | 47.30 ± 28.95% | N/A | N/A | 70.00 ± 9.43% | 70.00 ± 9.43% | 1.80 ± 2.74 | 8.80 ± 3.26 | 2.92 ± 0.87 |
| Online Distributed Greedy | L3_High | 215.60 ± 38.09 | 39.06 ± 25.53% | N/A | N/A | 70.00 ± 9.43% | 70.00 ± 9.43% | 0.60 ± 0.97 | 7.60 ± 1.65 | 3.07 ± 1.06 |
| Online Distributed Greedy | L4_Full | 215.60 ± 38.09 | 39.06 ± 25.53% | N/A | N/A | 70.00 ± 9.43% | 70.00 ± 9.43% | 0.60 ± 0.97 | 7.60 ± 1.65 | 3.54 ± 1.36 |
| Online Greedy (no-message convention) | L0_None | 338.40 ± 105.86 | 121.47 ± 81.91% | N/A | N/A | 61.00 ± 9.94% | 61.00 ± 9.94% | 7.70 ± 7.63 | 0.00 ± 0.00 | 3.35 ± 0.92 |
| Online Greedy (no-message convention) | L1_Low | 253.80 ± 46.96 | 63.19 ± 28.23% | N/A | N/A | 65.00 ± 8.50% | 65.00 ± 8.50% | 0.80 ± 1.93 | 7.10 ± 1.37 | 2.89 ± 0.88 |
| Online Greedy (no-message convention) | L2_Medium | 247.80 ± 40.14 | 59.67 ± 26.56% | N/A | N/A | 65.00 ± 8.50% | 65.00 ± 8.50% | 0.20 ± 0.63 | 6.70 ± 0.95 | 2.67 ± 0.65 |
| Online Greedy (no-message convention) | L3_High | 247.80 ± 40.14 | 59.67 ± 26.56% | N/A | N/A | 65.00 ± 8.50% | 65.00 ± 8.50% | 0.20 ± 0.63 | 6.70 ± 0.95 | 2.85 ± 0.53 |
| Online Greedy (no-message convention) | L4_Full | 247.80 ± 40.14 | 59.67 ± 26.56% | N/A | N/A | 65.00 ± 8.50% | 65.00 ± 8.50% | 0.20 ± 0.63 | 6.70 ± 0.95 | 2.77 ± 0.61 |
| Random Masked Policy | L0_None | 304.80 ± 48.36 | 96.05 ± 30.67% | N/A | N/A | 60.00 ± 9.43% | 60.00 ± 9.43% | 3.50 ± 2.84 | 0.00 ± 0.00 | 4.57 ± 1.02 |
| Random Masked Policy | L1_Low | 285.80 ± 33.20 | 88.11 ± 46.85% | N/A | N/A | 57.00 ± 6.75% | 57.00 ± 6.75% | 0.50 ± 0.71 | 6.20 ± 1.23 | 3.89 ± 0.60 |
| Random Masked Policy | L2_Medium | 284.50 ± 38.71 | 83.00 ± 24.18% | N/A | N/A | 58.00 ± 7.89% | 58.00 ± 7.89% | 0.50 ± 0.71 | 6.30 ± 1.16 | 4.16 ± 0.77 |
| Random Masked Policy | L3_High | 287.30 ± 37.81 | 86.34 ± 35.30% | N/A | N/A | 57.00 ± 8.23% | 57.00 ± 8.23% | 0.50 ± 0.71 | 6.20 ± 1.32 | 3.86 ± 0.69 |
| Random Masked Policy | L4_Full | 282.60 ± 23.01 | 83.75 ± 30.80% | N/A | N/A | 58.00 ± 4.22% | 58.00 ± 4.22% | 0.30 ± 0.68 | 6.10 ± 0.74 | 4.10 ± 0.78 |

### Table 2. Pre-stated hypotheses checked against the data (descriptive)

Thresholds are the ones stated before the experiments. Paired per-seed tests are produced separately by `scripts/paired_tests.py` (`results/paired_tests.md`).

| Hypothesis | Method | Observation | Verdict by stated criterion |
|---|---|---|---|
| H1: cost falls as communication rises (L0 to L3) | Online Distributed Greedy | end-to-end cost 511.6 -> 283.7 -> 227.6 -> 215.6 | supported |
| H1: cost falls as communication rises (L0 to L3) | Distributed Greedy | end-to-end cost 287.0 -> 225.3 -> 225.3 -> 227.9 | partly supported (L3 below L0 but not monotone) |
| H1: cost falls as communication rises (L0 to L3) | Cooperative MARL (shared DQfD) | end-to-end cost 280.7 -> 269.5 -> 263.2 -> 257.4 | supported |
| H2: conflicts fall by more than 80% from L0 to L2 | Online Distributed Greedy | 25.30 -> 1.80 (92.9% reduction) | supported |
| H2: conflicts fall by more than 80% from L0 to L2 | Distributed Greedy | 2.20 -> 1.00 (54.5% reduction) | not supported |
| H2: conflicts fall by more than 80% from L0 to L2 | Cooperative MARL (shared DQfD) | 2.00 -> 0.30 (85.0% reduction) | supported |
| H3: MARL coordinates under sparse communication (L1_Low) | Cooperative MARL (IDQN) | Cooperative MARL (IDQN): 0.00 conflicts, 55% on time, cost 289; Online Distributed Greedy: 6.50 conflicts, 68% on time, cost 284; Distributed Greedy: 1.00 conflicts, 70% on time, cost 225; Random Masked Policy: 0.50 conflicts, 57% on time, cost 286 | NOT meaningfully (>5%) cheaper than a random masked policy: learning is not demonstrated; not clearly cheaper than online greedy; not clearly cheaper than the static Distributed Greedy plan |
| H3: MARL coordinates under sparse communication (L1_Low) | Cooperative MARL (shared DQfD) | Cooperative MARL (shared DQfD): 0.90 conflicts, 62% on time, cost 270; Online Distributed Greedy: 6.50 conflicts, 68% on time, cost 284; Distributed Greedy: 1.00 conflicts, 70% on time, cost 225; Random Masked Policy: 0.50 conflicts, 57% on time, cost 286 | more than 5% cheaper than a random masked policy (check the paired tests for significance); cheaper than online greedy; not clearly cheaper than the static Distributed Greedy plan |
| H3: MARL coordinates under sparse communication (L2_Medium) | Cooperative MARL (IDQN) | Cooperative MARL (IDQN): 0.30 conflicts, 54% on time, cost 296; Online Distributed Greedy: 1.80 conflicts, 70% on time, cost 228; Distributed Greedy: 1.00 conflicts, 70% on time, cost 225; Random Masked Policy: 0.50 conflicts, 58% on time, cost 284 | NOT meaningfully (>5%) cheaper than a random masked policy: learning is not demonstrated; not clearly cheaper than online greedy; not clearly cheaper than the static Distributed Greedy plan |
| H3: MARL coordinates under sparse communication (L2_Medium) | Cooperative MARL (shared DQfD) | Cooperative MARL (shared DQfD): 0.30 conflicts, 62% on time, cost 263; Online Distributed Greedy: 1.80 conflicts, 70% on time, cost 228; Distributed Greedy: 1.00 conflicts, 70% on time, cost 225; Random Masked Policy: 0.50 conflicts, 58% on time, cost 284 | more than 5% cheaper than a random masked policy (check the paired tests for significance); not clearly cheaper than online greedy; not clearly cheaper than the static Distributed Greedy plan |

### Table 3. Communication trade-off

| Method | Comm | Messages | End-to-end cost | Change in cost vs L0 | Static objective gap to assignment ILP |
|---|---|---:|---:|---:|---:|
| Distributed Greedy | L0_None | 0.0 | 287.0 | +0.0 | 50.7% |
| Distributed Greedy | L1_Low | 8.5 | 225.3 | -61.7 | 30.3% |
| Distributed Greedy | L2_Medium | 8.5 | 225.3 | -61.7 | 30.3% |
| Distributed Greedy | L3_High | 8.2 | 227.9 | -59.1 | 30.5% |
| Distributed Greedy | L4_Full | 8.2 | 227.9 | -59.1 | 30.5% |
| SSI Auction | L0_None | 0.0 | 306.6 | +0.0 | 56.3% |
| SSI Auction | L1_Low | 7.8 | 322.0 | +15.4 | 78.0% |
| SSI Auction | L2_Medium | 9.4 | 272.6 | -34.0 | 56.3% |
| SSI Auction | L3_High | 9.4 | 272.6 | -34.0 | 56.3% |
| SSI Auction | L4_Full | 9.4 | 272.6 | -34.0 | 56.3% |
| Online Distributed Greedy | L0_None | 0.0 | 511.6 | +0.0 | N/A (online policy) |
| Online Distributed Greedy | L1_Low | 9.6 | 283.7 | -227.9 | N/A (online policy) |
| Online Distributed Greedy | L2_Medium | 8.8 | 227.6 | -284.0 | N/A (online policy) |
| Online Distributed Greedy | L3_High | 7.6 | 215.6 | -296.0 | N/A (online policy) |
| Online Distributed Greedy | L4_Full | 7.6 | 215.6 | -296.0 | N/A (online policy) |
| Online Greedy (no-message convention) | L0_None | 0.0 | 338.4 | +0.0 | N/A (online policy) |
| Online Greedy (no-message convention) | L1_Low | 7.1 | 253.8 | -84.6 | N/A (online policy) |
| Online Greedy (no-message convention) | L2_Medium | 6.7 | 247.8 | -90.6 | N/A (online policy) |
| Online Greedy (no-message convention) | L3_High | 6.7 | 247.8 | -90.6 | N/A (online policy) |
| Online Greedy (no-message convention) | L4_Full | 6.7 | 247.8 | -90.6 | N/A (online policy) |
| Cooperative MARL (IDQN) | L0_None | 0.0 | 357.4 | +0.0 | N/A (online policy) |
| Cooperative MARL (IDQN) | L1_Low | 5.5 | 289.3 | -68.1 | N/A (online policy) |
| Cooperative MARL (IDQN) | L2_Medium | 5.7 | 295.7 | -61.7 | N/A (online policy) |
| Cooperative MARL (IDQN) | L3_High | 5.6 | 290.1 | -67.3 | N/A (online policy) |
| Cooperative MARL (IDQN) | L4_Full | 5.6 | 290.1 | -67.3 | N/A (online policy) |
| Cooperative MARL (shared DQfD) | L0_None | 0.0 | 280.7 | +0.0 | N/A (online policy) |
| Cooperative MARL (shared DQfD) | L1_Low | 6.9 | 269.5 | -11.2 | N/A (online policy) |
| Cooperative MARL (shared DQfD) | L2_Medium | 6.5 | 263.2 | -17.5 | N/A (online policy) |
| Cooperative MARL (shared DQfD) | L3_High | 6.4 | 257.4 | -23.3 | N/A (online policy) |
| Cooperative MARL (shared DQfD) | L4_Full | 6.4 | 257.4 | -23.3 | N/A (online policy) |
| Random Masked Policy | L0_None | 0.0 | 304.8 | +0.0 | N/A (online policy) |
| Random Masked Policy | L1_Low | 6.2 | 285.8 | -19.0 | N/A (online policy) |
| Random Masked Policy | L2_Medium | 6.3 | 284.5 | -20.3 | N/A (online policy) |
| Random Masked Policy | L3_High | 6.2 | 287.3 | -17.5 | N/A (online policy) |
| Random Masked Policy | L4_Full | 6.1 | 282.6 | -22.2 | N/A (online policy) |
