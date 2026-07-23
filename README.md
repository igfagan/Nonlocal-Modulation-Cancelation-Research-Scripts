# Nonlocal-Modulation-Cancelation-Research-Scripts

# Nonlocal Modulation Cancelation — Lab Scripts

Research code from the quantum photonics lab of **Professor Joseph Lukens at Purdue University**, conducted in partnership with **Sandia National Laboratories** (LDRD FY25-0763).

> ⚠️ **Note:** Experimental data and proprietary device parameters are not included in this repository out of respect for the Sandia National Laboratories collaboration agreement. Scripts are shared for transparency and portfolio purposes.

---

## Overview

This repository contains Python and MATLAB scripts developed to support an ongoing experiment demonstrating **nonlocal modulation cancelation** using time-energy entangled photon pairs transmitted over a deployed fiber-optic network.

The experiment leverages **acousto-optic phase modulators (AOPMs)** developed at Sandia National Laboratories as quantum interconnect components. A key goal is to replace conventional electro-optic modulators (EOMs) with AOPMs, which offer a significant Vπ·L advantage and intrinsic unitarity — properties critical for preserving quantum information in frequency-bin encoded states.

---

## Background

In nonlocal modulation cancelation, a pair of time-energy entangled photons is split and each photon is routed to a spatially separated modulator. When both modulators are driven at the same RF frequency with sub-picosecond synchronization, the modulation sidebands cancel in coincidence measurements — a nonlocal quantum effect with no classical analog.

Key milestones this experiment targets:
- Demonstration at **true kilometer-scale** fiber distances using ASU's deployed network
- **Vπ < 1 V** modulation depth via Bessel zero-crossing in the AOPM
- Sub-picosecond RF synchronization via **radio-over-fiber (RFoF)** clock distribution

---

## Repository Contents

```
├── eom_characterization/
│   ├── vpi_sideband_fit.py        # OSA sideband fitting to extract EOM Vπ
│   ├── npz_parser.py              # Parse and analyze OSA .npz data files
│   └── frequency_rolloff.py       # Model Vπ frequency rolloff from DC spec
│
├── instrument_control/
│   ├── santec_tsl570.py           # TCP/IP control of Santec TSL-570 tunable laser
│   ├── santec_mpm210h.py          # TCP/IP control of Santec MPM-210H power meter
│   └── wavelength_sweep.py        # Automated wavelength sweep + power logging
│
├── aopm_analysis/
│   ├── klayout_oasis_parser.py    # Parse AOPM chip OASIS layout files
│   └── mzi_cell_mapper.py         # Map MZI cells (MZNC1–21) and IDT arrays
│
└── utils/
    ├── plot_jsi.py                # Joint Spectral Intensity (JSI) visualization
    └── coincidence_analysis.py    # Coincidence counting and histogram analysis
```

---

## Key Results

- Extracted EOM Vπ ≈ **27.3 V at 30 GHz** from OSA sideband fitting, consistent with frequency rolloff from a ~3–5 V DC specification
- Successfully performed **fiber-to-chip coupling alignment** on unsuspended AOPM chip
- Identified IDT arrays, MZI cells, CMOS driver regions, and edge couplers in AOPM OASIS layout

---

## Dependencies

```
numpy
scipy
matplotlib
pyvisa
```

---

## Affiliation

Purdue University — Department of Electrical and Computer Engineering  
Prof. Joseph Lukens Quantum Photonics Lab  
Sandia National Laboratories — LDRD FY25-0763
