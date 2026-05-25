# Novyte Materials AI/ML-DFT Materials Discovery Proof Package

This repository contains a compact, non-confidential proof package for Novyte Materials' completed AI/ML-assisted materials discovery run with first-principles validation.

## Contents

- `docs/Novyte_Materials_DRDO_Technical_Brief.md`  
  Short non-confidential technical brief prepared for DRDO-style technical assessment.

- `run_evidence/`  
  Lightweight run evidence copied from the local calculation folders. It includes Quantum ESPRESSO input files, output logs, phonon/dynamical-matrix files, elastic output files, and thermodynamic plot files. Heavy scratch files, wavefunctions, charge-density files, temporary restart folders, and cache artifacts are intentionally excluded.

- `docs/RUN_EVIDENCE_MANIFEST.md`  
  Explanation of what the run evidence proves.

- `docs/Novyte_AI_ML_DFT_Inverse_Materials_Workflow.md`  
  Technical note explaining the Novyte workflow, algebraic formulation, DFT validation ladder, and how the same workflow maps to inverse design for thermal-signature-reduction and infrared-response-controlled materials.

## Positioning

The completed work demonstrates that Novyte Materials can:

- generate candidate functional crystalline materials using custom reinforcement learning (RL) and deep learning models,
- screen candidates using structural and chemical descriptors,
- validate shortlisted candidates using DFT calculations,
- evaluate thermodynamic, dynamic, mechanical, and thermal stability,
- extend the same workflow toward emissivity, infrared response, and coating-material optimization.

This package is intended as proof of computational capability and run traceability without exposing confidential thesis material.
