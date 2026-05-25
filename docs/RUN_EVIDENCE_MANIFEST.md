# Run Evidence Manifest

This folder is a GitHub-ready subset of the local computational run folders. It keeps the files needed to demonstrate the workflow while excluding confidential thesis material and heavy numerical scratch artifacts.

## Primary Public Proof Files

- `docs/Novyte_Materials_DRDO_Technical_Brief.md`
  - Non-confidential technical brief.
  - Maps Novyte Materials' completed AI/ML-DFT work to DRDO's requested activity areas.

- `docs/Novyte_AI_ML_DFT_Inverse_Materials_Workflow.md`
  - Expanded workflow note.
  - Documents the algebraic objective functions, inverse-design loop, DFT validation ladder, and extension to emissivity/IR-response optimization.

## Included Run Evidence

The `run_evidence/` folder includes:

- `.in` files: Quantum ESPRESSO and related input decks.
- `.out` files: calculation output logs from relaxation, SCF, phonon, elastic, and thermal workflows.
- `.dynG` and `.dyn*` files: dynamical-matrix/phonon calculation artifacts.
- `.ps` and `.png` files: generated plots from equation-of-state and Debye/thermal runs.
- `thermo_control`: thermodynamic workflow control files.
- `elastic.out`: elastic tensor and mechanical stability output files.
- `relaxed_structure.txt`, `.mold`, `.axsf`: selected structure and visualization artifacts where present.

## Excluded Scratch Data

The local system contains much larger raw calculation folders. These are not required for a compact proof package and are not suitable for normal GitHub storage.

Excluded items include:

- wavefunction files,
- charge-density files,
- temporary SCF/phonon scratch folders,
- restart folders,
- cache folders,
- large `.save` directories,
- generated binary/intermediate files.

## What The Evidence Demonstrates

The included files demonstrate that the work was not only conceptual. The run folders contain direct traces of:

- candidate structure preparation,
- geometry relaxation,
- self-consistent field calculations,
- phonon/dynamic stability checks,
- elastic/mechanical stability checks,
- thermal/Debye output generation,
- computational screening and validation of candidate materials.

This is the run evidence supporting Novyte Materials' demonstrated AI/ML-DFT materials discovery capability.
