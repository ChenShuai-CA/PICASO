# Paper evidence freeze

This folder is a read-only synthesis of completed P2 experiments. It does not run training, simulation, or a second heldout evaluation.

The manuscript is manuscript_zh.md; manuscript_outline.md records the research-question and evidence mapping; references.bib contains the bibliography. figures/ contains PNG previews and vector PDF/SVG exports. tables/ contains standalone Markdown tables. evidence/ contains machine-readable CSV files and evidence_manifest.json with input/output hashes.

RQ3 uses the P2.11 independent screen for both K=1 and K=2. P2.12 remains the sole preregistered K=1 heldout confirmation. No post-unblinding K=2 heldout result is reported as confirmatory evidence.

Rebuild the package inside WSL Ubuntu with:

    /home/shuai/.venvs/scenario-gpu/bin/python scripts/build_paper_evidence.py
