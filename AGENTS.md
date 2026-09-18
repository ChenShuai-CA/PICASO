# Workspace and execution environment

- The sole active workspace is `D:\Projects\Scenario_Generation_Research`, exposed inside WSL2 Ubuntu as `/mnt/d/Projects/Scenario_Generation_Research`.
- Run development, Python, tests, data processing and training inside WSL distribution `Ubuntu` (24.04), not native Windows Python.
- Use `/home/shuai/.venvs/scenario-gpu/bin/python`, or activate that environment first.
- From PowerShell, launch Linux commands using `wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research -- <command>`.
- Keep project source, results and data in this workspace. Do not create or migrate to a second project workspace under `/home/shuai/projects`.
- Preserve existing unrelated edits and deletions. Do not infer finished experiments from plans or pilot artifacts.
