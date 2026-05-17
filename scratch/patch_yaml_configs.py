import os
from pathlib import Path

configs_dir = Path("/Users/andrisoueslati/Code/TikZ/configs")
for path in configs_dir.glob("*.yaml"):
    content = path.read_text(encoding="utf-8")
    if "collapse_probe:" in content:
        lines = content.splitlines()
        has_probe_at_end_only = any("probe_at_end_only" in line for line in lines)
        
        if not has_probe_at_end_only:
            final_lines = []
            for line in lines:
                final_lines.append(line)
                if line.rstrip().endswith("collapse_probe:"):
                    # Indentation of nested properties should be current line indent + 2 spaces
                    indent = len(line) - len(line.lstrip()) + 2
                    final_lines.append(" " * indent + "probe_at_end_only: true")
            path.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
            print(f"Patched {path.name}")
        else:
            print(f"Already configured in {path.name}")
