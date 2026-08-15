"""Rate every heat exchanger spec in data/heat_exchangers/ and write a
summary table to figures/hx_ratings.txt."""
from _common import FIG
import os
from pathlib import Path
from thermohx import specs

spec_dir = specs.DATA_DIR / "heat_exchangers"
lines = [f"{'exchanger':<34} {'arrangement':<15} {'q [kW]':>8} {'eps':>6} "
         f"{'NTU':>6} {'Th_out [K]':>11} {'Tc_out [K]':>11}"]
lines.append("-" * len(lines[0]))
for path in sorted(spec_dir.glob("*.yaml")):
    spec = specs.load_spec(path)
    r, info = specs.rate_hx_spec(spec)
    lines.append(f"{spec['name']:<34} {spec['arrangement']:<15} "
                 f"{r.q/1e3:8.1f} {r.effectiveness:6.3f} {r.NTU:6.3f} "
                 f"{r.Th_out:11.1f} {r.Tc_out:11.1f}")
text = "\n".join(lines) + "\n"
print(text)
out = Path(FIG) / "hx_ratings.txt"
out.write_text(text)
print("wrote", out)
