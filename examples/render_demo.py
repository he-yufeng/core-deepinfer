"""Render actual CLI stdout as a dependency-free README terminal capture."""

import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape


def main():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "core_deepinfer"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = ["$ python -m core_deepinfer", "", *result.stdout.rstrip().splitlines()]
    width = max(980, max(map(len, lines)) * 9 + 64)
    height = 68 + len(lines) * 22
    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        "<title>Core DeepInfer: actual local CPU demo output</title>",
        "<desc>Untrained CPU model with an injected clock and synthetic costs. Not a benchmark.</desc>",
        f'<rect width="{width}" height="{height}" rx="12" fill="#111827"/>',
        (
            '<text x="28" y="32" fill="#9ca3af" font-family="monospace" font-size="13">'
            "core-deepinfer / CPU reference</text>"
        ),
        '<g fill="#f3f4f6" font-family="monospace" font-size="14">',
    ]
    for index, line in enumerate(lines):
        svg.append(f'<text x="28" y="{64 + index * 22}" xml:space="preserve">{escape(line)}</text>')
    svg.extend(["</g>", "</svg>"])
    output = root / "assets/demo.svg"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(output.relative_to(root))


if __name__ == "__main__":
    main()
