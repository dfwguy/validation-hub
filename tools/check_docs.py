#!/usr/bin/env python3
"""Check tracked Markdown local links and SVG structure without network access."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    failures = []
    count = 0
    for name in filter(None, names):
        path = ROOT / name
        if path.suffix.lower() == '.md':
            text = path.read_text()
            # Ignore fenced examples: their placeholder links are not navigation.
            text = re.sub(r'```[\s\S]*?```', '', text)
            links = re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', text)
            links += re.findall(r'<(?:img|a)\b[^>]*?(?:src|href)=["\']([^"\']+)', text)
            for link in links:
                link = link.strip('<>')
                parsed = urlsplit(link)
                if parsed.scheme or not parsed.path:
                    continue
                target = (path.parent / unquote(parsed.path)).resolve()
                if not target.is_relative_to(ROOT) or not target.exists():
                    failures.append(f'{name}: broken local link {link}')
                count += 1
        if path.suffix == '.svg':
            try:
                svg = ET.parse(path).getroot()
                if svg.tag != '{http://www.w3.org/2000/svg}svg' or not svg.get('viewBox'):
                    failures.append(f'{name}: SVG needs namespace and viewBox')
            except ET.ParseError:
                failures.append(f'{name}: invalid SVG XML')
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'Documentation links checked: {count}; SVG structure passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
