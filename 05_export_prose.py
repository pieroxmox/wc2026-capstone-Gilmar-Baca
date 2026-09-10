"""Export the report text from docs/index.html to a plain-text file.

Headings and body paragraphs only; tables, figures and captions are omitted.

Output: docs/prose_for_review.txt
"""

import html
import re

src = open("docs/index.html", encoding="utf-8").read()
body = src[src.index("<header"):src.index("<h3>Reproducibility")]
body = re.sub(r"<table.*?</table>", "", body, flags=re.S)
body = re.sub(r"<figure>.*?</figure>", "", body, flags=re.S)
body = re.sub(r"<nav.*?</nav>", "", body, flags=re.S)

out = []
for tag, text in re.findall(r"<(h1|h2|h3|p)[^>]*>(.*?)</\1>", body, flags=re.S):
    t = html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        continue
    if tag in ("h1", "h2", "h3"):
        out.append(f"\n\n{t.upper()}\n")
    else:
        out.append(f"{t}\n")

txt = "".join(out).lstrip("\n")
open("docs/prose_for_review.txt", "w", encoding="utf-8").write(txt)
print(f"docs/prose_for_review.txt: {len(txt.split()):,} words")
