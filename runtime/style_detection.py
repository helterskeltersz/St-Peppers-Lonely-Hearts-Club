"""Art-style vocabulary detection (VERIFIED port of winner
`trainer/utils/style_detection.py`).

A task is "style-shaped" when a known art-style term appears in at least
ZAYDEN_STYLE_MIN_PCT (25%) of the dataset captions. Matched on word boundaries,
longest surface form first, one count per caption. Truthy result => style path
(AdamW). This replaces the crude single-word regex that misfired on a stray
"aesthetic" inside a clearly person-triggered set.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_STYLE_ALIASES = {
    "Watercolor Painting": ["watercolor"], "Oil Painting": ["oil paint", "oil painted"],
    "Acrylic Painting": ["acrylic paint", "acrylic painted"],
    "Digital Art": ["digital artwork", "digitally created"],
    "Pencil Sketch": ["pencil drawing", "pencil sketched", "pencil"],
    "Comic Book Style": ["comic book", "comic style"],
    "Cyberpunk": ["cyberpunk style", "cyberpunk aesthetic"],
    "Steampunk": ["steampunk style", "steampunk aesthetic"], "Impressionist": ["impressionistic"],
    "Pop Art": ["pop art style"], "Minimalist": ["minimalistic"],
    "Gothic": ["gothic style", "gothic aesthetic"], "Art Nouveau": ["art nouveau style"],
    "Pixel Art": ["pixel graphics", "8-bit art", "8 bit art"], "Anime": ["anime style", "anime-style"],
    "3D Render": ["3d rendered", "3d rendering", "three dimensional render"], "Low Poly": ["low polygon"],
    "Photorealistic": ["photo realistic"], "Vector Art": ["vector graphics", "vector illustration"],
    "Abstract Expressionism": ["abstract expressionist", "abstract expressionistic"],
    "Realism": ["realist"], "Futurism": ["futurist", "futuristic"], "Cubism": ["cubist", "cubistic"],
    "Surrealism": ["surrealist", "surrealistic"], "Baroque": ["baroque style"],
    "Renaissance": ["renaissance style"], "Fantasy Illustration": [],
    "Sci-Fi Illustration": ["sci-fi", "science fiction"], "Ukiyo-e": [],
    "Line Art": ["line drawing", "line work"],
    "Black and White Ink Drawing": ["black and white ink", "ink drawing"], "Graffiti Art": [],
    "Stencil Art": [], "Flat Design": ["flat style"], "Isometric Art": [],
    "Retro 80s Style": ["80s style", "eighties style"],
    "Vaporwave": ["vaporwave aesthetic", "vaporwave style"], "Dreamlike": ["dream-like"],
    "High Fantasy": [], "Dark Fantasy": [], "Medieval Art": [], "Art Deco": ["art deco style"],
    "Hyperrealism": ["hyperrealistic", "hyperrealist"], "Sculpture Art": [], "Caricature": [],
    "Chibi": ["chibi style"], "Noir Style": ["noir", "film noir"], "Lowbrow Art": [],
    "Psychedelic Art": ["psychedelic"], "Vintage Poster": [], "Manga": ["manga style", "manga-style"],
    "Holographic": [], "Kawaii": ["kawaii style"], "Monochrome": ["monochromatic"],
    "Geometric Art": [], "Photocollage": [], "Mixed Media": [], "Ink Wash Painting": [],
    "Charcoal Drawing": [], "Concept Art": [], "Digital Matte Painting": [], "Pointillism": [],
    "Expressionism": ["expressionist", "expressionistic"], "Sumi-e": [],
    "Retro Futurism": ["retro futuristic", "retrofuturistic"], "Pixelated Glitch Art": [],
    "Neon Glow": [], "Street Art": [], "Bauhaus": [], "Flat Cartoon Style": [],
    "Carved Relief Art": [], "Fantasy Realism": [],
}

IMAGE_STYLES = list(_STYLE_ALIASES.keys())
ZAYDEN_STYLE_MIN_PCT = 25.0


def _compile():
    rows = []
    for style, aliases in _STYLE_ALIASES.items():
        forms = sorted({style.lower(), *(a.lower() for a in aliases)}, key=len, reverse=True)
        for form in forms:
            rows.append((style, len(form), re.compile(rf"\b{re.escape(form)}\b")))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


_MATCHERS = _compile()


def detect_styles_in_prompts(prompts: List[str], style_list=None) -> List[Tuple[str, float]]:
    """Return [(style, percent)] for styles mentioned in >= 25% of captions.
    Truthy == style-shaped (AdamW path)."""
    allowed = set(style_list) if style_list is not None else None
    total = len(prompts)
    if total == 0:
        return []
    hits = {}
    for prompt in prompts:
        text = (prompt or "").lower()
        seen = set()
        for style, _len, rx in _MATCHERS:
            if style in seen:
                continue
            if allowed is not None and style not in allowed:
                continue
            if rx.search(text):
                seen.add(style)
        specific = {s for s in seen
                    if not any(s != o and s.lower() in o.lower() for o in seen)}
        for style in specific:
            hits[style] = hits.get(style, 0) + 1
    out = []
    for style, count in hits.items():
        pct = round(count / total * 100, 2)
        if pct >= ZAYDEN_STYLE_MIN_PCT:
            out.append((style, pct))
    return out
