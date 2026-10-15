import re
from pathlib import Path

index_html = Path("web/index.html").read_text(encoding="utf-8")
style_css = Path("web/css/style.css").read_text(encoding="utf-8")
app_js = Path("web/js/app.js").read_text(encoding="utf-8")
scope_js = Path("web/js/oscilloscope.js").read_text(encoding="utf-8")

# 1. Back button check
assert 'id="btn-back-overview"' in index_html, "btn-back-overview missing in HTML"
assert "btn-back-overview" in app_js, "btn-back-overview missing in app.js"

# 2. 8 Views check
views = [
    "view-overview",
    "view-face",
    "view-ecg",
    "view-ppg",
    "view-analytics",
    "view-history",
    "view-reports",
    "view-settings",
]
for v in views:
    assert f'id="{v}"' in index_html, f"{v} section missing in index.html"
    assert f'data-view="{v}"' in index_html, f"{v} nav button missing in index.html"
    assert f'"{v}"' in app_js, f"{v} missing in titles dictionary in app.js"

# 3. Design token checks
assert "#f6f5ee" in style_css.lower(), "Cream base token missing in CSS"
assert "#faf9f5" in style_css.lower(), "Card subtle token missing in CSS"
assert "#ece8df" in style_css.lower(), "Border subtle token missing in CSS"
assert "#18181b" in style_css.lower(), "Pitch black token missing in CSS"
assert "#59554e" in style_css.lower(), "Taupe secondary text missing in CSS"
assert "#78736b" in style_css.lower(), "Eyebrow tertiary text missing in CSS"

# 4. Zero legacy hex codes
legacy = [
    "#222222",
    "#717171",
    "#059669",
    "#d97706",
    "#e11d48",
    "#f7f7f7",
    "#dddddd",
    "#ebebeb",
    "#2563eb",
    "#10b981",
    "#3b82f6",
    "#f43f5e",
    "#f59e0b",
]
all_files = [
    ("index.html", index_html),
    ("style.css", style_css),
    ("app.js", app_js),
    ("oscilloscope.js", scope_js),
]
for fname, content in all_files:
    for code in legacy:
        matches = re.findall(re.escape(code), content, re.IGNORECASE)
        assert len(matches) == 0, f"Found {len(matches)} occurrences of {code} in {fname}"

print("ALL 6 DESIGN SYSTEM VERIFICATIONS PASSED PERFECTLY!")
