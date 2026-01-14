import datetime
import unicodedata
import re
from pathlib import Path

#  note: use "{:,}".format for comma-style number formatting
def format_k(n):  # 2514 -> 2.5k
    if n == 0:
        return "-"
    return f"{n / 1000:.1f}k".rstrip("0").rstrip(".")


def tidy_unicode(txt):
    txt = unicodedata.normalize("NFKC", txt)
    replacements = {
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "§": "§",  # keep if you want, or replace with 'section'
        "\u00a0": " ",  # non-breaking space
    }
    for k, v in replacements.items():
        txt = txt.replace(k, v)
    return txt

# Write EXL regexes in latex-friendly format
def make_exl_regex_latex(outf, exl_regex):
    with open(outf, "w", encoding="utf-8") as f:
        for cat in exl_regex:
            f.write(f"{cat}: {tidy_unicode(exl_regex[cat].pattern)}\n")
        f.write("% Generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", str(s))[:20]


def write_reviews(df_subset, out_dir, ascending):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(df_subset.sort_values("Q", ascending=ascending).iterrows(), start=1):
        fname = (
            f"{i:04d}_"
            f"{safe_filename(row['campaign'])}__"
            f"{safe_filename(row['reviewID'])}.txt"
        )
        path = out_dir / fname

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"reviewID: {row['reviewID']}\n")
            f.write(f"campaign: {row['campaign']}\n")
            f.write(f"Q: {row['Q']:.4f}\n\n")
            f.write(row["flattenedReview"])