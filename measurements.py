import re, datetime

# m_act, m_ver and m_gnd are computed externally and calculated in loaders.py during data loading

# LEN: Length of the flattened review in characters
def m_len(flattened_review):
    return len(flattened_review)/1000  # simple and effective :)

# ITX: Number of items in core sections. Count calculated during data reading (see loaders.py)
def m_itx(seg_summary_count, seg_strength_count, seg_weaknesses_count):
    return seg_summary_count + seg_strength_count + seg_weaknesses_count

# EXL: Number of explicit links (Figures, Tables...) in the flattened review
def m_exl(flattened_review, debug=False, context_size=50):
    matches = {}
    blacklist = ["L0", "L1", "L2", "L3", "l0", "l1", "l2", "l3"]
    for cat, pattern in EXL_REGEX.items():
        matches[cat] = []
        for m in pattern.finditer(flattened_review):
            if m.group(0) not in blacklist:
                start, end = m.start(), m.end()
                left = max(0, start - context_size)
                right = min(len(flattened_review), end + context_size)

                matches[cat].append({  # explicit links grouped by type
                    "match": m.group(0),
                    "context": flattened_review[left:right],
                })
    if debug:
        return matches  # return all matches for debugging
    else:
        return sum(len(v) for v in matches.values())  # otherwise return total expl. links found

# Dump explicit matches for manual analysis
def util_dump_m_exl(matches, outf):
    with open(outf, "w") as f:
        for k in matches:
            for x in matches[k]:
                ctx = x['context'].strip().replace('\t', '').replace("\n", "")
                f.write(f"{k}\t{x['match']}\t{ctx}\n")
        f.write("% Generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

EXL_REGEX = {
    "section": re.compile(
        r"\b(?:sections?|sec\.?|§|chapter|subsection)\s*[0-9]+(?:\.[0-9]+)*"  # Section 5, Sec. 5, Chapter 8
        r"|\b(?:introduction|related work|method(?:s|ology)?|approach|experiments?|results?|"  # Introduction section
        r"analysis|discussion|limitations?|conclusion|ablation|future work)\s+section\b",
        re.IGNORECASE,  # important -- section at the end. not sure about that one.
    ),
    "table": re.compile(r"\b(?:table|tab\.)\s*[0-9]+[a-zA-Z]?", re.IGNORECASE),  # Table 4, Tab. 2a
    "figure": re.compile(r"\b(?:figure|fig\.)\s*[0-9]+[a-zA-Z]?", re.IGNORECASE),  # Fig. 3a
    "theorem": re.compile(r"\btheorem\s*[0-9]+[a-zA-Z]?", re.IGNORECASE),  # Theorem 2
    "lemma": re.compile(r"\blemma\s*[0-9]+[a-zA-Z]?", re.IGNORECASE),  # Lemma 3
    "equation": re.compile(  # Eq. 3
        r"\b(?:equation|eq\.)\s*\(?[0-9]+(?:\.[0-9]+)?\)?", re.IGNORECASE  # Eq. 4
    ),
    "formula": re.compile(r"\bformula\s*\(?[0-9]+(?:\.[0-9]+)?\)?", re.IGNORECASE),  # Formula 3a
    "algorithm": re.compile(r"\b(algorithm|alg\.?)\s*\(?[0-9]+(?:\.[0-9]+)?\)?", re.IGNORECASE),  # Formula 3a
    "paragraph": re.compile(r"\bparagraphs?\b", re.IGNORECASE),
    "line": re.compile(
        r"(?<![a-zA-Z])(?:line|lines|l\.?|L)\s*[\(\s]*[0-9]+(?:\s*(?:-|–|—|,|to)\s*[0-9]+)*[\)\s]*",
        # line 4, l. 9, L9
        re.IGNORECASE,
    ),
    "page": re.compile(r"(?<![a-z])(?:page|pg?\.?)\s*[0-9]+", re.IGNORECASE),  # p.5
    "footnote": re.compile(r"\bfootnote\s*[0-9]+", re.IGNORECASE),  # footnote 4
    "appendix": re.compile(r"\bappendix\b", re.IGNORECASE)
}
