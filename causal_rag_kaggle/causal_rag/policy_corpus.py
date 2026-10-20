"""
Stage 3a - REAL policy corpus (no synthetic text).

Sources are public regulatory / GSE underwriting documents fetched from their official sites
(Kaggle notebook with Internet ON), plus ANY extra .txt/.html/.pdf you place in a Kaggle dataset
folder named `policy_docs/` (they are merged automatically).

If a download fails the stage keeps going with whatever was retrieved, and prints which sources
are missing -- it never invents text. Treatments with no supporting chunk are reported as
"analyst-defined risk factor" by the retriever (see retrieval.RetrieverAgent.policy_for).
"""
import re
import json
import time
import pathlib
import pandas as pd

from . import paths

SOURCES = [
    dict(id="fannie_b3_6_02", publisher="Fannie Mae Selling Guide", title="B3-6-02 Debt-to-Income Ratios",
         url="https://selling-guide.fanniemae.com/sel/b3-6-02/debt-income-ratios",
         alt=["https://fanniemae.com/content/guide/selling/b3/6/02.html"]),
    dict(id="fannie_b3_5_1_01", publisher="Fannie Mae Selling Guide", title="B3-5.1-01 General Requirements for Credit Scores",
         url="https://selling-guide.fanniemae.com/sel/b3-5.1-01/general-requirements-credit-scores",
         alt=["https://fanniemae.com/content/guide/selling/b3/5.1/01.html"]),
    dict(id="cfpb_reg_z_1026_43", publisher="CFPB / eCFR", title="12 CFR 1026.43 Minimum standards (ability to repay)",
         url="https://www.ecfr.gov/current/title-12/chapter-X/part-1026/subpart-E/section-1026.43",
         alt=["https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-12?part=1026&section=1026.43"]),
    dict(id="cfpb_reg_b_1002_6", publisher="CFPB / eCFR", title="12 CFR 1002.6 Rules concerning evaluation of applications (ECOA / Reg B)",
         url="https://www.ecfr.gov/current/title-12/chapter-X/part-1002/section-1002.6",
         alt=["https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-12?part=1002&section=1002.6"]),
]

# entity vocabulary used to link chunks <-> graph variables / treatments
ENTITY_PATTERNS = {
    "dti": r"debt[- ]to[- ]income|\bDTI\b|monthly (debt|obligation)",
    "income": r"\bincome\b|stable monthly income|qualifying income",
    "credit_score": r"credit score|representative score|\bFICO\b|credit history",
    "utilization": r"utili[sz]ation|credit (line|limit)s?|available credit|open[- ]end(ed)? credit",
    "repayment": r"ability to repay|repayment ability|repay the (loan|extension)|payment burden|loan amount",
    "prohibited_basis": r"\brace\b|\bcolor\b|\breligion\b|national origin|\bsex\b|marital status|\bage\b|public assistance",
}
UA = {"User-Agent": "Mozilla/5.0 (research; causal-rag-kaggle)"}


def _html_to_text(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "noscript", "form", "svg"]):
        t.decompose()
    node = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [re.sub(r"\s+", " ", x).strip() for x in node.get_text("\n").split("\n")]
    return "\n".join(x for x in lines if len(x) > 2)


def fetch_policy_docs(out_dir=None, verbose=True):
    import requests
    out = pathlib.Path(out_dir or paths.ART / "policy_docs")
    out.mkdir(parents=True, exist_ok=True)
    got, missing = [], []
    for s in SOURCES:
        f = out / f"{s['id']}.txt"
        if f.exists() and f.stat().st_size > 500:
            got.append(s["id"]); continue
        text = ""
        for url in [s["url"]] + s.get("alt", []):
            try:
                r = requests.get(url, headers=UA, timeout=40)
                if r.status_code == 200 and len(r.text) > 1000:
                    text = _html_to_text(r.text)
                    if len(text) > 800:
                        s["fetched_from"] = url; break
            except Exception as e:                                  # network / TLS / parse
                if verbose:
                    print(f"  ! {s['id']}: {type(e).__name__} on {url}")
            time.sleep(1)
        if len(text) > 800:
            f.write_text(text); got.append(s["id"])
            (out / f"{s['id']}.json").write_text(json.dumps({k: v for k, v in s.items()}))
        else:
            missing.append(s["id"])
    if verbose:
        print(f"policy docs fetched/cached: {got}\nmissing: {missing}")
    return got, missing


def load_user_docs():
    """Extra real documents you attach in a Kaggle dataset folder named policy_docs/ (txt, html, pdf)."""
    docs = []
    for d in paths.find_input("policy_docs", first=False) or []:
        for f in pathlib.Path(d).glob("*"):
            try:
                if f.suffix.lower() in {".txt", ".md"}:
                    t = f.read_text(errors="ignore")
                elif f.suffix.lower() in {".html", ".htm"}:
                    t = _html_to_text(f.read_text(errors="ignore"))
                elif f.suffix.lower() == ".pdf":
                    from pypdf import PdfReader
                    t = "\n".join((p.extract_text() or "") for p in PdfReader(str(f)).pages)
                else:
                    continue
                docs.append(dict(id=f"user_{f.stem}", publisher="user-supplied", title=f.stem, text=t))
            except Exception as e:
                print("  ! could not read", f.name, type(e).__name__)
    return docs


def chunk_text(text, max_words=150, overlap_sents=1):
    sents = re.split(r"(?<=[.;:])\s+|\n+", text)
    sents = [s.strip() for s in sents if len(s.split()) >= 4]
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= max_words:
            chunks.append(" ".join(cur)); cur = cur[-overlap_sents:] if overlap_sents else []
    if cur and (not chunks or len(" ".join(cur).split()) > 25):
        chunks.append(" ".join(cur))
    return chunks


def tag_entities(text):
    return [k for k, p in ENTITY_PATTERNS.items() if re.search(p, text, flags=re.I)]


def build_chunks(fetch=True, verbose=True):
    if fetch:
        fetch_policy_docs(verbose=verbose)
    rows = []
    docs = []
    root = paths.ART / "policy_docs"
    for s in SOURCES:
        f = root / f"{s['id']}.txt"
        if f.exists():
            docs.append(dict(id=s["id"], publisher=s["publisher"], title=s["title"], text=f.read_text()))
    docs += load_user_docs()
    if not docs:
        raise RuntimeError("No policy documents available. Turn Internet ON for the notebook, or attach a dataset "
                           "containing a policy_docs/ folder with real .txt/.html/.pdf documents.")
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"])):
            rows.append(dict(chunk_id=f"{d['id']}#{i:03d}", doc_id=d["id"], publisher=d["publisher"],
                             title=d["title"], text=c, entities=tag_entities(c)))
    df = pd.DataFrame(rows)
    df.to_parquet(paths.art("policy_chunks.parquet"))
    if verbose:
        print(f"{len(df)} policy chunks from {df.doc_id.nunique()} documents")
        print(df.explode("entities").entities.value_counts(dropna=False).to_string())
    return df
