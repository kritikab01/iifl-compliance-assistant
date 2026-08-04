import json, re, time
import streamlit as st
from rank_bm25 import BM25Okapi
from groq import Groq

st.set_page_config(page_title="IIFL Gold Loan Compliance Assistant", layout="wide")

MODEL = "llama-3.3-70b-versatile"

LABELS = {
    "rbi_nbfc_credit":   "RBI NBFC – Credit Facilities Directions, 2025",
    "rbi_nbfc_conduct":  "RBI NBFC – Responsible Business Conduct Directions, 2025",
    "rbi_nbfc_kyc":      "RBI NBFC – Know Your Customer Directions, 2025",
    "rbi_iifl_ban":      "RBI Press Release, 4 March 2024 (IIFL)",
    "rbi_consolidation": "RBI Press Release, 28 Nov 2025 (Consolidated MDs)",
    "iifl_annual":       "IIFL Finance Annual Report",
    "iifl_gl_rates":     "IIFL Gold Loan – Rates & Charges",
    "iifl_gl_repay":     "IIFL Gold Loan – Repayment",
    "iifl_gl_process":   "IIFL Gold Loan – Documents",
    "iifl_gl_eligibility": "IIFL Gold Loan – Eligibility",
}

HINDI_MAP = {
    "गोल्ड": "gold", "सोना": "gold", "सोने": "gold", "चांदी": "silver",
    "लोन": "loan", "ऋण": "loan", "एलटीवी": "ltv", "ब्याज": "interest",
    "लाख": "lakh", "अधिकतम": "maximum", "वापस": "return", "दिन": "days",
    "कागज": "documents", "दस्तावेज": "documents", "नीलामी": "auction",
    "गिरवी": "pledge", "शुद्धता": "purity", "वजन": "weight", "ग्राम": "gram",
    "आभूषण": "ornaments", "सिक्का": "coins", "सिक्के": "coins",
    "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "आठ": "8",
    "क्या": "what", "कितना": "how much", "कितने": "how many", "पर": "on",
}

def label(src):
    return LABELS.get(src, src)

@st.cache_data
def load_corpus():
    with open("corpus.json") as f:
        return json.load(f)

@st.cache_resource
def build_index(texts):
    return BM25Okapi([re.findall(r"[a-z0-9]+", t.lower()) for t in texts])

data = load_corpus()
chunks = data["chunks"]
bm25 = build_index([c["text"] for c in chunks])

def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def retrieve(query, k=5):
    """BM25 retrieval. Devanagari terms mapped to English for search only.
    No extra API call — the map is local and deterministic."""
    toks = re.findall(r"[a-z0-9]+", query.lower())
    mapped = None
    if len(toks) < 2:
        mapped = " ".join(HINDI_MAP.get(w.strip("?।,."), "") for w in query.split())
        mapped = re.sub(r"\s+", " ", mapped).strip()
        toks = re.findall(r"[a-z0-9]+", mapped.lower())
    if not toks:
        return [], mapped
    scores = bm25.get_scores(toks)
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(chunks[i], scores[i]) for i in idx if scores[i] > 0], mapped

SYSTEM = """You are a compliance assistant for gold loan branch staff at an Indian NBFC.

RULES — follow exactly:
1. Answer ONLY from the numbered PASSAGES provided. Use no outside knowledge.
2. After every factual statement, cite like this: [P2]
3. If the passages do not contain the answer, reply with exactly: NOT FOUND
   Then say which document a branch employee should check.
4. Never guess a number. If a figure is not in the passages, say NOT FOUND.
5. If the question asks about a specific figure and the passages only contain a
   related but different figure, say NOT FOUND. Do not substitute.
6. Quote paragraph numbers from the passages when they appear.
7. Be brief. A branch employee is reading this with a customer waiting.
8. Reply in the same language the question was asked in.
9. The identity of the borrower - name, gender, religion, city, age, occupation -
   is IRRELEVANT to what the rules permit. Never let it change your answer."""

def ask(query):
    hits, mapped = retrieve(query)
    if not hits:
        return "NOT FOUND — no relevant passage in the indexed corpus.", [], 0.0, mapped
    block = "\n\n".join(
        f"[P{i+1}] ({label(c['source'])}, page {c['page']})\n{c['text']}"
        for i, (c, s) in enumerate(hits)
    )
    t0 = time.time()
    r = get_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"PASSAGES:\n{block}\n\nQUESTION: {query}"},
        ],
    )
    return r.choices[0].message.content, hits, time.time() - t0, mapped

# ---------- UI ----------
st.title("IIFL Gold Loan Compliance Assistant")
st.caption(
    f"{data['n_chunks']} passages · {data['n_sources']} sources · "
    f"corpus frozen {data['frozen_on']} · every answer cited or NOT FOUND"
)

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing. Add it under Manage app → Settings → Secrets.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Ask", "Case check", "Evidence", "Stack"])

with tab1:
    q = st.text_input(
        "Question — English, Hindi or Hinglish",
        "What is the maximum LTV for a consumption gold loan of Rs 2 lakh?",
    )
    if st.button("Ask", type="primary"):
        with st.spinner("Searching the rulebook..."):
            try:
                ans, hits, secs, mapped = ask(q)
            except Exception as e:
                msg = str(e)
                if "rate_limit" in msg or "429" in msg:
                    st.warning(
                        "Daily free-tier token limit reached (100,000 tokens/day). "
                        "Resets shortly. This is a quota ceiling on the free plan, "
                        "not a system fault — a paid tier or on-premise deployment "
                        "removes it."
                    )
                else:
                    st.error(f"Error: {msg}")
                st.stop()
        if ans.strip().upper().startswith("NOT FOUND"):
            st.warning(ans)
        else:
            st.success(ans)
        if mapped:
            st.caption(f"Retrieved using mapped terms: “{mapped}”")
        st.caption(f"Answered in {secs:.1f}s")
        with st.expander(f"Sources used ({len(hits)})"):
            for i, (c, s) in enumerate(hits):
                st.markdown(f"**[P{i+1}] {label(c['source'])} — page {c['page']}**")
                st.text(c["text"][:900])

with tab2:
    st.subheader("Check a case against the rules")
    c1, c2, c3 = st.columns(3)
    wt    = c1.number_input("Gross weight (g)", value=40.0, step=1.0)
    carat = c2.number_input("Purity (carat)", value=22.0, step=1.0)
    rate  = c3.number_input("24K rate (Rs/g)", value=7000.0, step=100.0)
    c4, c5 = st.columns(2)
    loan  = c4.number_input("Loan amount (Rs)", value=200000.0, step=10000.0)
    cash  = c5.number_input("Cash disbursed (Rs)", value=0.0, step=1000.0)

    if st.button("Check", type="primary"):
        value = wt * (carat / 24.0) * rate
        ltv = (loan / value * 100) if value else 0
        cap = 85 if loan <= 250000 else (80 if loan <= 500000 else 75)

        m1, m2, m3 = st.columns(3)
        m1.metric("Collateral value", f"Rs {value:,.0f}")
        m2.metric("Actual LTV", f"{ltv:.1f}%")
        m3.metric("Permitted cap", f"{cap}%")

        if ltv > cap:
            st.error(f"BREACH — LTV {ltv:.1f}% exceeds the {cap}% cap. "
                     "(Credit Facilities Directions 2025, para 43)")
        else:
            st.success(f"LTV within the {cap}% cap. (para 43)")

        if wt > 1000:
            st.error("BREACH — gold ornaments above 1 kg per borrower. (para 39)")
        if cash > 20000:
            st.error("BREACH — cash above Rs 20,000. "
                     "(Income Tax Act s.269SS, referenced at para 46)")
        st.caption("Valuation uses intrinsic metal only — no gems or stones. (para 42)")

with tab3:
    st.subheader("Measured results")
    st.caption("47 automated tests, temperature 0, run against this corpus.")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Accuracy", "14/15")
    e2.metric("Bias variance", "0 / 18")
    e3.metric("Hallucination", "9/10")
    e4.metric("Median latency", "0.9s")

    st.markdown("""
**Bias — 18 tests, five dimensions.** One case held constant (40 g, 22 carat,
₹2,00,000). Only borrower identity varied. The permitted LTV never moved from 85%.

| Dimension | Variants tested | Answer varied |
|---|---|---|
| Gender | Rajesh / Rajni / Priya / Suresh | No |
| Region | Mumbai / Muzaffarpur / Imphal / Srinagar / Bengaluru | No |
| Community | Sharma / Khan / D'Souza / Singh | No |
| Occupation | Salaried / daily wage labourer / small farmer | No |
| Age | 32 / 68 | No |

**Language access — before and after.** BM25 tokenised on Latin characters only, so
a Devanagari query produced an empty search and never reached the rulebook. A local
domain term map fixed it, with no additional model call.

| Input | Before | After |
|---|---|---|
| English | Pass | Pass |
| Hinglish | Pass | Pass |
| Hindi | **Fail — NOT FOUND** | **Pass, answered in Hindi** |
| Broken English | Pass | Pass |
| Rate | 75% | **100%** |

**Hallucination — 10 fabricated rules.** Nine correctly returned NOT FOUND. The
tenth asked for IIFL's gold loan NPA ratio; the tool returned the company-wide
Gross NPA instead. Not invention — substitution of an adjacent real figure, which
is harder to detect than a refusal. Rule 5 of the instruction now forbids it.

**Latency and throughput.** 0.9s median on single queries; 6.7s under sustained
load. The free tier caps at 100,000 tokens per day — at roughly 2,100 tokens per
query, about 47 queries. Adequate for a pilot, not for 2,800 branches. The
constraint is quota, not inference.
    """)

with tab4:
    st.subheader("Why this stack")
    st.markdown("""
| Layer | Choice | Cost | Why |
|---|---|---|---|
| Retrieval | BM25 (rank_bm25) | Rs 0 | Regulatory queries are terminology-exact. Dense embeddings need PyTorch, which exceeds the free deployment memory ceiling. |
| Generation | Groq, Llama 3.3 70B | Rs 0 | Sub-second inference. Temperature 0 for reproducibility. |
| Multilingual | Local term map | Rs 0 | Hindi handled without a second API call. |
| Interface | Streamlit | Rs 0 | |
| Hosting | Streamlit Community Cloud | Rs 0 | |

**No model was trained.** Retrieval-augmented generation only. When RBI reissues a
direction we re-index one PDF; a fine-tuned model would need retraining and could
not cite a paragraph number.

**Corpus:** 745 raw pages reduced to 296 indexed. 447 pages of financial statements
were excluded at ingestion because unrelated high-frequency text degrades retrieval
precision on regulatory queries. Every document downloaded directly from rbi.org.in
or iifl.com — no third-party reproductions.

**Known limit:** free tier caps at 100,000 tokens/day. Dev tier or on-premise
deployment removes it.
    """)
