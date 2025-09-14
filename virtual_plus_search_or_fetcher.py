#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
virtual_plus_search_or_fetcher.py  

Environment:
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install pandas requests beautifulsoup4 rapidfuzz unidecode tqdm

Input CSV requirements:
    Must contain: title, virtualsite_url
    ******************************************************************Must be title but not Title ***********************************************************************
    Example row:
        Active learning of neural population dynamics using two-photon holographic optogenetics,https://neurips.cc//virtual/2024/poster/93697

Typical command:
python virtual_plus_search_or_fetcher.py \
  --input csv_file_name \
  --enable-web-search-fallback \
  --max-workers 5 \
  --fuzzy-high 95 --fuzzy-low 88 \
  --web-query-tokens 8 \
  --save-fail-html \
  --row-sleep 0.4

Resume unmatched (salvage):
python virtual_plus_search_or_fetcher.py \
  --input csv_file_name \
  --enable-web-search-fallback \
  --resume-unmatched \
  --max-workers 3 \
  --fuzzy-high 93 --fuzzy-low 85 \
  --web-query-tokens 9 \
  --save-fail-html \
  --row-sleep 0.5

Use override file:
python virtual_plus_search_or_fetcher.py \
  --input csv_file_name \
  --override-file override_links.csv \
  --resume-unmatched

To keep meta columns in output, add:
  --keep-meta


You need to delete the last column by yourself :)
"""

import argparse, os, re, sys, json, time, random, unicodedata
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
from rapidfuzz import fuzz
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- Constants ---------------- #
FORUM_PREFIX = "https://openreview.net/forum?id="
OPENREVIEW_SEARCH_BASE = "https://openreview.net/search?term="
OPENREVIEW_FULL_REGEX = re.compile(r"https://openreview\.net/forum\?id=[A-Za-z0-9_-]+")
OPENREVIEW_RELAX_REGEX = re.compile(r"forum\?id=[A-Za-z0-9_-]+")
FORUM_ID_ONLY_REGEX = re.compile(r'["\']?forum["\']?\s*[:=]\s*["\']([A-Za-z0-9_-]{6,})["\']')
POSSIBLE_JSON_REGEX = re.compile(r'\{.*\}', re.DOTALL)

STOPWORDS = set("""
a an the of for to in on and with without by via from at is are be been being this that those these using use toward towards
into over under between among across against as it its their his her our your toward towarding towarded data model learning
""".split())

GREEK_MAP = {
    'α':'alpha','β':'beta','γ':'gamma','δ':'delta','ε':'epsilon','θ':'theta',
    'λ':'lambda','μ':'mu','ν':'nu','π':'pi','ρ':'rho','σ':'sigma','τ':'tau',
    'φ':'phi','χ':'chi','ψ':'psi','ω':'omega','∇':'nabla'
}
SUPERSCRIPT_MAP = {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'}

# -------------- Normalization -------------- #
def normalize_title(title: str) -> str:
    if not isinstance(title,str):
        return ""
    t = unicodedata.normalize("NFKC", title)
    for k,v in GREEK_MAP.items(): t = t.replace(k,v)
    for k,v in SUPERSCRIPT_MAP.items(): t = t.replace(k,v)
    t = re.sub(r'\$[^$]*\$', ' ', t)
    t = re.sub(r'\\\(|\\\)', ' ', t)
    t = unidecode(t).lower()
    t = re.sub(r'\^(\d+)', r'\1', t)
    t = re.sub(r'[^a-z0-9\s]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def build_search_tokens(title: str, max_tokens: int) -> List[str]:
    base = normalize_title(title).split()
    tokens = [w for w in base if w not in STOPWORDS and len(w) > 2]
    if not tokens:
        tokens = base
    return tokens[:max_tokens]

def build_queries_multi(title: str, max_tokens: int, min_tokens: int = 2) -> List[str]:
    tokens = build_search_tokens(title, max_tokens)
    queries = []
    for k in range(len(tokens), min_tokens-1, -1):
        q = " ".join(tokens[:k])
        if q and q not in queries:
            queries.append(q)
    return queries

# -------------- HTTP -------------- #
def fetch(url: str, timeout: int, retries: int, base_sleep: float, jitter: float,
          accept=(200,), giveup=(400,401,403,404)) -> Optional[str]:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ORFetcherV4/1.0)"})
    attempt=0
    while attempt < retries:
        try:
            r = sess.get(url, timeout=timeout)
            if r.status_code in accept:
                return r.text
            if r.status_code in giveup:
                return None
        except Exception:
            pass
        attempt += 1
        time.sleep(base_sleep*(1.6**(attempt-1))+random.uniform(0,jitter))
    return None

# -------------- Virtual Parsing -------------- #
def normalize_virtual_url(u: str) -> str:
    if not u: return ""
    u = u.strip()
    if u.startswith("//"):
        u = "https:" + u
    elif not u.startswith("http"):
        u = "https://" + u.lstrip("/")
    u = re.sub(r'(?<!:)//+', '/', u)
    if u.startswith("https:/") and not u.startswith("https://"):
        u = u.replace("https:/","https://",1)
    return u

def extract_forum_link_deep(html: str) -> str:
    if not html: return ""
    m = OPENREVIEW_FULL_REGEX.search(html)
    if m: return m.group(0)
    m2 = OPENREVIEW_RELAX_REGEX.search(html)
    if m2:
        fid = m2.group(0).split("forum?id=")[-1]
        return FORUM_PREFIX + fid
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.startswith(FORUM_PREFIX):
            return h
        if "forum?id=" in h:
            fid = h.split("forum?id=")[-1]
            return FORUM_PREFIX + fid
        fid_attr = a.get("data-forum-id")
        if fid_attr:
            return FORUM_PREFIX + fid_attr
    for tag in soup.find_all(attrs=True):
        for k,v in tag.attrs.items():
            if isinstance(v,str) and "forum?id=" in v:
                fid = v.split("forum?id=")[-1]
                return FORUM_PREFIX + fid
    for sc in soup.find_all("script"):
        txt = sc.string or sc.get_text()
        if not txt: continue
        for m3 in FORUM_ID_ONLY_REGEX.finditer(txt):
            fid = m3.group(1)
            if len(fid) >= 6:
                return FORUM_PREFIX + fid
        if "forum?id=" in txt:
            m4 = OPENREVIEW_RELAX_REGEX.search(txt)
            if m4:
                fid = m4.group(0).split("forum?id=")[-1]
                return FORUM_PREFIX + fid
    return ""

def try_virtual_all(base_url: str, timeout: int, retries: int,
                    base_sleep: float, jitter: float,
                    save_fail_html: bool, idx: int) -> Tuple[str,str]:
    variants = []
    core = normalize_virtual_url(base_url)
    if not core:
        return "", "virtual_no_url"
    variants.append( (core, "virtual_primary") )
    variants.append( (core + "?showControls=true", "virtual_variant_controls") )
    variants.append( (core + "?showPaper=true", "virtual_variant_paper") )
    if not core.endswith("/"):
        variants.append( (core + "/", "virtual_variant_slash") )

    for url,label in variants:
        html = fetch(url, timeout, retries, base_sleep, jitter)
        if html:
            link = extract_forum_link_deep(html)
            if link:
                return link, label
            else:
                if save_fail_html:
                    os.makedirs("debug_html/virtual",exist_ok=True)
                    with open(f"debug_html/virtual/row_{idx}_{label}.html","w",encoding="utf-8") as f:
                        f.write(html[:200000])
        else:
            if save_fail_html:
                os.makedirs("debug_html/virtual_fail",exist_ok=True)
                with open(f"debug_html/virtual_fail/row_{idx}_{label}.txt","w",encoding="utf-8") as f:
                    f.write(f"FETCH_FAIL {url}")
    return "", "virtual_all_failed"

# -------------- Web Search Fallback -------------- #
def parse_search_html(html: str) -> List[Tuple[str,str]]:
    anchors=[]
    if not html: return anchors
    soup = BeautifulSoup(html,"html.parser")

    for a in soup.find_all("a", href=True):
        h=a["href"]
        if "/forum?id=" in h:
            fid = h.split("forum?id=")[-1].split("&")[0]
            anchors.append((a.get_text(" ",strip=True), FORUM_PREFIX+fid))

    script_texts=[]
    for sc in soup.find_all("script"):
        txt = sc.string or sc.get_text()
        if not txt: continue
        if "forum?id=" in txt or '"forum"' in txt:
            script_texts.append(txt)

    fids=set()
    for txt in script_texts:
        for m in re.finditer(r'forum\?id=([A-Za-z0-9_-]{6,})', txt):
            fids.add(m.group(1))
        for m in FORUM_ID_ONLY_REGEX.finditer(txt):
            fids.add(m.group(1))

    titles=[]
    for txt in script_texts:
        for m in re.finditer(r'"title"\s*:\s*"([^"]{5,200})"', txt):
            raw=m.group(1)
            if "\\u" in raw:
                try:
                    raw = bytes(raw,'utf-8').decode('unicode_escape')
                except Exception:
                    pass
            titles.append(raw)

    if fids and titles:
        if len(fids)==len(titles):
            for t,fid in zip(titles, list(fids)):
                anchors.append((t, FORUM_PREFIX+fid))
        else:
            for fid in fids:
                anchors.append(("", FORUM_PREFIX+fid))
    elif fids:
        for fid in fids:
            anchors.append(("", FORUM_PREFIX+fid))
    return anchors

def web_search_multi_queries(title: str,
                             high: int, low: int,
                             max_tokens: int,
                             timeout: int, retries: int,
                             base_sleep: float, jitter: float,
                             save_fail_html: bool, idx: int,
                             cache: Dict[str,str], cache_dirty: List[bool]) -> Tuple[str,str,float,List[dict],str]:
    queries = build_queries_multi(title, max_tokens, min_tokens=2)
    all_candidates=[]
    for q in queries:
        if q in cache:
            html=cache[q]
        else:
            html=fetch(OPENREVIEW_SEARCH_BASE + requests.utils.quote(q),
                       timeout, retries, base_sleep, jitter)
            if html:
                cache[q]=html
                cache_dirty[0]=True
            else:
                if save_fail_html:
                    os.makedirs("debug_html/search_fail",exist_ok=True)
                    with open(f"debug_html/search_fail/row_{idx}_query_{len(q.split())}.txt","w",encoding="utf-8") as f:
                        f.write(f"HTTP_FAIL query={q}")
                continue
        pairs=parse_search_html(html)
        for (t,link) in pairs:
            all_candidates.append((t,link,q))
        if len(all_candidates) >= 50:
            break

    dedup={}
    for t,link,q in all_candidates:
        if link not in dedup:
            dedup[link]=(t,q)
    scored=[]
    norm_target=normalize_title(title)
    for link,(t,q) in dedup.items():
        norm_c=normalize_title(t)
        score=fuzz.token_set_ratio(norm_target,norm_c) if norm_c else 0
        scored.append({"candidate_title":t,"query":q,"score":score,"link":link,"normalized":norm_c})
    scored.sort(key=lambda x:x["score"], reverse=True)

    if not scored:
        return "unmatched","",0.0,scored,"web_search_no_candidates"

    best=scored[0]
    if best["score"]>=high:
        return "accepted", best["link"], best["score"], scored,"web_search_accept"
    elif best["score"]>=low:
        return "ambiguous","", best["score"], scored,"web_search_ambiguous"
    return "unmatched","", best["score"], scored,"web_search_unmatched"

# -------------- Override -------------- #
def load_override(path: Optional[str]) -> Dict[str,str]:
    if not path or not os.path.exists(path):
        return {}
    import csv
    mp={}
    with open(path,"r",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for r in reader:
            t=r.get("title") or r.get("Title")
            link=r.get("openreview_link") or r.get("link")
            if t and link and link.startswith(FORUM_PREFIX):
                mp[normalize_title(t)] = link.strip()
    print(f"[OVERRIDE] loaded {len(mp)} entries.")
    return mp

# -------------- Row Processing -------------- #
def process_row(idx:int, row:pd.Series, args, override_map, web_cache, web_cache_dirty, salvage:bool=False):
    title=str(row["title"])
    norm=normalize_title(title)
    existing_link=row.get("openreview_link","")
    existing_status=row.get("match_status","")
    existing_reason=row.get("reason_detail","")

    if salvage:
        if existing_status != "unmatched":
            return {
                "row_index": idx,
                "openreview_link": existing_link,
                "match_status": existing_status,
                "match_strategy": row.get("match_strategy",""),
                "match_score": row.get("match_score",""),
                "reason_detail": existing_reason or "resume_skip"
            }

    if norm in override_map:
        return {
            "row_index": idx,
            "openreview_link": override_map[norm],
            "match_status":"accepted",
            "match_strategy":"override",
            "match_score":"",
            "reason_detail":"override"
        }

    if not salvage and isinstance(existing_link,str) and existing_link.startswith(FORUM_PREFIX):
        return {
            "row_index": idx,
            "openreview_link": existing_link,
            "match_status": existing_status or "accepted",
            "match_strategy": row.get("match_strategy","previous"),
            "match_score": row.get("match_score",""),
            "reason_detail": existing_reason or "previous"
        }

    virtual_reason=""
    if not args.skip_virtual:
        vurl=str(row.get("virtualsite_url","")).strip()
        if vurl:
            link, vr = try_virtual_all(
                vurl,
                timeout=args.virtual_timeout,
                retries=args.virtual_retries,
                base_sleep=args.retry_base_sleep,
                jitter=args.retry_jitter,
                save_fail_html=args.save_fail_html,
                idx=idx
            )
            if link:
                return {
                    "row_index": idx,
                    "openreview_link": link,
                    "match_status":"accepted",
                    "match_strategy":"virtualsite",
                    "match_score":"",
                    "reason_detail": vr
                }
            else:
                virtual_reason = vr
        else:
            virtual_reason = "virtual_no_url"

    if args.enable_web_search_fallback:
        status, link, score, candidates, wr = web_search_multi_queries(
            title,
            args.fuzzy_high,
            args.fuzzy_low,
            args.web_query_tokens,
            timeout=args.web_search_timeout,
            retries=args.web_search_retries,
            base_sleep=args.retry_base_sleep,
            jitter=args.retry_jitter,
            save_fail_html=args.save_fail_html,
            idx=idx,
            cache=web_cache,
            cache_dirty=web_cache_dirty
        )
        if status=="accepted":
            return {
                "row_index": idx,
                "openreview_link": link,
                "match_status":"accepted",
                "match_strategy":"websearch",
                "match_score":score,
                "reason_detail": f"{virtual_reason}+{wr}" if virtual_reason else wr
            }
        elif status=="ambiguous":
            return {
                "row_index": idx,
                "openreview_link":"",
                "match_status":"ambiguous",
                "match_strategy":"websearch",
                "match_score":score,
                "reason_detail": f"{virtual_reason}+{wr}" if virtual_reason else wr,
                "candidates": candidates
            }
        else:
            combined = f"{virtual_reason}+{wr}" if virtual_reason else wr
            return {
                "row_index": idx,
                "openreview_link":"",
                "match_status":"unmatched",
                "match_strategy":"none",
                "match_score":"",
                "reason_detail": combined
            }

    return {
        "row_index": idx,
        "openreview_link":"",
        "match_status":"unmatched",
        "match_strategy":"none",
        "match_score":"",
        "reason_detail": virtual_reason or "unmatched"
    }

# -------------- Main -------------- #
def autodetect_conf_year(name:str):
    base=os.path.basename(name)
    m=re.search(r"(NeurIPS|ICLR|ICML|CVPR|ECCV|AAAI|KDD)[-_]?(\d{4})", base, re.IGNORECASE)
    if m: return m.group(1), m.group(2)
    return "Conference","Year"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    ap.add_argument("--conference")
    ap.add_argument("--year")
    ap.add_argument("--override-file")
    ap.add_argument("--skip-virtual", action="store_true")
    ap.add_argument("--enable-web-search-fallback", action="store_true")
    ap.add_argument("--web-search-timeout", type=int, default=20)
    ap.add_argument("--web-search-retries", type=int, default=2)
    ap.add_argument("--web-query-tokens", type=int, default=6)
    ap.add_argument("--fuzzy-high", type=int, default=96)
    ap.add_argument("--fuzzy-low", type=int, default=90)
    ap.add_argument("--virtual-timeout", type=int, default=18)
    ap.add_argument("--virtual-retries", type=int, default=3)
    ap.add_argument("--retry-base-sleep", type=float, default=1.0)
    ap.add_argument("--retry-jitter", type=float, default=0.3)
    ap.add_argument("--row-sleep", type=float, default=0.4)
    ap.add_argument("--row-jitter", type=float, default=0.3)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--limit-rows", type=int)
    ap.add_argument("--resume-unmatched", action="store_true")
    ap.add_argument("--save-fail-html", action="store_true")
    # 新增: 是否保留 meta 列
    ap.add_argument("--keep-meta", action="store_true",
                    help="保留 match_status/match_strategy/match_score/status_log 列 (默认输出前会删除)")
    args=ap.parse_args()

    if not os.path.exists(args.input):
        print("[ERROR] input not found")
        sys.exit(1)

    conf,year=autodetect_conf_year(args.input)
    if args.conference: conf=args.conference
    if args.year: year=args.year
    if not args.output:
        args.output = f"{conf}_{year}_openreview_link.csv"

    df=pd.read_csv(args.input)
    if "title" not in df.columns:
        print("[ERROR] need title column")
        sys.exit(1)
    if args.limit_rows:
        df=df.head(args.limit_rows).copy()

    # 确保列存在（内部流程使用）；最终根据 keep-meta 决定是否删除
    for col in ["openreview_link","match_status","match_strategy","match_score","reason_detail","status_log"]:
        if col not in df.columns:
            df[col]=""

    override_map=load_override(args.override_file)

    web_cache_path="websearch_cache_v4.json"
    if os.path.exists(web_cache_path):
        try:
            web_cache=json.load(open(web_cache_path,"r",encoding="utf-8"))
        except Exception:
            web_cache={}
    else:
        web_cache={}
    web_cache_dirty=[False]

    if args.resume_unmatched:
        indices=[i for i in df.index if df.at[i,"match_status"]=="unmatched"]
        print(f"[RESUME] unmatched rows: {len(indices)} / {len(df)}")
    else:
        indices=list(df.index)
        print(f"[RUN] total rows to process: {len(indices)}")

    results={}
    def dispatch(i):
        return process_row(i, df.loc[i], args, override_map, web_cache, web_cache_dirty, salvage=args.resume_unmatched)

    if args.max_workers<=1:
        for i in tqdm(indices, desc="Processing"):
            results[i]=dispatch(i)
            time.sleep(args.row_sleep+random.uniform(0,args.row_jitter))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            fut_map={ex.submit(dispatch,i):i for i in indices}
            for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="Processing"):
                res=fut.result()
                results[res["row_index"]]=res

    ambiguous=[]
    unmatched=[]
    for i in df.index:
        if i in results:
            r=results[i]
            df.at[i,"openreview_link"]=r["openreview_link"]
            df.at[i,"match_status"]=r["match_status"]
            df.at[i,"match_strategy"]=r["match_strategy"]
            df.at[i,"match_score"]=r["match_score"]
            df.at[i,"reason_detail"]=r["reason_detail"]
            if r["match_status"]=="ambiguous":
                ambiguous.append({
                    "row_index": i,
                    "title": df.at[i,"title"],
                    "candidates": r.get("candidates",[])
                })
            if r["match_status"]=="unmatched":
                unmatched.append({
                    "row_index": i,
                    "title": df.at[i,"title"],
                    "normalized": normalize_title(df.at[i,"title"]),
                    "reason_detail": r["reason_detail"]
                })

    # 输出原始统计（内部用）
    total=len(df)
    accepted=(df["match_status"]=="accepted").sum()
    amb_cnt=(df["match_status"]=="ambiguous").sum()
    un_cnt=(df["match_status"]=="unmatched").sum()
    print("========== STATS (before strip) ==========")
    print(f"Total: {total}")
    print(f"Accepted: {accepted}")
    print(f"Ambiguous: {amb_cnt}")
    print(f"Unmatched: {un_cnt}")
    print("Strategy distribution:")
    print(df["match_strategy"].value_counts())
    print("Reason detail (top 15):")
    print(df["reason_detail"].value_counts().head(15))
    print("================================")

    if ambiguous:
        json.dump(ambiguous, open("ambiguous.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[INFO] ambiguous: {len(ambiguous)}")
    if unmatched:
        json.dump(unmatched, open("unmatched.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[INFO] unmatched: {len(unmatched)}")
        pd.DataFrame(unmatched).to_csv("unmatched_diagnosis.csv", index=False)

    if web_cache_dirty[0]:
        json.dump(web_cache, open(web_cache_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print("[CACHE] websearch cache updated.")

    # -------- 新增：根据是否保留 meta 列输出 --------
    if args.keep_meta:
        df_out = df
    else:
        drop_cols = [c for c in ["match_status","match_strategy","match_score","status_log"] if c in df.columns]
        if drop_cols:
            print(f"[OUTPUT] dropping meta columns: {drop_cols}")
            df_out = df.drop(columns=drop_cols)
        else:
            df_out = df

    df_out.to_csv(args.output, index=False)
    print(f"[DONE] wrote: {args.output}")

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")