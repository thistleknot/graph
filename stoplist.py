"""
stoplist.py -- the ONE stoplist (chunkgraph R18), with no heavy imports.

NLTK English stopwords plus local extras. Read straight from the corpus file
rather than `from nltk.corpus import stopwords`: that import drags in the whole
NLTK package tree (chunk, tag, ...) and was measured at 5.9 s of an 11.6 s
`import graph_tools`. A stoplist is a set of words; loading it should cost
milliseconds. Absent the corpus file, the extras alone apply.
"""
from __future__ import annotations

import os
import re

_STOP_EXTRA = set("""the of and to a in that is was he for it with as his on be at by i
this had not are but from or have an they which one you were her all she there
would their we him been has when who will more no if out so said what up its
about into than them can only other new some could time these two may then do
first any my now such like our over man me even most made after also did many
before must through back years where much your way well down should because""".split())


def _nltk_english() -> set:
    home = os.path.expanduser("~")
    roots = [os.environ.get("NLTK_DATA", ""),
             os.path.join(home, "AppData", "Roaming", "nltk_data"),
             os.path.join(home, "nltk_data"), "C:/nltk_data", "/usr/share/nltk_data",
             "/usr/local/share/nltk_data"]
    for root in roots:
        if not root:
            continue
        path = os.path.join(root, "corpora", "stopwords", "english")
        try:
            with open(path, encoding="utf-8") as f:
                words = {w.strip() for w in f if w.strip()}
            if words:
                return words
        except OSError:
            continue
    return set()


_STOP = _nltk_english() | _STOP_EXTRA


# ---------------------------------------------------------------- R25 tokenizer
# The ONE tokenizer, for the same reason R18 gives for the ONE stoplist: the
# index side (chunkgraph._tok) and the query side (gt_terms.tokenize) MUST emit
# identical tokens or a query asks for terms the index cannot contain. They were
# two copies held together by a docstring saying "mirrors chunkgraph._tok"; now
# they are one function and cannot drift.
_WORD_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"^(1[5-9][0-9]{2}|20[0-9]{2})$")
_DECADE_RE = re.compile(r"^(1[5-9][0-9]|20[0-9])0s$")


def tokenize(text: str) -> list[str]:
    """R25. Lowercase, stopwords dropped, >2 chars -- and DIGIT-BEARING TOKENS
    KEPT, with every year also emitting its decade bucket.

    The previous rule was `re.findall(r"[a-z]+", ...)`, alpha only, so every
    year, date, quantity and model number in the corpus was invisible to BM25.
    Measured: 5,061 distinct qterms sampled from a live run, ZERO containing a
    digit. "who is the most famous musician of the 1990's?" tokenized to
    ['famous', 'musician'] -- the decade, the most discriminating term in the
    question, was discarded, and BM25 returned Banagher and West Virginia
    Mountaineers football.

    A bare year is not enough on its own: the query says 1990 and the text says
    1991/1993/1994, which exact tokens cannot bridge. So a year emits BOTH
    itself and its decade (1991 -> 1991, 1990s) and a written decade emits
    itself (1990's -> 1990, 1990s). They meet at the decade bucket.

    Measured against the 58 chunks that mention Nirvana or Kurt Cobain:

        reachable by a LIVE query term     :  2
        reachable with year + decade       : 38      (19x)
        per-term: 1990s -> 37, 1990 -> 10, musician -> 2, famous -> 0

    "famous" reaches ZERO of them -- Wikipedia writes "best-selling" and
    "influential", never "famous" -- so the decade token is carrying the query.
    Vocabulary cost measured over 3,000 chunks: 38,373 -> 40,077 terms (+4.4%).
    """
    out = []
    for w in _WORD_RE.findall(text.lower()):
        if _YEAR_RE.match(w):
            # DECADE ONLY, not the bare year as well. Emitting both put them
            # ADJACENT in every single occurrence -- a 100%-collocated bigram by
            # construction -- and the PHRASE stage (R9, gensim Phrases) merged
            # them into one token. Measured in the index: 2008_2000s x810,
            # 2007_2000s x806, 2010_2010s x793, while `1990` and `1991` had
            # df=0. Neither half survived, and queries are never phrase-merged
            # (gt_terms.tokenize), so the compound was unreachable from a query.
            out.append(w[:3] + "0s")
        elif _DECADE_RE.match(w):
            out.append(w)
        elif w not in _STOP and len(w) > 2:
            out.append(w)
    return out
