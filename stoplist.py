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
