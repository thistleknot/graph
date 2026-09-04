"""patch_neo4j_browser.py -- produce the auto-login browser zip the chunkgraph
neo4j container mounts.

Spec: playbook.md T26 lessons (operator: zero-click login to the embedded
console). The stock unified console ignores auto-connect URL params; the
CLASSIC browser honors ?preselectAuthMethod=NO_AUTH&connectURL=... but is only
selected via localStorage. This script injects a one-line seed into the served
index.html so fresh profiles default to the classic browser, then the walker's
iframe params connect with zero clicks (server runs NEO4J_AUTH=none, local).

Usage:
    python patch_neo4j_browser.py            # reads zip from the container
Then recreate the container:
    docker stop chunkgraph-neo4j; docker rm chunkgraph-neo4j
    docker run -d --name chunkgraph-neo4j -p 7474:7474 -p 7687:7687 \
      -v chunkgraph-neo4jdata:/data \
      -v "<repo>/.neo4j-web/<zipname>:/var/lib/neo4j/web/<zipname>" \
      -e NEO4J_AUTH=none \
      -e "NEO4J_dbms_security_http__static__content__security__policy__header=\
default-src 'self'; script-src 'self' 'unsafe-inline' cdn.segment.com canny.io; \
img-src 'self' guides.neo4j.com data:; style-src 'self' fonts.googleapis.com \
'unsafe-inline'; font-src 'self' fonts.gstatic.com; base-uri 'none'; \
object-src 'none'; frame-ancestors 'self' http://localhost:8501; \
connect-src 'self' api.canny.io api.segment.io ws: wss: http: https:" \
      neo4j:latest
"""
import os
import subprocess
import sys
import zipfile

SEED = ('<script>try{if(localStorage.getItem("prefersOldBrowser")===null)'
        'localStorage.setItem("prefersOldBrowser","true");}catch(e){}</script>')
OUT_DIR = ".neo4j-web"


def main():
    name = subprocess.check_output(
        ["docker", "exec", "chunkgraph-neo4j", "sh", "-c",
         "ls /var/lib/neo4j/web/"], text=True).strip()
    src = os.path.join(OUT_DIR, "_stock_" + name)
    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.check_call(["docker", "cp",
                           f"chunkgraph-neo4j:/var/lib/neo4j/web/{name}", src])
    zin = zipfile.ZipFile(src)
    html = zin.read("browser/index.html").decode("utf-8")
    if SEED in html:
        print("already patched (container serves the patched zip); nothing to do")
        return 0
    html = html.replace("<head>", "<head>" + SEED, 1)
    dst = os.path.join(OUT_DIR, name)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = (html.encode("utf-8") if item.filename == "browser/index.html"
                    else zin.read(item.filename))
            zout.writestr(item, data)
    os.remove(src)
    print("patched zip:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
