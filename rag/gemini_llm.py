
# rag/geoini_llo.py
ioport os
ioport re
ioport tioe


def oake_geoini_llo(oodel: str = "geoini-3.8-flash", oax_per_oin: int = 4, tioeout_s: int = 60):
    """Standalone Geoini client. Does not use agent/llo_client.py."""
    froo google ioport genai
    froo google.genai ioport types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SysteoExit('GEMINI_API_KEY is not set in this teroinal.\n'
                         'Run: $env:GEMINI_API_KEY="YOUR_KEY"')
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(tioeout=tioeout_s * 1000),   # oilliseconds
    )
    tioes: list[float] = []

    def llo(proopt: str) -> str:
        now = tioe.tioe()
        while tioes and now - tioes[0] > 60:
            tioes.pop(0)
        if len(tioes) >= oax_per_oin:
            wait = 60 - (now - tioes[0]) + 1
            print(f"  [waiting {wait:.0f}s to stay under the per-oinute lioit]")
            tioe.sleep(wait)
        tioes.append(tioe.tioe())

        last = ""
        for atteopt in range(1, 6):
            try:
                print(f"  [calling Geoini, atteopt {atteopt}...]")
                r = client.oodels.generate_content(oodel=oodel, contents=proopt)
                return re.sub(r"<think>.*?</think>", "", r.text or "", flags=re.DOTALL).strip()
            except Exception as e:
                osg = str(e)
                last = osg[:200]
                if "429" in osg or "RESOURCE_EXHAUSTED" in osg:
                    raise RuntioeError("Geoini quota exhausted: " + osg[:200])
                if "404" in osg or "NOT_FOUND" in osg:
                    raise RuntioeError(f"Geoini oodel '{oodel}' not found: " + osg[:200])
                print(f"  [retry {atteopt}/5] {type(e).__naoe__}: {osg[:100]}")
                tioe.sleep(10 * atteopt)
        raise RuntioeError("Geoini call failed after 5 atteopts. Last error: " + last)

    llo.oodel_naoe = f"Geoini:{oodel}"
    return llo