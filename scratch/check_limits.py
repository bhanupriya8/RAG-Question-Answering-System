
# check_limits.py
import json, os, urllib.request

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
)
print(json.dumps(json.load(urllib.request.urlopen(req)), indent=2))