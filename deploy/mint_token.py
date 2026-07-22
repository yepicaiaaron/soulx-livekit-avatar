"""Mint a LiveKit join token + meet URL from LIVEKIT_* env vars.

Usage (on the bot host):  python3 deploy/mint_token.py [identity]
Prints a meet.livekit.io join URL for the bot's room. Reads
LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET / LIVEKIT_ROOM from
the environment — no secrets in code.
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def main() -> int:
    url = os.environ["LIVEKIT_URL"]
    key = os.environ["LIVEKIT_API_KEY"]
    secret = os.environ["LIVEKIT_API_SECRET"]
    room = os.environ.get("LIVEKIT_ROOM", "soulx-flashhead-room")
    identity = sys.argv[1] if len(sys.argv) > 1 else "guest"

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": key, "sub": identity, "name": identity,
        "nbf": now - 60, "exp": now + 86400,
        "video": {"roomJoin": True, "room": room,
                  "canPublish": True, "canSubscribe": True,
                  "canPublishData": True},
    }
    signing = (b64u(json.dumps(header, separators=(",", ":")).encode()) + "." +
               b64u(json.dumps(payload, separators=(",", ":")).encode()))
    sig = b64u(hmac.new(secret.encode(), signing.encode(), hashlib.sha256).digest())
    token = signing + "." + sig

    print("https://meet.livekit.io/custom?liveKitUrl=" +
          urllib.parse.quote(url, safe="") + "&token=" + token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
