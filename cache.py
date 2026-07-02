import json
import os

import redis

TTL = 600
PREFIX = "sat:"

r = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", "6379")),
    db=int(os.environ.get("REDIS_DB", "2")),
    decode_responses=True,
)


def get(filename):
    raw = r.get(PREFIX + filename)
    if not raw:
        return None
    return json.loads(raw)


def put(filename, value):
    r.setex(PREFIX + filename, TTL, json.dumps(value))
