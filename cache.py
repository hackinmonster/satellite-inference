import hashlib
import json
import os

import redis

TTL = 600
PREFIX = "sat:"
r = None


def connect():
    global r
    try:
        client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=int(os.environ.get("REDIS_DB", "2")),
            decode_responses=True,
        )
        client.ping()
        r = client
        return True
    except redis.RedisError as e:
        print(f"redis unavailable ({e}), continuing without cache")
        r = None
        return False


def _key(data: bytes):
    return PREFIX + hashlib.sha256(data).hexdigest()


def get(data: bytes):
    if r is None:
        return None
    raw = r.get(_key(data))
    if not raw:
        return None
    return json.loads(raw)


def put(data: bytes, value):
    if r is None:
        return
    r.setex(_key(data), TTL, json.dumps(value))
