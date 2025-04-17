# >>>
# https://gist.github.com/Cilyan/9424144
import base64
import struct


def fnv64(data):
    hash_ = 0xcbf29ce484222325
    for b in data:
        hash_ *= 0x100000001b3
        hash_ &= 0xffffffffffffffff
        hash_ ^= b
    return hash_

def hash_dn(dn: str, salt: str):
    # Turn dn into bytes with a salt, dn is expected to be ascii data
    data = salt.encode("ascii") + dn.encode("ascii")
    # Hash data
    hash_ = fnv64(data)
    # Pack hash (int) into bytes
    bhash = struct.pack("<Q", hash_)
    # Encode in base64. There is always a padding "=" at the end, because the
    # hash is always 64bits long. We don't need it.
    return base64.urlsafe_b64encode(bhash)[:-1].decode("ascii")
# <<<
