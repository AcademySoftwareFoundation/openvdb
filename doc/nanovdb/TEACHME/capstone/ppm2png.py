#!/usr/bin/env python3
"""Convert a binary P6 PPM to PNG using only the Python standard library
(the container has no ImageMagick / PIL). Usage: python3 ppm2png.py in.ppm out.png"""
import sys, struct, zlib, binascii

def main(src, dst):
    data = open(src, "rb").read()
    i = 0
    def tok(i):
        n = len(data)
        while i < n and data[i] in b" \t\n\r": i += 1
        s = i
        while i < n and data[i] not in b" \t\n\r": i += 1
        if s == i: raise ValueError("truncated or non-binary PPM")
        return data[s:i], i
    magic, i = tok(i)
    assert magic == b"P6", "not a binary PPM"
    w, i = tok(i); h, i = tok(i); mx, i = tok(i)
    i += 1                       # single whitespace after maxval
    W, H = int(w), int(h)
    px = data[i:i + W * H * 3]

    def chunk(typ, payload):
        return (struct.pack(">I", len(payload)) + typ + payload +
                struct.pack(">I", binascii.crc32(typ + payload) & 0xffffffff))

    raw = bytearray()
    for y in range(H):
        raw.append(0)            # filter: none
        raw += px[y * W * 3:(y + 1) * W * 3]
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    open(dst, "wb").write(png)
    print(f"wrote {dst} ({W}x{H})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
