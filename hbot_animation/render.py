#!/usr/bin/env python3
"""Render a 30 s, 3840x2160 MP4 of the HyperVan control panel running an
HBOT pressurization sequence: main lock first, then the entry lock.

The panel photo (panel.jpg) is a static, full-frame background. Only the
analogue controls move: gauge needles (removed by inpainting and redrawn per
frame), rotary knobs and quarter-turn valves (rotated in place) and indicator
lamps. No overlays, captions or digital readouts are drawn. The last frame
matches the first (everything off, all gauges at zero) so the clip loops
seamlessly.

Profile (time-compressed) follows the US Navy Treatment Table 6 shape:
compress to 60 fsw (with an ear-clearing hold at 10 fsw), hold at depth,
decompress to 30 fsw, hold, decompress to surface. The entry lock then
pressurizes to 60 fsw, holds, and vents to surface.
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "panel.jpg")
OUT = os.path.join(HERE, "hypervan_hbot_sequence_4k.mp4")

W, H = 3840, 2160
FPS = 30
DUR = 30.0
S = H / 1294                 # source -> plate scale (panel fills the height)
OX, OY = (W - round(2000 * S)) // 2, 0
BG = (21, 21, 21)            # matches the panel photo's background

# ---- element positions in source-image pixels -----------------------------
LIGHTS = {"supply": (203, 232, 38), "main": (203, 634, 38), "entry": (203, 1054, 38)}
VALVES = {"ac": (408, 232, 49), "main": (408, 631, 49), "entry": (408, 1051, 49)}
KNOBS = {"main_rate": (635, 627, 61), "main_set": (857, 627, 61),
         "entry_rate": (635, 1047, 61), "entry_set": (857, 1047, 61)}
BIG_GAUGES = {"main_cur": (1150, 614), "main_set": (1489, 614),
              "entry_cur": (1150, 1034), "entry_set": (1489, 1034)}
SMALL_GAUGES = {"source": (1156, 227), "mid": (1490, 227)}


def sp(v):
    return v * S


# ---- timeline ---------------------------------------------------------------
def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def keys(k):
    def f(t):
        if t <= k[0][0]:
            return k[0][1]
        for (t0, v0), (t1, v1) in zip(k, k[1:]):
            if t <= t1:
                return v0 + (v1 - v0) * smooth((t - t0) / (t1 - t0))
        return k[-1][1]
    return f


def step_on(t, t_on, t_off=1e9, ramp=0.15):
    return smooth((t - t_on) / ramp) * (1 - smooth((t - t_off) / ramp))


main_p = keys([(0, 0), (4.4, 0), (5.6, 10), (6.0, 10), (8.8, 60), (11.9, 60),
               (13.8, 30), (15.4, 30), (17.0, 0)])
main_set = keys([(0, 0), (3.6, 0), (4.3, 60), (11.5, 60), (12.0, 30),
                 (14.9, 30), (15.3, 0)])
main_rate = keys([(0, 0), (3.1, 0), (3.7, 0.55), (11.2, 0.55), (11.7, 0.2),
                  (17.1, 0.2), (17.6, 0)])
main_valve = keys([(0, 0), (2.8, 0), (3.4, 1), (17.0, 1), (17.5, 0)])

entry_p = keys([(0, 0), (19.8, 0), (20.6, 10), (20.9, 10), (23.2, 60),
                (25.4, 60), (27.8, 0)])
entry_set = keys([(0, 0), (19.0, 0), (19.7, 60), (25.0, 60), (25.4, 0)])
entry_rate = keys([(0, 0), (18.6, 0), (19.2, 0.8), (25.0, 0.8), (25.3, 0.5),
                   (27.9, 0.5), (28.4, 0)])
entry_valve = keys([(0, 0), (18.2, 0), (18.8, 1), (27.9, 1), (28.4, 0)])

ac_valve = keys([(0, 0), (1.0, 0), (1.6, 1), (28.8, 1), (29.3, 0)])
supply_p = keys([(0, 0), (1.4, 0), (3.0, 150), (28.9, 150), (29.7, 0)])

def flow(t):
    """Normalised gas draw from supply (compression only)."""
    dt = 0.12
    d = (max(main_p(t + dt) - main_p(t - dt), 0) + max(entry_p(t + dt) - entry_p(t - dt), 0)) / (2 * dt)
    return min(d / 25.0, 1.0)


def state(t):
    fl = flow(t)
    wob = lambda a, f: a * math.sin(t * f) * math.sin(t * f * 0.37 + 1.3)
    sup = supply_p(t)
    return {
        "supply_light": step_on(t, 0.8, 29.4, ramp=0.08) * (0.55 + 0.45 * (1 if t > 1.05 or (t * 30) % 3 > 1 else 0.3)),
        "main_light": step_on(t, 2.6, 17.7),
        "entry_light": step_on(t, 18.0, 28.5),
        "ac": ac_valve(t), "main_valve": main_valve(t), "entry_valve": entry_valve(t),
        "main_rate": main_rate(t), "main_setk": main_set(t) / 165.0,
        "entry_rate": entry_rate(t), "entry_setk": entry_set(t) / 165.0,
        "main_cur": main_p(t) + (wob(0.35, 5.0) if 8.8 < t < 11.9 else 0),
        "main_set": main_set(t - 0.25),
        "entry_cur": entry_p(t) + (wob(0.35, 5.0) if 23.2 < t < 25.4 else 0),
        "entry_set": entry_set(t - 0.25),
        "source": max(sup - 10 * fl + (wob(1.2, 23) if fl > 0 else 0), 0),
        "mid": max(sup * 0.8 - 30 * fl + (wob(1.5, 29) if fl > 0 else 0), 0),
    }


# ---- plate preparation -------------------------------------------------------
def detect_hub(gray, cx, cy, win=12):
    y0, x0 = int(cy - win), int(cx - win)
    patch = gray[y0:y0 + 2 * win, x0:x0 + 2 * win]
    ys, xs = np.nonzero(patch < 60)
    return x0 + xs.mean(), y0 + ys.mean()


def detect_needle(gray, cx, cy, r0, r1):
    best = []
    for a in np.arange(0, 360, 0.5):
        rr = np.linspace(r0, r1, 40)
        xs = (cx + rr * math.sin(math.radians(a))).astype(int)
        ys = (cy - rr * math.cos(math.radians(a))).astype(int)
        best.append((255 - gray[ys, xs].astype(float)).mean())
    best = np.array(best)
    a1 = np.argmax(best) * 0.5
    opp = [(a1 + 180 + d) % 360 for d in np.arange(-20, 20.5, 0.5)]
    a2 = max(opp, key=lambda a: best[int(round(a * 2)) % 720])
    return a1, a2


def needle_mask(shape, gray, cx, cy, angles, rmax, half_w, hub_r):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    m = (dx * dx + dy * dy) < hub_r * hub_r
    for a in angles:
        ux, uy = math.sin(math.radians(a)), -math.cos(math.radians(a))
        along = dx * ux + dy * uy
        perp = np.abs(-dx * uy + dy * ux)
        m |= (along > 0) & (along < rmax) & (perp < half_w) & (gray < 150)
    m = m.astype(np.uint8) * 255
    return cv2.dilate(m, np.ones((3, 3), np.uint8))


def disc_patch(img, cx, cy, r):
    """RGBA crop of a disc with a soft edge."""
    r = int(math.ceil(r))
    x0, y0 = int(round(cx)) - r - 2, int(round(cy)) - r - 2
    size = 2 * r + 4
    crop = img.crop((x0, y0, x0 + size, y0 + size)).convert("RGBA")
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(xx - (cx - x0), yy - (cy - y0))
    a = np.clip((r - d) / 1.5 + 0.5, 0, 1)
    crop.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    return crop, (x0, y0)


def build_plate():
    src = Image.open(SRC).convert("RGB")
    plate = src.resize((int(src.width * S), int(src.height * S)), Image.LANCZOS)
    arr = np.array(plate)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gauges = {}
    for name, (cx, cy) in {**BIG_GAUGES, **SMALL_GAUGES}.items():
        big = name in BIG_GAUGES
        hx, hy = detect_hub(gray, sp(cx), sp(cy), win=int(sp(10 if big else 6)))
        r0, r1 = (sp(18), sp(52)) if big else (sp(8), sp(24))
        a1, a2 = detect_needle(gray, hx, hy, r0, r1)
        rmax = sp(66) if big else sp(37)
        m = needle_mask(gray.shape, gray, hx, hy, (a1, a2), rmax,
                        sp(4.2) if big else sp(6.5), sp(13) if big else sp(6))
        arr = cv2.inpaint(arr, m, 3, cv2.INPAINT_NS)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gauges[name] = (hx, hy)
    plate = Image.fromarray(arr)

    # rotating parts and lamp states
    discs = {}
    for name, (cx, cy, r) in {**VALVES, **{k: v for k, v in KNOBS.items()}}.items():
        discs[name] = disc_patch(plate, sp(cx), sp(cy), sp(r))
    lamps = {}
    for name, (cx, cy, r) in LIGHTS.items():
        on, pos = disc_patch(plate, sp(cx), sp(cy), sp(r))
        rgb = np.array(on).astype(np.float32)
        lum = rgb[..., :3].mean(axis=2, keepdims=True)
        off = rgb.copy()
        off[..., :3] = (0.55 * lum + 0.45 * rgb[..., :3]) * 0.22
        on_boost = rgb.copy()
        on_boost[..., :3] = np.clip(rgb[..., :3] * 1.18 + 12, 0, 255)
        off_img = Image.fromarray(off.astype(np.uint8), "RGBA")
        on_img = Image.fromarray(on_boost.astype(np.uint8), "RGBA")
        # glow sprite
        colr = (80, 255, 60) if name == "supply" else (255, 40, 30)
        gs = int(sp(r) * 5)
        g = Image.new("RGBA", (gs, gs), colr + (0,))
        yy, xx = np.mgrid[0:gs, 0:gs]
        dd = np.hypot(xx - gs / 2, yy - gs / 2) / (sp(r))
        ga = np.clip(np.exp(-((dd - 0.6).clip(0)) ** 2 / 1.1) * 0.55, 0, 1) * (dd > 0.85)
        g.putalpha(Image.fromarray((ga * 255).astype(np.uint8)))
        g = g.filter(ImageFilter.GaussianBlur(sp(4)))
        lamps[name] = (off_img, on_img, pos, g, (int(sp(cx) - gs / 2), int(sp(cy) - gs / 2)))
        plate.paste(off_img, pos, off_img)

    return plate, gauges, discs, lamps


# ---- drawing helpers ---------------------------------------------------------
SS = 3  # supersampling for vector bits


def draw_big_needle(frame, cx, cy, val):
    ang = math.radians(-135 + 0.6 * min(max(val, -3), 460))
    L, T = sp(60), sp(26)
    size = int(2 * max(L, T) + sp(16))
    img = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    c = size * SS / 2
    ux, uy = math.sin(ang), -math.cos(ang)
    px, py = -uy, ux

    def pts(ofs):
        o = ofs * SS
        tip = [(c + o + ux * (L + sp(2)) * SS, c + o + uy * (L + sp(2)) * SS),
               (c + o + ux * (L - sp(6)) * SS - px * sp(2.6) * SS, c + o + uy * (L - sp(6)) * SS - py * sp(2.6) * SS),
               (c + o - px * sp(3.6) * SS, c + o - py * sp(3.6) * SS),
               (c + o - ux * T * SS - px * sp(0.9) * SS, c + o - uy * T * SS - py * sp(0.9) * SS),
               (c + o - ux * T * SS + px * sp(0.9) * SS, c + o - uy * T * SS + py * sp(0.9) * SS),
               (c + o + px * sp(3.6) * SS, c + o + py * sp(3.6) * SS),
               (c + o + ux * (L - sp(6)) * SS + px * sp(2.6) * SS, c + o + uy * (L - sp(6)) * SS + py * sp(2.6) * SS)]
        return tip

    def hexagon(ofs):
        o = ofs * SS
        r = sp(10.5) * SS
        return [(c + o + r * math.cos(ang + k * math.pi / 3), c + o + r * math.sin(ang + k * math.pi / 3)) for k in range(6)]

    ds = ImageDraw.Draw(sh)
    ds.polygon(pts(sp(2.5)), fill=(0, 0, 0, 90))
    ds.polygon(hexagon(sp(2.5)), fill=(0, 0, 0, 90))
    sh = sh.filter(ImageFilter.GaussianBlur(sp(2) * SS))
    d = ImageDraw.Draw(img)
    d.polygon(pts(0), fill=(12, 12, 12, 255))
    d.polygon(hexagon(0), fill=(18, 18, 18, 255))
    r2 = sp(2.2) * SS
    d.ellipse((c - r2, c - r2, c + r2, c + r2), fill=(70, 70, 70, 255))
    out = Image.alpha_composite(sh, img).resize((size, size), Image.LANCZOS)
    frame.alpha_composite(out, (int(round(cx - size / 2)), int(round(cy - size / 2))))


def draw_small_needle(frame, cx, cy, val):
    ang = math.radians(-135 + 1.35 * min(max(val, -2), 205))
    L, T = sp(28), sp(34)
    size = int(2 * T + sp(12))
    img = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    c = size * SS / 2
    ux, uy = math.sin(ang), -math.cos(ang)
    px, py = -uy, ux
    k = SS

    def P(a, b):
        return (c + (ux * a + px * b) * k, c + (uy * a + py * b) * k)

    d = ImageDraw.Draw(img)
    d.polygon([P(-T, -sp(0.7)), P(-T, sp(0.7)), P(sp(10), sp(1.6)), P(sp(22), sp(5.2)),
               P(L, sp(3.0)), P(L + sp(1.5), 0), P(L, -sp(3.0)), P(sp(22), -sp(5.2)),
               P(sp(10), -sp(1.6))], fill=(14, 14, 14, 255))
    r = sp(5) * k
    d.ellipse((c - r, c - r, c + r, c + r), fill=(20, 20, 20, 255))
    out = img.resize((size, size), Image.LANCZOS)
    frame.alpha_composite(out, (int(round(cx - size / 2)), int(round(cy - size / 2))))


def rotated(disc, deg):
    patch, pos = disc
    return patch.rotate(-deg, resample=Image.BICUBIC), pos


# ---- frame ---------------------------------------------------------------------
_G = {}


def init_worker():
    _G["plate"], _G["gauges"], _G["discs"], _G["lamps"] = build_plate()


def render(i):
    t = i / FPS
    st = state(t)
    plate, gauges, discs, lamps = _G["plate"], _G["gauges"], _G["discs"], _G["lamps"]
    layer = plate.copy().convert("RGBA")

    # lamps
    for name, key in (("supply", "supply_light"), ("main", "main_light"), ("entry", "entry_light")):
        off, on, pos, glow, gpos = lamps[name]
        a = st[key]
        if a > 0.01:
            blend = Image.blend(off, on, a)
            layer.alpha_composite(blend, pos)
            gl = glow.copy()
            gl.putalpha(gl.getchannel("A").point(lambda v, a=a: int(v * a)))
            layer.alpha_composite(gl, gpos)

    # quarter-turn valves: closed = bar horizontal, open = bar vertical
    for name, key in (("ac", "ac"), ("main", "main_valve"), ("entry", "entry_valve")):
        patch, pos = rotated(discs[name], 45 - 90 * st[key])
        layer.alpha_composite(patch, pos)

    # set knobs: indicator dot sweeps -135..+135 deg; original dot sits at -45
    for name, key in (("main_rate", "main_rate"), ("main_set", "main_setk"),
                      ("entry_rate", "entry_rate"), ("entry_set", "entry_setk")):
        patch, pos = rotated(discs[name], (-135 + 270 * st[key]) + 45)
        layer.alpha_composite(patch, pos)

    for name in BIG_GAUGES:
        cx, cy = gauges[name]
        draw_big_needle(layer, cx, cy, st[name])
    for name in SMALL_GAUGES:
        cx, cy = gauges[name]
        draw_small_needle(layer, cx, cy, st[name])

    frame = Image.new("RGBA", (W, H), BG + (255,))
    frame.alpha_composite(layer, (OX, OY))
    return frame.convert("RGB").tobytes()


def main():
    n = int(DUR * FPS)
    if len(sys.argv) > 1 and sys.argv[1] == "--stills":
        init_worker()
        for tt in map(float, sys.argv[2:]):
            Image.frombytes("RGB", (W, H), render(int(tt * FPS))).save(os.path.join(HERE, f"still_{tt:05.1f}.png"))
        return
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-level", "5.1", "-movflags", "+faststart", OUT]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(3, initializer=init_worker) as pool:
        for k, buf in enumerate(pool.imap(render, range(n), chunksize=4)):
            enc.stdin.write(buf)
            if k % 60 == 0:
                print(f"frame {k}/{n}", flush=True)
    enc.stdin.close()
    enc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
