import argparse
import math
import threading
import time
from typing import Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def log(message: str):
    try:
        print(message, flush=True)
    except OSError:
        # Stdout can be detached by some runner/supervisor setups.
        pass


def quat_from_euler_xyz(rx: float, ry: float, rz: float):
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    qw = cx * cy * cz - sx * sy * sz
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    return qx, qy, qz, qw


def start_listener(ip: str, port: int):
    disp = Dispatcher()

    def handler(addr, *args: Any):
        ts = time.strftime("%H:%M:%S")
        log(f"[LISTEN {ts}] {addr} {args}")

    disp.set_default_handler(handler)
    server = ThreadingOSCUDPServer((ip, port), disp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log(f"[LISTEN] Listening on {ip}:{port}")
    return server


class DualClient:
    def __init__(self, host: str, port: int, mirror_host: str | None = None, mirror_port: int | None = None):
        self.primary = SimpleUDPClient(host, port)
        self.mirror = SimpleUDPClient(mirror_host, mirror_port) if (mirror_host and mirror_port) else None

    def send(self, addr: str, args):
        self.primary.send_message(addr, args)
        if self.mirror:
            self.mirror.send_message(addr, args)


class VMCSender:
    def __init__(self, host: str, port: int, mirror_host: str | None = None, mirror_port: int | None = None):
        self.client = DualClient(host, port, mirror_host, mirror_port)

    def ok(self):
        # Minimal status packet supported by old and new VMC implementations.
        self.client.send("/VMC/Ext/OK", 1)

    def tick(self):
        self.client.send("/VMC/Ext/T", float(time.time()))

    def root(self, name: str, x: float, y: float, z: float, q):
        qx, qy, qz, qw = q
        self.client.send(
            "/VMC/Ext/Root/Pos",
            [name, float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)],
        )

    def bone(self, name: str, x: float, y: float, z: float, q):
        qx, qy, qz, qw = q
        self.client.send(
            "/VMC/Ext/Bone/Pos",
            [name, float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)],
        )

    def tracker(self, name: str, x: float, y: float, z: float, q):
        qx, qy, qz, qw = q
        self.client.send(
            "/VMC/Ext/Tra/Pos",
            [name, float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)],
        )

    def blend(self, name: str, value: float):
        self.client.send("/VMC/Ext/Blend/Val", [name, float(value)])

    def apply(self):
        self.client.send("/VMC/Ext/Blend/Apply", [])


# Send both VRM0 and VRM1-style keys so avatars with different mappings still respond.
EXPRESSION_PRESETS = {
    "neutral": [],
    "happy": [("Joy", 1.0), ("happy", 1.0)],
    "anger": [("Angry", 1.0), ("angry", 1.0)],
    "sad": [("Sorrow", 1.0), ("sad", 1.0)],
    "fun": [("Fun", 1.0), ("relaxed", 1.0)],
    "surprise": [("Surprised", 1.0), ("surprised", 1.0)],
}

BLEND_KEYS_TO_CLEAR = [
    "Neutral",
    "neutral",
    "Joy",
    "happy",
    "Angry",
    "angry",
    "Sorrow",
    "sad",
    "Fun",
    "relaxed",
    "Surprised",
    "surprised",
    "Blink",
    "Blink_L",
    "Blink_R",
    "blink",
    "blinkLeft",
    "blinkRight",
]

BLINK_KEYS = ["Blink", "Blink_L", "Blink_R", "blink", "blinkLeft", "blinkRight"]


def compute_idle_pose(t: float, strength: float):
    s = float(strength)
    pitch = math.sin(t * 1.10) * 0.10 * s
    yaw = math.sin(t * 0.75) * 0.14 * s
    roll = math.sin(t * 0.50) * 0.06 * s
    y_root = math.sin(t * 1.30) * 0.01 * s
    return y_root, pitch, yaw, roll


def compute_blink(t: float):
    period = 3.2
    width = 0.11
    phase = t % period
    if phase < width:
        x = phase / width
        return 1.0 - abs(x * 2.0 - 1.0)
    return 0.0


def apply_expression_and_blink(
    vmc: VMCSender,
    expr: str,
    intensity: float,
    blink: float,
    custom_expr_key: str | None = None,
    state: dict[str, float] | None = None,
    alpha: float = 1.0,
):
    if state is None:
        state = {k: 0.0 for k in BLEND_KEYS_TO_CLEAR}

    target = {k: 0.0 for k in BLEND_KEYS_TO_CLEAR}

    if custom_expr_key:
        target[custom_expr_key] = clamp01(intensity)
    else:
        for name, value in EXPRESSION_PRESETS.get(expr, []):
            target[name] = float(value) * clamp01(intensity)

    b = clamp01(blink)
    for key in BLINK_KEYS:
        target[key] = b

    a = clamp01(alpha)
    for key in BLEND_KEYS_TO_CLEAR:
        prev = state.get(key, 0.0)
        nxt = prev + (target.get(key, 0.0) - prev) * a
        state[key] = nxt
        vmc.blend(key, nxt)

    vmc.apply()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39540, help="VSeeFace VMC receiver port")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--expr", choices=list(EXPRESSION_PRESETS.keys()), default="happy")
    parser.add_argument("--expr-key", default="", help="Exact blendshape key name to drive directly")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--idle-strength", type=float, default=1.0)
    parser.add_argument("--head-bone", default="Head")
    parser.add_argument("--send-tracker", action="store_true", help="Also send /VMC/Ext/Tra/Pos")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--expr-smooth", type=float, default=0.25, help="Seconds to ease expressions/blinks")
    parser.add_argument("--pose-smooth", type=float, default=0.20, help="Seconds to ease head/root pose")

    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--listen-ip", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=39541)
    parser.add_argument("--selftest", action="store_true")

    # Mirror outgoing packets to the local listener to inspect packets from this script.
    parser.add_argument("--mirror", action="store_true")

    args = parser.parse_args()

    if args.listen:
        start_listener(args.listen_ip, args.listen_port)

    if args.selftest:
        SimpleUDPClient("127.0.0.1", args.listen_port).send_message("/SELFTEST", ["ok", time.time()])
        log("[SELFTEST] sent")

    mirror_host = "127.0.0.1" if args.mirror else None
    mirror_port = args.listen_port if args.mirror else None

    vmc = VMCSender(args.host, args.port, mirror_host=mirror_host, mirror_port=mirror_port)
    vmc.ok()

    custom_expr_key = args.expr_key.strip() or None
    log(
        f"[SEND] -> {args.host}:{args.port} | expr={args.expr} | expr_key={custom_expr_key or '-'} "
        f"| intensity={args.intensity} | idle_strength={args.idle_strength} | tracker={args.send_tracker} "
        f"| mirror={args.mirror}"
    )

    dt = 1.0 / max(1, args.fps)
    start = time.time()
    next_ok = start + 2.0
    next_log = start + 1.0
    last_time = start
    blend_state: dict[str, float] = {k: 0.0 for k in BLEND_KEYS_TO_CLEAR}
    smoothed_pitch = 0.0
    smoothed_yaw = 0.0
    smoothed_roll = 0.0
    smoothed_root_y = 0.0

    while True:
        now = time.time()
        t = now - start
        frame_dt = max(1e-6, now - last_time)
        last_time = now

        if now >= next_ok:
            vmc.ok()
            next_ok = now + 2.0

        vmc.tick()

        root_y, pitch, yaw, roll = compute_idle_pose(t, args.idle_strength)
        pose_alpha = 1.0 - math.exp(-frame_dt / max(1e-3, args.pose_smooth))
        smoothed_root_y += (root_y - smoothed_root_y) * pose_alpha
        smoothed_pitch += (pitch - smoothed_pitch) * pose_alpha
        smoothed_yaw += (yaw - smoothed_yaw) * pose_alpha
        smoothed_roll += (roll - smoothed_roll) * pose_alpha
        head_q = quat_from_euler_xyz(smoothed_pitch, smoothed_yaw, smoothed_roll)
        vmc.root("root", 0.0, smoothed_root_y, 0.0, (0.0, 0.0, 0.0, 1.0))
        vmc.bone(args.head_bone, 0.0, 0.0, 0.0, head_q)

        if args.send_tracker:
            vmc.tracker(args.head_bone, 0.0, 0.0, 0.0, head_q)

        expr_alpha = 1.0 - math.exp(-frame_dt / max(1e-3, args.expr_smooth))
        blend_state = apply_expression_and_blink(
            vmc=vmc,
            expr=args.expr,
            intensity=args.intensity,
            blink=compute_blink(t),
            custom_expr_key=custom_expr_key,
            state=blend_state,
            alpha=expr_alpha,
        )

        if args.verbose and now >= next_log:
            log(
                f"[LIVE] t={t:6.2f}s rootY={root_y:+.4f} "
                f"expr={custom_expr_key or args.expr} blink={compute_blink(t):.2f}"
            )
            next_log = now + 1.0

        time.sleep(dt)


if __name__ == "__main__":
    main()
