import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FINGER_JOINTS = {
    "Police": [1, 2, 3, 4],
    "Aratator": [5, 6, 7, 8],
    "Mijlociu": [9, 10, 11, 12],
    "Inelar": [13, 14, 15, 16],
    "Mic": [17, 18, 19, 20],
}
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

HAND_MASS_KG = 0.45
PALM_REAL_M = 0.10


def hand_position_meters(lms, w, h):
    wx, wy = lms[0].x * w, lms[0].y * h
    mx, my = lms[9].x * w, lms[9].y * h
    palm_px = np.hypot(mx - wx, my - wy)
    if palm_px < 5:
        return None

    scale = PALM_REAL_M / palm_px
    cx = float(np.mean([lm.x for lm in lms])) * w * scale
    cy = float(np.mean([lm.y for lm in lms])) * h * scale
    cz = float(np.mean([lm.z for lm in lms])) * w * scale
    return np.array([cx, cy, cz])


def calc_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    cos_value = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_value)))


def count_fingers(lms, handedness):
    up = []
    if handedness == "Right":
        up.append(lms[4].x < lms[3].x)
    else:
        up.append(lms[4].x > lms[3].x)

    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        up.append(lms[tip].y < lms[pip].y)

    return up


def recognize_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    if not any(fingers):
        return "PUMN"
    if all(fingers):
        return "PALMA DESCHISA"
    if index and middle and not ring and not pinky and not thumb:
        return "VICTORIE"
    if thumb and not index and not middle and not ring and not pinky:
        return "THUMBS UP"
    if index and not middle and not ring and not pinky and not thumb:
        return "ARATARE"
    if thumb and pinky and not index and not middle and not ring:
        return "ROCK"
    if thumb and index and not middle and not ring and not pinky:
        return "PISTOL"
    return f"{sum(fingers)} DEGETE"


def draw_axes(frame, origin, size=70):
    ox, oy = origin
    cv2.arrowedLine(frame, (ox, oy), (ox + size, oy), (0, 0, 255), 2, tipLength=0.2)
    cv2.arrowedLine(frame, (ox, oy), (ox, oy - size), (0, 255, 0), 2, tipLength=0.2)
    cv2.arrowedLine(
        frame,
        (ox, oy),
        (ox - int(size * 0.6), oy + int(size * 0.6)),
        (255, 100, 0),
        2,
        tipLength=0.2,
    )
    cv2.putText(frame, "X", (ox + size + 4, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Y", (ox - 5, oy - size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Z",
        (ox - int(size * 0.6) - 15, oy + int(size * 0.6) + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 100, 0),
        2,
    )


def draw_info_panel(frame, hand_idx, label, fingers, gesture, angles, depth):
    x = 10 if hand_idx == 0 else frame.shape[1] - 290
    y = 30
    cv2.rectangle(frame, (x - 5, y - 25), (x + 280, y + 230), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 5, y - 25), (x + 280, y + 230), (200, 200, 200), 1)

    cv2.putText(frame, f"Mana: {label}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Gest: {gesture}", (x, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"Degete ridicate: {sum(fingers)}",
        (x, y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Adancime(Z): {depth:+.2f}",
        (x, y + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 100),
        1,
    )

    finger_names = ["Police", "Aratator", "Mijlociu", "Inelar", "Mic"]
    for i, (name, is_up, angle) in enumerate(zip(finger_names, fingers, angles)):
        color = (0, 255, 0) if is_up else (120, 120, 120)
        state = "SUS" if is_up else "JOS"
        cv2.putText(
            frame,
            f"{name:9s} {angle:5.1f}deg [{state}]",
            (x, y + 105 + i * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


class EnergyTracker:
    def __init__(self):
        self.recording = False
        self.t0 = None
        self.prev_t = None
        self.prev_pos = {}
        self.last_total_ke = 0.0
        self.peak_ke = 0.0
        self.work_j = 0.0
        self.cur_speed = 0.0
        self.summary = None
        self.samples = []

    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

    def start(self):
        self.recording = True
        self.t0 = time.time()
        self.prev_t = None
        self.prev_pos = {}
        self.last_total_ke = 0.0
        self.peak_ke = 0.0
        self.work_j = 0.0
        self.cur_speed = 0.0
        self.summary = None
        self.samples = []

    def stop(self):
        duration = time.time() - self.t0 if self.t0 else 0.0
        csv_path, png_path = self._export(duration) if self.samples else (None, None)
        self.summary = {
            "work_j": self.work_j,
            "duration": duration,
            "peak_ke_j": self.peak_ke,
            "avg_power_w": (self.work_j / duration) if duration > 0 else 0.0,
            "csv": csv_path,
            "png": png_path,
            "frames": len(self.samples),
        }
        self.recording = False

    def _export(self, duration):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "recordings")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(out_dir, f"{ts}.csv")
        png_path = os.path.join(out_dir, f"{ts}.png")

        fields = [
            "t_sec",
            "num_hands",
            "speed_max_m_s",
            "ke_total_j",
            "work_cumulative_j",
            "x_m",
            "y_m",
            "z_m",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("sep=;\n")
            writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(sample)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"[Salvat CSV] {csv_path}")
            return csv_path, None

        times = [s["t_sec"] for s in self.samples]
        ke_values = [s["ke_total_j"] for s in self.samples]
        speed_values = [s["speed_max_m_s"] for s in self.samples]
        work_values = [s["work_cumulative_j"] for s in self.samples]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        title = (
            f"Sesiune {ts} | durata {duration:.2f} s | "
            f"E_k varf: {self.peak_ke:.4f} J | lucru: {self.work_j:.4f} J"
        )
        fig.suptitle(title, fontsize=11)

        axes[0].plot(times, speed_values, color="tab:orange", linewidth=1.5)
        axes[0].set_ylabel("Viteza (m/s)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, ke_values, color="tab:green", linewidth=1.5)
        axes[1].set_ylabel("Energie cinetica (J)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(times, work_values, color="tab:blue", linewidth=1.5)
        axes[2].set_ylabel("Lucru mecanic cumulat (J)")
        axes[2].set_xlabel("Timp (s)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(png_path, dpi=120)
        plt.close(fig)

        print(f"[Salvat CSV] {csv_path}")
        print(f"[Salvat PNG] {png_path}")
        return csv_path, png_path

    def update(self, hand_positions, now):
        if self.prev_t is None:
            self.prev_t = now
            self.prev_pos = dict(hand_positions)
            return

        dt = now - self.prev_t
        if dt <= 1e-4:
            return

        total_ke = 0.0
        max_v = 0.0
        for idx, pos in hand_positions.items():
            if idx in self.prev_pos:
                velocity = np.linalg.norm(pos - self.prev_pos[idx]) / dt
                total_ke += 0.5 * HAND_MASS_KG * velocity * velocity
                max_v = max(max_v, velocity)

        if self.recording:
            self.work_j += abs(total_ke - self.last_total_ke)
            self.peak_ke = max(self.peak_ke, total_ke)

            if hand_positions:
                first_pos = next(iter(hand_positions.values()))
                x_m, y_m, z_m = float(first_pos[0]), float(first_pos[1]), float(first_pos[2])
            else:
                x_m, y_m, z_m = "", "", ""

            self.samples.append(
                {
                    "t_sec": round(now - self.t0, 4),
                    "num_hands": len(hand_positions),
                    "speed_max_m_s": round(max_v, 5),
                    "ke_total_j": round(total_ke, 6),
                    "work_cumulative_j": round(self.work_j, 6),
                    "x_m": x_m if x_m == "" else round(x_m, 5),
                    "y_m": y_m if y_m == "" else round(y_m, 5),
                    "z_m": z_m if z_m == "" else round(z_m, 5),
                }
            )

        self.last_total_ke = total_ke
        self.prev_t = now
        self.prev_pos = dict(hand_positions)
        self.cur_speed = max_v


class LiveChart:
    def __init__(self, window_sec=5.0):
        self.window_sec = window_sec
        self.data = []

    def push(self, t, value):
        self.data.append((t, float(value)))
        cutoff = t - self.window_sec
        while self.data and self.data[0][0] < cutoff:
            self.data.pop(0)

    def draw(self, frame, x, y, w_box, h_box, title, color, unit):
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (180, 180, 180), 1)

        if not self.data:
            cv2.putText(
                frame,
                f"{title} (fara date)",
                (x + 6, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
            return

        values = [v for _, v in self.data]
        current_value = values[-1]
        raw_max = max(values)
        display_max = raw_max * 1.15 if raw_max > 1e-6 else 1.0
        t_end = self.data[-1][0]
        t_start = t_end - self.window_sec

        base_y = y + h_box - 8
        cv2.line(frame, (x + 4, base_y), (x + w_box - 4, base_y), (60, 60, 60), 1)

        pts = []
        for t, value in self.data:
            px = int(x + 4 + (t - t_start) / self.window_sec * (w_box - 8))
            py = int(base_y - (value / display_max) * (h_box - 28))
            pts.append((px, py))

        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"{title} acum {current_value:.3f} max {raw_max:.3f} {unit}",
            (x + 6, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )


def draw_energy_overlay(frame, tracker):
    w = frame.shape[1]
    cx = w // 2

    if tracker.recording:
        elapsed = time.time() - tracker.t0
        blink = int(elapsed * 2) % 2 == 0
        if blink:
            cv2.circle(frame, (cx - 90, 30), 9, (0, 0, 255), -1)

        cv2.putText(frame, "REC", (cx - 75, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        lines = [
            f"Timp: {elapsed:5.2f} s",
            f"E_k acum: {tracker.last_total_ke:6.3f} J",
            f"E_k varf: {tracker.peak_ke:6.3f} J",
            f"Lucru L: {tracker.work_j:6.3f} J",
            f"Viteza: {tracker.cur_speed:5.2f} m/s",
        ]
        y = 60
        cv2.rectangle(frame, (cx - 130, y - 20), (cx + 130, y + len(lines) * 22), (0, 0, 0), -1)
        cv2.rectangle(frame, (cx - 130, y - 20), (cx + 130, y + len(lines) * 22), (0, 0, 255), 1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (cx - 120, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    elif tracker.summary is not None:
        summary = tracker.summary
        lines = [
            "REZULTAT DE INREGISTRARE",
            f"Durata: {summary['duration']:6.2f} s  ({summary.get('frames', 0)} frame-uri)",
            f"E_k de varf: {summary['peak_ke_j']:7.4f} J",
            f"Lucru mecanic: {summary['work_j']:7.4f} J",
            f"Putere medie: {summary['avg_power_w']:7.4f} W",
        ]
        if summary.get("csv"):
            lines.append(f"CSV: {os.path.basename(summary['csv'])}")
        if summary.get("png"):
            lines.append(f"PNG: {os.path.basename(summary['png'])}")

        y = 30
        box_h = len(lines) * 22 + 6
        cv2.rectangle(frame, (cx - 220, y - 20), (cx + 220, y + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (cx - 220, y - 20), (cx + 220, y + box_h), (0, 255, 0), 2)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (cx - 210, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Nu pot deschide camera.")
        return

    prev_time = time.time()
    show_skeleton = True
    show_angles = True
    energy = EnergyTracker()
    speed_chart = LiveChart(window_sec=5.0)
    ke_chart = LiveChart(window_sec=5.0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_positions = {}

        if results.multi_hand_landmarks:
            for idx, (hand_lm, hand_info) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                label = hand_info.classification[0].label
                lms = hand_lm.landmark
                pos = hand_position_meters(lms, w, h)
                if pos is not None:
                    hand_positions[idx] = pos

                if show_skeleton:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                wrist = (int(lms[0].x * w), int(lms[0].y * h))
                draw_axes(frame, wrist)

                angles = []
                for joint_ids in FINGER_JOINTS.values():
                    angles.append(calc_angle(lms[joint_ids[0]], lms[joint_ids[1]], lms[joint_ids[2]]))

                if show_angles:
                    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
                        px, py = int(lms[tip].x * w), int(lms[tip].y * h)
                        angle = calc_angle(lms[pip - 1], lms[pip], lms[tip])
                        cv2.putText(
                            frame,
                            f"{angle:.0f}",
                            (px + 8, py),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                z = lms[8].z
                tip_x, tip_y = int(lms[8].x * w), int(lms[8].y * h)
                radius = int(np.clip(25 - z * 250, 6, 50))
                cv2.circle(frame, (tip_x, tip_y), radius, (255, 0, 255), 2)

                fingers = count_fingers(lms, label)
                gesture = recognize_gesture(fingers)
                draw_info_panel(frame, idx, label, fingers, gesture, angles, z)

        energy.update(hand_positions, now)
        speed_chart.push(now, energy.cur_speed)
        ke_chart.push(now, energy.last_total_ke)
        draw_energy_overlay(frame, energy)

        chart_h = 70
        chart_y = h - chart_h - 28
        chart_w = (w - 60) // 2
        speed_chart.draw(frame, 20, chart_y, chart_w, chart_h, "Viteza", (0, 255, 255), "m/s")
        ke_chart.draw(frame, w - 40 - chart_w, chart_y, chart_w, chart_h, "Energie cinetica", (0, 255, 0), "J")

        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:5.1f}", (w - 140, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(
            frame,
            "Q=iesire S=schelet A=unghiuri R=record energie P=screenshot",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Hand Tracker - Schelet + Unghiuri + Axe + Gesturi", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            show_skeleton = not show_skeleton
        if key == ord("a"):
            show_angles = not show_angles
        if key == ord("r"):
            energy.toggle()
        if key == ord("p"):
            cv2.imwrite("screenshot.png", frame)
            print("Screenshot salvat ca screenshot.png")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
