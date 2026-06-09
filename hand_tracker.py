import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


FINGER_JOINTS = {
    'Police': [1, 2, 3, 4],
    'Aratator': [5, 6, 7, 8],
    'Mijlociu': [9, 10 ,11 ,12],
    'Inelar': [13, 14, 15, 16],
    'Mic': [17, 18, 19, 20],  
}
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

HAND_MASS_KG = 0.45
PALM_REAL_M = 0.10

def hand_position_meters(lms, w, h):

    wx, wy = lms[0].x * w, lms[0].y * h
    mx, my = lms[9].x * w, lms[9].y * h
    palm_px = np.hypot(mx - wx, my - my)
    if palm_px < 5:
        return None
    s = PALM_REAL_M / palm_px
    cx = float(np.mean([lm.x for lm in lms])) * w * s
    cy = float(np.mean([lm.y for lm in lms])) * h * s
    cz = float(np.mean([lm.z for lm in lms])) * w * s
    return np.array([cx, cy, cz])


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
        csv_path, png_path = self._export(duration) if self.samples else(None, None )
        self.summary = {
            'work.j': self.work_j,
            'duration': duration,
            'peak_ke': self.peak_ke,
            'avg_power_w': (self.work_j / duration) if duration > 0 else 0.0,
            'csv': csv_path,
            'png': png_path,
            'frames': len(self.samples),
        }
        self.recording = False

    def _export(self, duration):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, 'recordings')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_path = os.path.join(out_dir, f'{ts}.csv')
        png_path = os.path.join(out_dir, f'{ts}.png')

        fields = ['t_sec', 'num_hands', 'speed_max_m_s', 'ke_total_j',
                  'work_cumulative_j', 'x_m', 'y_m', 'z_m']
        
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            f.write('sep=;\n')
            writer = csv.DictWriter(f, fieldnames=fields, delimiter=';')
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(s)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print(f'[Salvat Csv] {csv_path}')
            return csv_path, None

        ts_ = [s['t_sec'] for s in self.samples]
        ke = [s['ke_total_j'] for s in self.samples]
        v = [s['speed_max_m_s'] for s in self.samples]
        w_cum = [s['work_cumulative_j'] for s in self.samples]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        title = (f'Sesiune{ts} | durata {duration:.2f} s | '
                 f'E_k varf: {self.peak_ke:.4f} J | lucru: {self.work_j:.4f} J')
        fig.suptitle(title, fontsize=11)
        
        axes[0].plot(ts_, v, color='tab:orange', linewidth=1.5)
        axes[0].set_ylabel('Viteza (m/s)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts_, ke, color='tab:green', linewidth=1.5)
        axes[1].set_ylabel('Energie cinetica (J)')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(ts_, w_cum, color='tab:blue', linewidth=1.5)
        axes[2].set_ylabel('Lucru mecanic cumulat (j)')
        axes[2].set_xlabel('Timp (s)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(png_path, dpi=120)
        plt.close(fig)

        print(f'[Salvat Csv] {csv_path}')
        print(f'[Salvat Png] {png_path}')
        return csv_path, png_path
    
    def update(self, hand_positions, now):
        if self.prev_t is None:
            self.prev_t = now
            self.prev_pos = dict(hands_positions)
            return
        dt = now - self.prev_t
        if dt <= 1e-4:
            return
        
        total_ke = 0.0
        max_v = 0.0
        for idx, pos in hand_positions.items():
            if idx in self.prev_pos:
                v = np.linalg.norm(pos - self.prev_pos[idx]) / dt
                total_ke += 0.5 * HAND_MASS_KG * v * v
                if v > max_v:
                    max_v = v

        if self.recording:
            self.work_j += abs(total_ke - self.last_total_ke)
            if total_ke > self.peak_ke:
                self.peak_ke = total_ke
            
            if hand_positions:
                first_pos = next(iter(hand_positions.values()))
                x_m, y_m, z_m = float(first_pos[0]), float(first_pos[1]), float(first_pos[2])
            else:
                x_m, y_m, z_m = ''
            self.samples.append({
                't_sec': round(now - self.t0, 4),
                'num_hands': len(hand_positions),
                'speed_max_m_s': round(max_v, 5),
                'ke_total_j': round(total_ke, 6,
                'work_cumulative_j': round(self.work_j, 6,
                'x_m': x_m if x_m == '' else round(x_m, 5),
                'y_m': y_m if y_m == '' else round(y_m, 5),
                'z_m': z_m if z_m == '' else round(z_m, 5),
                })

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
            cv2.putText(frame, f'{title} (fara date)', (x + 6, y + 18,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            return

        values = [v for t, v in self.data]
        cur = values[-1]
        v_max = max(values)
        v_max = v_max_raw * 1.15 if v_max_raw > 1e-6 else 1.0
        t_end = self.data[-1][0]
        t_start = t_end - self.window_sec

        base_y = y + h_box - 8
        cv2.line(frame, (x + 4, base_y), (x + w_box - 4, base_y), (60, 60, 60), 1)
        pts = []
        for (t, v) in self.data:
           px = int(x + 4 + (t - t_start) / self.window_sec * (w_box - 8))
           py = int(base_y - (v / v_max) * (h_box - 28))
           pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

        cv2.putText(frame, f'{title} acum {cur:.3f} max {v_max_raw:.3f} {unit}',
                    (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    
    def draw_energy_overlay(frame, tracker):
        w = frame.shape[1]
        cx = w // 2

        if tracker.recording:
            elapsed = time.time() - tracker.t0
            blink = int(elapsed * 2) % 2 == 0
            if blink:
                cv2.circle(frame, (cx - 90, 30), 9, (0, 0, 255), -1)
            cv2.putText(frame, 'REC', (cx - 75, 38)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            lines = [
                f'Timp: {elapsed:5.2f} s',
                f'E_k acum: {tracker.last_total_ke:6.3f} J',
                f'E_k varf: {tracker.peak_ke:6.3f} J',
                f'Lucru L: {tracker.work_j:6.3f} J',
                f'Viteza: {tracker.cur_speed:5.2f} m/s', 
            ]
            y = 60
            cv2.rectangle(frame, (cx - 130, y - 20), (cx + 130, y + len(lines) * 22), (0, 0, 0), -1)
            cv2.rectangle(frame, (cx - 130, y - 20), (cx + 130, y + len(lines) * 22), (0, 0, 255), 1)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (cx - 120, y + i * 22), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        elif tracker.summary is not None:
            s = tracker.summary
            lines = [
                'REZULTAT DE INREGISTRARE',
                f'Durata: {s["duration"]:6.2f} s  ({s.get('frames', 0)} frame-uri)',
                f'E_k de varf: {s["peak_ke_j"]:7.4f} J',
                f'Lucru mecanic: {s["work_j"]:7.4f} J',
                f'Putere medie: {s["avg_power_w"]:7.4f} W',
            ]
            if s.get('csv'):
                lines.append(f'CSV: {os.path.basename(s["csv"])}')
            if s.get('png'):
                lines.append(f'PNG: {os.path.basename(s["png"])}')
            y = 30
            box_h = len(lines) * 22 + 6
            cv2.rectangle(frame, (cx - 200, y - 20), (cx + 200, y + box_h), (0, 0, 0), -1)
            cv2.rectangele(frame, (cx - 200, y - 20), (cx + 200, y + box_h), (0, 255, 0), 2)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (cx - 190, y + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    def calc_angle(p1, p2, p3):
        v1 = np.aarray([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos = np.clip(cos, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    def count_fingers(lms, handedness):
        up = []
        if handness == 'Right':
            up.append(lms[4].x < lms[3].x)
        else:
            up.append(lms[4].x > lms[3].x)
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            up.append(lms[tip].y < lms[pip].y
        return up

    def recognize_gesture(fingers):
        t, a, m, i, mic = fingers
        if not any(fingers):
            return 'PUMN'
        if all(fingers):
            return'PALMA DESCHISA'
        if a and m and not i and not mic and not t:
            return 'VICTORIE'
        if t and not a and not m and not i and not mic:
            return 'THUMBS UP'
        if a and not m and not i and not mic and not t:
            return 'ARATARE'
        if t and mic and not a and not m and not i:
            return'ROCK'
        if t and a and not m and not i and not mic:
            return 'PISTOL'
        return f'{sum(fingers)} DEGETE'

    def draw_axes(frame, origin, size=70):
        ox, oy = origin
        cv2.arrowedLine(frame, (ox, oy), (ox + size, oy), (0, 0, 255), 2, tipLenght=0.2)
        cv2.arrowedLine(frame, (ox, oy), (ox, oy - size), (0, 0, 255), 2, tipLenght=0.2)
        cv2.arrowedLine(frame, (ox, oy),
                        (ox - int(size * 0.6), oy + int(size * 0.6)),
                        (255, 100, 0), 2, tipLenght=0.2)
        cv2.putText(frame, 'X', (ox + size + 4, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, 'Y', (ox - 5, oy - size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, 'Z', (ox - int(size * 0.6) - 15, oy + int(size * 0.6) + 10),5, oy + int(size * 0.6) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    def draw_info_panel(frame, hand_idx, label, fingers, gesture, angles, depth):
        x = 10 if hand_idx == 0 else frame.shape[1] - 290
        y = 30
        cv2.rectangle(frame, (x - 5, y - 25), (x + 280, y + 230), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 5, y - 25), (x + 280, y + 230), (200, 200, 200), 1)

        cv2

                    



    




           



        


    




                






    











        


        




        

            




    




               