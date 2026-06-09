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





    











        


        




        

            




    




               