"""
pipeline.py - Moteur d'analyse sprint : pose -> stabilisation caméra -> mètres -> appuis -> métriques.

Chaîne :
 1. Lecture vidéo image par image (réduite à 960 px de large max).
 2. Pose (RTMPose via rtmlib) sur l'athlète.
 3. Stabilisation : le mouvement de la caméra est estimé sur l'ARRIÈRE-PLAN (flux optique
    + transformation de similarité robuste RANSAC), athlète masqué. Tout est ramené
    au repère de la 1re image (translation, rotation ET zoom compensés).
 4. Échelle : longueur des jambes mesurée / (0,491 x taille de l'athlète).
 5. Appuis au sol dans le repère monde + métriques avec incertitudes.
"""
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

MAX_WIDTH = 960
LEG_RATIO = 0.491          # (cuisse + tibia) / taille, Dempster
SCALE_UNCERTAINTY = 0.06   # incertitude relative sur l'échelle (taille -> longueur des jambes)
CONTACT_MODEL_BAND_S = 0.035


# =====================================================================
# 1. Signal : trous, lissage
# =====================================================================

def fill_gaps(sig, valid, max_gap=6):
    sig = np.asarray(sig, float).copy()
    valid = np.asarray(valid, bool)
    sig[~valid] = np.nan
    n = len(sig)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if (j - i) <= max_gap and i > 0 and j < n:
            sig[i:j] = np.interp(np.arange(i, j), [i - 1, j], [sig[i - 1], sig[j]])
        i = j
    return sig


def butter_smooth(sig, fps, cutoff=15.0, order=4):
    sig = np.asarray(sig, float)
    out = np.full_like(sig, np.nan)
    cutoff = min(cutoff, fps / 2.0 - 1.0)
    b, a = butter(order, cutoff / (fps / 2.0))
    ok = ~np.isnan(sig)
    i, n = 0, len(sig)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        seg = sig[i:j]
        out[i:j] = filtfilt(b, a, seg) if len(seg) > 3 * max(len(a), len(b)) else seg
        i = j
    return out


def derivative(sig, fps):
    return np.gradient(sig, 1.0 / fps)


# =====================================================================
# 2. Appuis au sol
# =====================================================================

@dataclass
class Contact:
    foot: str
    t_on: float
    t_off: float
    x: float

    @property
    def duration(self):
        return self.t_off - self.t_on


def _cross(t0, t1, v0, v1, thr):
    if v1 == v0:
        return t0
    return t0 + (thr - v0) / (v1 - v0) * (t1 - t0)


def detect_contacts(x, h, fps, foot="L", alpha=0.07, k_noise=3.0, h_tol=0.06,
                    min_dur=0.04, max_dur=0.35):
    """Contact = pied lent ET bas (repère sol, en mètres). Seuil de vitesse adaptatif."""
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    n = len(x)
    t = np.arange(n) / fps
    speed = np.hypot(derivative(x, fps), derivative(h, fps))
    ok = ~np.isnan(speed)
    if ok.sum() < 10:
        return []
    vp = np.nanpercentile(speed, 90)
    low = np.sort(speed[ok])[:max(5, ok.sum() // 5)]
    thr = max(alpha * vp, k_noise * np.median(low), 1e-6)
    h_base = np.nanpercentile(h, 5)
    score = np.maximum(speed / thr, (h - h_base) / h_tol)
    mask = (score < 1.0) & ok
    res, i = [], 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if i > 0 and j < n:
            t_on = _cross(t[i - 1], t[i], score[i - 1], score[i], 1.0)
            t_off = _cross(t[j - 1], t[j], score[j - 1], score[j], 1.0)
            if min_dur <= (t_off - t_on) <= max_dur:
                res.append(Contact(foot, float(t_on), float(t_off), float(np.nanmedian(x[i:j]))))
        i = j
    return res


def sprint_metrics(contacts, fps, scale_unc=SCALE_UNCERTAINTY):
    cs = sorted(contacts, key=lambda c: c.t_on)
    if len(cs) < 3:
        return {"erreur": "Pas assez d'appuis détectés (il en faut au moins 3)."}
    ct_unc = float(np.hypot(1.0 / fps, CONTACT_MODEL_BAND_S))
    td = np.array([c.t_on for c in cs])
    to = np.array([c.t_off for c in cs])
    xs = np.array([c.x for c in cs])
    step_t = np.diff(td)
    keep = np.abs(step_t - np.median(step_t)) < 0.3 * np.median(step_t)
    if keep.sum() < 2:
        return {"erreur": "Foulées trop irrégulières pour calculer des métriques."}
    st = step_t[keep]
    flight = (td[1:] - to[:-1])[keep]
    sl = np.diff(xs)[keep]
    ct = np.array([c.duration for c in cs])
    v = float(sl.sum() / st.sum())
    return {
        "appuis_detectes": len(cs),
        "foulees_rejetees": int((~keep).sum()),
        "temps_contact_ms": {"valeur": round(float(1000 * ct.mean()), 1), "incertitude": round(1000 * ct_unc, 1)},
        "temps_vol_ms": {"valeur": round(float(1000 * flight.mean()), 1), "incertitude": round(1000 * ct_unc, 1)},
        "frequence_foulee_hz": {"valeur": round(float((1 / st).mean()), 2), "incertitude": round(float((1 / st).std()), 2)},
        "longueur_foulee_m": {"valeur": round(float(sl.mean()), 2),
                              "incertitude": round(float(np.hypot(sl.std(), sl.mean() * scale_unc)), 2)},
        "vitesse_moyenne_ms": {"valeur": round(v, 2), "incertitude": round(v * scale_unc, 2)},
    }


# =====================================================================
# 3. Stabilisation caméra (arrière-plan)
# =====================================================================

def _background_mask(shape, box):
    m = np.full(shape, 255, np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        pad = 0.2 * max(x1 - x0, y1 - y0)
        m[max(0, int(y0 - pad)):int(y1 + pad), max(0, int(x0 - pad)):int(x1 + pad)] = 0
    return m


def camera_step(prev_gray, gray, box):
    """Transformation de similarité image courante -> image précédente (fond uniquement)."""
    mask = _background_mask(prev_gray.shape, box)
    p0 = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 12:
        return None, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(31, 31), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 12:
        return None, 0.0
    M, inl = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if M is None:
        return None, 0.0
    return M, float(inl.mean())


# =====================================================================
# 4. Pose (RTMPose via rtmlib)
# =====================================================================

_POSE = {"tracker": None, "lock": threading.Lock()}


def _get_tracker():
    with _POSE["lock"]:
        if _POSE["tracker"] is None:
            try:
                from rtmlib import PoseTracker, BodyWithFeet as Solution
            except ImportError:
                from rtmlib import PoseTracker, Wholebody as Solution
            _POSE["tracker"] = PoseTracker(Solution, det_frequency=5, mode="lightweight",
                                           backend="onnxruntime", device="cpu", tracking=False)
        return _POSE["tracker"]


def rtmlib_pose(frame_bgr, prev_center):
    """Retourne (keypoints (K,2), scores (K,), bbox) de l'athlète, ou (None, None, None)."""
    kpts, scores = _get_tracker()(frame_bgr)
    kpts, scores = np.asarray(kpts), np.asarray(scores)
    if kpts.ndim != 3 or len(kpts) == 0:
        return None, None, None
    best, best_val = 0, -1e18
    for i in range(len(kpts)):
        good = scores[i] > 0.3
        if good.sum() < 5:
            continue
        pts = kpts[i][good]
        h = pts[:, 1].max() - pts[:, 1].min()
        c = pts.mean(axis=0)
        val = h if prev_center is None else -np.hypot(*(c - prev_center))
        if val > best_val:
            best, best_val = i, val
    good = scores[best] > 0.3
    if good.sum() < 5:
        return None, None, None
    pts = kpts[best][good]
    box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    return kpts[best], scores[best], box


def keypoint_map(K):
    """Indices des points utiles selon le modèle (Halpe-26 ou Wholebody-133)."""
    if K == 26:
        return dict(hip=(11, 12), knee=(13, 14), ankle=(15, 16), toe=(20, 21), heel=(24, 25))
    if K == 133:
        return dict(hip=(11, 12), knee=(13, 14), ankle=(15, 16), toe=(17, 20), heel=(19, 22))
    raise ValueError(f"Modèle de pose non pris en charge ({K} points)")


# =====================================================================
# 5. Pipeline complet
# =====================================================================

def run_pipeline(video_path, height_m=1.75, fps_override=None, max_seconds=8.0,
                 progress=lambda frac: None, pose_fn=None, out_video=None, playback="slow", pauses=True,
                 camera="mobile"):
    """pose_fn(frame_index, frame_bgr, prev_center) -> (kpts, scores, box) ; par défaut RTMPose.
    progress(frac) avec frac entre 0 et 1. Si out_video est donné, écrit la vidéo annotée."""
    pose_fn = pose_fn or (lambda i, f, c: rtmlib_pose(f, c))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Vidéo illisible.")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps_override) if fps_override else float(file_fps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    n_max = int(max_seconds * fps)
    n_target = min(total, n_max) if total else n_max

    kp_list, sc_list, T_list, inl_list = [], [], [], []
    T = np.eye(3)
    prev_gray, prev_box, prev_center = None, None, None
    K = None
    i = 0
    while i < n_max:
        ok, frame = cap.read()
        if not ok:
            break
        h0, w0 = frame.shape[:2]
        if w0 > MAX_WIDTH:
            s = MAX_WIDTH / w0
            frame = cv2.resize(frame, (MAX_WIDTH, int(h0 * s)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -- caméra
        if prev_gray is not None and camera == "mobile":
            M, inl = camera_step(prev_gray, gray, prev_box)
            if M is not None:
                T = T @ np.vstack([M, [0, 0, 1]])
            inl_list.append(inl)
        T_list.append(T.copy())

        # -- pose
        kp, sc, box = pose_fn(i, frame, prev_center)
        if kp is None:
            kp_list.append(None)
            sc_list.append(None)
        else:
            K = len(kp)
            kp_list.append(np.asarray(kp, float))
            sc_list.append(np.asarray(sc, float))
            prev_box = box
            prev_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        prev_gray = gray
        i += 1
        progress(0.7 * i / max(n_target, 1) if out_video else i / max(n_target, 1))
    cap.release()

    n = len(T_list)
    if n < 30 or K is None:
        raise ValueError("Trop peu d'images exploitables (athlète non détecté ?).")
    result, ctx = analyse(kp_list, sc_list, T_list, inl_list, fps, file_fps, height_m, K, camera)
    result["video_annotee"] = False
    if out_video:
        try:
            render_video(video_path, ctx, out_video, playback, lambda f: progress(0.7 + 0.3 * f), pauses)
            result["video_annotee"] = True
        except Exception as e:  # noqa
            result["avertissements"].append(f"Vidéo annotée non générée : {e}")
    return result


# =====================================================================
# 5b. Garde-fous physiques : phases, vitesse du bassin, validation des appuis
# =====================================================================

V_RUN = 2.0                      # m/s : vitesse minimale du bassin pour parler de "course"
V_MOVE = 0.5                     # m/s : début de mouvement (sortie des blocs)
MIN_RUN_S = 0.6                  # durée minimale d'une course
CONTACT_RANGE_S = (0.05, 0.30)   # durée d'appui plausible
STEP_TIME_RANGE_S = (0.15, 0.45) # temps entre deux appuis (pieds alternés)
STEP_LEN_RANGE_M = (0.4, 3.5)    # longueur de foulée plausible
SPEED_CONSIST = (0.65, 1.5)      # (longueur/temps de foulée) / vitesse du bassin
PHASE_NAMES = {0: "blocs / immobile", 1: "départ", 2: "course", 3: "fin de course"}


def windowed_speed(x_m, fps, half=0.1):
    """Vitesse |dx/dt| sur une fenêtre de +-half s : insensible au tremblement de la pose."""
    x_m = np.asarray(x_m, float)
    n = len(x_m)
    h = max(1, int(round(half * fps)))
    a = np.clip(np.arange(n) - h, 0, n - 1)
    b = np.clip(np.arange(n) + h, 0, n - 1)
    dt = (b - a) / fps
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > a, np.abs(x_m[b] - x_m[a]) / dt, np.nan)


def find_run(speed, fps):
    """Plus longue période où le bassin dépasse V_RUN. Retourne (début_mouvement, début_course, fin_course) ou None."""
    n = len(speed)
    sp = np.nan_to_num(np.asarray(speed, float), nan=0.0)
    ok = sp >= V_RUN
    gap = int(0.1 * fps)
    idx = np.where(ok)[0]
    filled = ok.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            filled[a:b] = True
    best, i = None, 0
    while i < n:
        if not filled[i]:
            i += 1
            continue
        j = i
        while j < n and filled[j]:
            j += 1
        if (j - i) / fps >= MIN_RUN_S and (best is None or (j - i) > (best[1] - best[0] + 1)):
            best = (i, j - 1)
        i = j
    if best is None:
        return None
    i0, i1 = best
    m = i0
    while m > 0 and sp[m - 1] >= V_MOVE:
        m -= 1
    return m, i0, i1


def phase_array(n, run):
    ph = np.zeros(n, int)
    if run is None:
        return ph
    m, i0, i1 = run
    ph[m:i0] = 1
    ph[i0:i1 + 1] = 2
    ph[i1 + 1:] = 3
    return ph


def validate_contacts(contacts, speed, fps):
    """Garde les appuis qui forment au moins une paire physiquement cohérente avec un voisin :
    pieds alternés, temps et longueur plausibles, vitesse des pieds ~ vitesse du bassin.
    Retourne (valides, rejetés[(contact, raison)])."""
    # 1) recoller les fragments d'un même appui coupé par le bruit (même pied, trou < 50 ms)
    merged = []
    for c in sorted(contacts, key=lambda c: c.t_on):
        prev = next((p for p in reversed(merged) if p.foot == c.foot), None)
        if prev is not None and c.t_on - prev.t_off < 0.05:
            prev.t_off = max(prev.t_off, c.t_off)
            prev.x = (prev.x + c.x) / 2
        else:
            merged.append(c)
    rej, cs = [], []
    for c in merged:
        if not (CONTACT_RANGE_S[0] <= c.duration <= CONTACT_RANGE_S[1]):
            rej.append((c, "durée d'appui irréaliste"))
        elif cs and cs[-1].foot == c.foot and c.t_on - cs[-1].t_on < 2 * STEP_TIME_RANGE_S[0]:
            # un même pied ne peut pas se poser deux fois en moins de 0,3 s : on garde l'appui le plus long
            if c.duration > cs[-1].duration:
                rej.append((cs[-1], "doublon du même pied"))
                cs[-1] = c
            else:
                rej.append((c, "doublon du même pied"))
        else:
            cs.append(c)
    good, why_of = set(), {}
    for a, b in zip(cs[:-1], cs[1:]):
        st, ln = b.t_on - a.t_on, b.x - a.x
        why = None
        if a.foot == b.foot:
            why = "deux appuis de suite du même pied"
        elif not (STEP_TIME_RANGE_S[0] <= st <= STEP_TIME_RANGE_S[1]):
            why = "temps entre appuis irréaliste"
        elif not (STEP_LEN_RANGE_M[0] <= ln <= STEP_LEN_RANGE_M[1]):
            why = "longueur de foulée irréaliste"
        else:
            seg = speed[int(a.t_on * fps):int(b.t_on * fps) + 1]
            hv = float(np.nanmean(seg)) if len(seg) and np.isfinite(seg).any() else float("nan")
            if not np.isfinite(hv) or hv <= 0:
                why = "bassin immobile"
            elif not (SPEED_CONSIST[0] <= (ln / st) / hv <= SPEED_CONSIST[1]):
                why = "vitesse des pieds incohérente avec celle du bassin"
        if why is None:
            good.update((id(a), id(b)))
        else:
            why_of.setdefault(id(a), why)
            why_of.setdefault(id(b), why)
    valid = [c for c in cs if id(c) in good]
    rej += [(c, why_of.get(id(c), "appui isolé")) for c in cs if id(c) not in good]
    return valid, rej


def _leg_px(kp, km, lo=0, hi=None):
    legs = []
    for side in (0, 1):
        sl = slice(lo, hi)
        hp, kn, an = (kp[sl, km[k][side], :] for k in ("hip", "knee", "ankle"))
        legs.append(np.hypot(*(hp - kn).T) + np.hypot(*(kn - an).T))
    legs = np.concatenate(legs)
    return legs[~np.isnan(legs)]


def _ds(a, step):
    return [None if not np.isfinite(v) else round(float(v), 2) for v in np.asarray(a)[::step]]


def analyse(kp_list, sc_list, T_list, inl_list, fps, file_fps, height_m, K, camera="mobile"):
    n = len(T_list)
    km = keypoint_map(K)
    kp = np.full((n, K, 2), np.nan)
    sc = np.zeros((n, K))
    for i in range(n):
        if kp_list[i] is None:
            continue
        pts = np.c_[kp_list[i], np.ones(K)] @ T_list[i][:2].T   # -> repère de la 1re image
        kp[i], sc[i] = pts, sc_list[i]
    kp[sc < 0.3] = np.nan

    def series(idx, axis):
        s = kp[:, idx, axis]
        return fill_gaps(s, ~np.isnan(s))

    # -- bassin : trajectoire (pixels du repère de référence)
    hip_raw = np.nanmean([kp[:, km["hip"][0], 0], kp[:, km["hip"][1], 0]], axis=0)
    if np.isnan(hip_raw).all():
        raise ValueError("Hanches jamais détectées.")
    hip_x = fill_gaps(hip_raw, ~np.isnan(hip_raw), max_gap=int(0.3 * fps))

    # -- échelle puis phases (2 passes : échelle globale, puis échelle sur la course seule)
    legs = _leg_px(kp, km)
    if len(legs) < 20:
        raise ValueError("Jambes trop rarement visibles pour estimer l'échelle.")
    ppm = float(np.percentile(legs, 90) / (LEG_RATIO * height_m))
    speed = windowed_speed(hip_x / ppm, fps)
    run = find_run(speed, fps)
    if run is not None:
        lw = _leg_px(kp, km, run[0], run[2] + 1)
        if len(lw) >= 20:
            ppm = float(np.percentile(lw, 90) / (LEG_RATIO * height_m))
            speed = windowed_speed(hip_x / ppm, fps)
            run = find_run(speed, fps)
    phase = phase_array(n, run)
    vmax = float(np.nanmax(speed)) if np.isfinite(speed).any() else 0.0

    # -- sens de la course
    if run is not None:
        a_, b_ = run[1], run[2]
        sign = 1.0 if np.nanmean(hip_x[max(b_ - 5, a_):b_ + 1]) >= np.nanmean(hip_x[a_:a_ + 6]) else -1.0
    else:
        v = np.where(~np.isnan(hip_x))[0]
        sign = 1.0 if hip_x[v[-1]] >= hip_x[v[0]] else -1.0

    # -- pieds en mètres (repère sol)
    lo, hi = (0, n - 1)
    if run is not None:
        pad = int(0.25 * fps)
        lo, hi = max(0, run[0] - pad), min(n - 1, run[2] + pad)
    feet = {}
    ys = []
    for side, name in ((0, "G"), (1, "D")):
        yh, yt = series(km["heel"][side], 1), series(km["toe"][side], 1)
        xh, xt = series(km["heel"][side], 0), series(km["toe"][side], 0)
        feet[name] = ((xh + xt) / 2, np.fmax(yh, yt))
        ys.append(feet[name][1][lo:hi + 1][~np.isnan(feet[name][1][lo:hi + 1])])
    y_ground = float(np.percentile(np.concatenate(ys), 95)) if sum(len(a) for a in ys) else 0.0

    # -- appuis : seulement pendant la course, puis validation physique
    candidates, cutoff = [], min(25.0, fps / 2 - 5)
    foot_h = {}
    for name, (x, y) in feet.items():
        foot_h[name] = (y_ground - y) / ppm
        if run is None:
            continue
        xm_w = butter_smooth(sign * x[lo:hi + 1] / ppm, fps, cutoff)
        hm_w = butter_smooth(foot_h[name][lo:hi + 1], fps, cutoff)
        for c in detect_contacts(xm_w, hm_w, fps, name):
            c.t_on += lo / fps
            c.t_off += lo / fps
            if run[0] / fps - 0.05 <= c.t_on <= run[2] / fps + 0.05:
                candidates.append(c)
    contacts, rejected = validate_contacts(candidates, speed, fps)
    rej_count = {}
    for _, why in rejected:
        rej_count[why] = rej_count.get(why, 0) + 1

    # -- message d'état
    inl_pct = float(np.median(inl_list)) if inl_list else 0.0
    if run is None:
        if camera == "mobile" and inl_pct < 0.5:
            msg = (f"Stabilisation de la caméra échouée (seulement {100 * inl_pct:.0f} % de points de fond fiables). "
                   "Quand la caméra suit le coureur devant un fond pauvre (piste, herbe, lignes parallèles), le mouvement "
                   f"du coureur est annulé avec celui de la caméra : bassin mesuré à {vmax:.1f} m/s au maximum. "
                   "Ce n'est pas une absence de course. Filme avec un trépied fixe et choisis « Caméra fixe », "
                   "ou place des repères nets dans le décor (plots, bandes).")
        else:
            msg = (f"Aucune course lancée détectée : le bassin n'a pas dépassé {V_RUN:.0f} m/s "
                   f"(maximum mesuré {vmax:.1f} m/s). Aucune foulée n'est comptée. Si l'athlète court vraiment, "
                   "vérifie la taille saisie, la cadence et que l'athlète est entier dans l'image.")
        metrics = {"erreur": msg}
    else:
        metrics = sprint_metrics(contacts, fps)
        if "erreur" in metrics:
            metrics = {"erreur": f"Course détectée mais moins de 3 appuis fiables "
                                 f"({len(candidates)} candidats, {len(rejected)} rejetés)."}

    # -- squelette lissé (repère 1re image) + angles
    need = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + list(km["heel"]) + list(km["toe"])
    kps = np.full((n, K, 2), np.nan)
    for j in need:
        for ax in (0, 1):
            col = kp[:, j, ax]
            kps[:, j, ax] = butter_smooth(fill_gaps(col, ~np.isnan(col), max_gap=10), fps, 15.0)

    def ang(a, b, c):
        v1, v2 = a - b, c - b
        cos = (v1 * v2).sum(1) / (np.hypot(*v1.T) * np.hypot(*v2.T) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    knee = np.stack([ang(kps[:, km["hip"][s_]], kps[:, km["knee"][s_]], kps[:, km["ankle"][s_]]) for s_ in (0, 1)], 1)
    sh = (kps[:, 5] + kps[:, 6]) / 2
    hp = (kps[:, 11] + kps[:, 12]) / 2
    d = sh - hp
    trunk = np.degrees(np.arctan2(d[:, 0] * sign, -d[:, 1]))       # + = penché vers l'avant

    if "erreur" not in metrics:
        seg = trunk[run[1]:run[2] + 1]
        if np.isfinite(seg).any():
            metrics["inclinaison_tronc_deg"] = {"valeur": round(float(np.nanmean(seg)), 1),
                                                "incertitude": round(float(np.nanstd(seg)), 1)}
        td_knee = []
        for c in contacts:
            fi, si = int(round(c.t_on * fps)), (0 if c.foot == "G" else 1)
            if 0 <= fi < n and not np.isnan(knee[fi, si]):
                td_knee.append(knee[fi, si])
        if td_knee:
            metrics["angle_genou_a_l_attaque_deg"] = {"valeur": round(float(np.mean(td_knee)), 1),
                                                      "incertitude": round(float(np.std(td_knee)), 1)}

    # -- qualité & avertissements
    pose_ok = float(np.mean([k is not None for k in kp_list]))
    foot_ok = float(np.mean(~np.isnan(kp[:, km["heel"][0], 0]) | ~np.isnan(kp[:, km["heel"][1], 0])))
    inl = float(np.median(inl_list)) if inl_list else 0.0
    warns = []
    if run is not None:
        warns.append(f"Course lancée détectée de {run[1] / fps:.2f} s à {run[2] / fps:.2f} s "
                     f"(mouvement dès {run[0] / fps:.2f} s). Les foulées ne sont comptées que dans cette période.")
    if rej_count:
        warns.append("Appuis rejetés par les contrôles physiques : " +
                     ", ".join(f"{v} × {k}" for k, v in rej_count.items()) + ".")
    if abs(fps - file_fps) > 1:
        warns.append(f"Cadence forcée à {fps:.0f} images/s (le fichier indique {file_fps:.0f}).")
    if fps < 100:
        warns.append("Cadence < 100 images/s : les temps de contact et de vol sont très peu fiables. Filme en 120-240 images/s.")
    if pose_ok < 0.9:
        warns.append(f"Athlète détecté sur seulement {100 * pose_ok:.0f} % des images.")
    if camera == "mobile" and inl < 0.5:
        warns.append("Stabilisation peu fiable (peu de fond net exploitable) : le mouvement du bassin peut être faussé.")
    warns.append("Les distances et vitesses dépendent de la taille saisie (±6 % environ). Les temps et la fréquence n'en dépendent pas.")
    warns.append("Hypothèse : caméra de profil, athlète dans un plan à peu près perpendiculaire à l'axe de la caméra.")

    unc_ms = 1000 * float(np.hypot(1.0 / fps, CONTACT_MODEL_BAND_S))
    strides = build_strides(contacts, fps, kps, knee, trunk, km, sign, ppm, unc_ms, phase) if "erreur" not in metrics else []

    # -- diagnostic (graphiques de la page)
    step = max(1, n // 240)
    tl = []
    if run is None:
        tl.append({"phase": PHASE_NAMES[0], "debut_s": 0.0, "fin_s": round(n / fps, 2)})
    else:
        m, i0, i1 = run
        tl = [{"phase": PHASE_NAMES[0], "debut_s": 0.0, "fin_s": round(m / fps, 2)},
              {"phase": PHASE_NAMES[1], "debut_s": round(m / fps, 2), "fin_s": round(i0 / fps, 2)},
              {"phase": PHASE_NAMES[2], "debut_s": round(i0 / fps, 2), "fin_s": round(i1 / fps, 2)},
              {"phase": PHASE_NAMES[3], "debut_s": round(i1 / fps, 2), "fin_s": round(n / fps, 2)}]
        tl = [p for p in tl if p["fin_s"] > p["debut_s"]]
    diagnostic = {
        "seuil_course_ms": V_RUN,
        "phases": tl,
        "vitesse_max_bassin_ms": round(vmax, 2),
        "appuis_candidats": len(candidates), "appuis_valides": len(contacts),
        "rejets": rej_count,
        "t": [round(i / fps, 3) for i in range(0, n, step)],
        "vitesse_bassin": _ds(speed, step),
        "pied_G_h_m": _ds(foot_h["G"], step), "pied_D_h_m": _ds(foot_h["D"], step),
        "appuis": [{"t": round(c.t_on, 3), "pied": c.foot, "ok": True} for c in contacts] +
                  [{"t": round(c.t_on, 3), "pied": c.foot, "ok": False} for c, _ in rejected],
    }

    ctx = dict(n=n, fps=fps, K=K, km=km, kps=kps, T=T_list, sign=sign, ppm=ppm, y_ground=y_ground,
               strides=strides, contacts=contacts, knee=knee, trunk=trunk, speed=speed, phase=phase)
    result = {
        "fps_utilise": fps,
        "images_analysees": n,
        "echelle_px_par_m": round(ppm, 1),
        "sens_course": "gauche -> droite" if sign > 0 else "droite -> gauche",
        "vitesse_max_hanches_ms": round(vmax, 2),
        "metriques": metrics,
        "appuis": [{"pied": c.foot, "debut_s": round(c.t_on, 3), "fin_s": round(c.t_off, 3),
                    "duree_ms": round(1000 * c.duration, 1), "x_m": round(c.x, 2)} for c in contacts],
        "qualite": {"athlete_detecte_pct": round(100 * pose_ok), "pieds_visibles_pct": round(100 * foot_ok),
                    "stabilisation_fiabilite_pct": (round(100 * inl) if camera == "mobile" else None)},
        "camera": camera,
        "avertissements": warns,
        "foulees": strides,
        "bilan": build_bilan(strides, unc_ms),
        "diagnostic": diagnostic,
    }
    return result, ctx


# =====================================================================
# 6. Fiches par foulée + commentaires
# =====================================================================

def _num(x, nd=1):
    try:
        return None if x is None or not np.isfinite(x) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _nm(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _fr(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def build_strides(contacts, fps, kps, knee, trunk, km, sign, ppm, unc_ms, phase=None):
    """Une fiche par appui. Commentaires : comparaison avec les foulées VOISINES de la même phase
    (départ ou course), jamais avec des valeurs d'élite inventées ni avec la course entière
    (en accélération, les foulées changent naturellement)."""
    n = len(kps)
    rows = []
    step_ts = np.diff([c.t_on for c in contacts]) if len(contacts) > 1 else np.array([])
    med_step = float(np.median(step_ts)) if len(step_ts) else None
    for k, c in enumerate(contacts):
        si = 0 if c.foot == "G" else 1
        fi = int(min(max(round(c.t_on * fps), 0), n - 1))
        fo = int(min(max(round(c.t_off * fps), fi), n - 1))
        ph = "départ" if (phase is not None and int(phase[fi]) == 1) else "course"
        r = {"n": k + 1, "pied": c.foot, "phase": ph, "t_on": round(c.t_on, 3), "t_off": round(c.t_off, 3), "frame": fi,
             "contact_ms": _num(1000 * c.duration), "vol_ms": None, "longueur_m": None,
             "frequence_hz": None, "vitesse_ms": None, "incertitude_ms": round(unc_ms)}
        if k + 1 < len(contacts) and med_step:
            nx = contacts[k + 1]
            st = nx.t_on - c.t_on
            if 0.5 * med_step <= st <= 1.6 * med_step and nx.x > c.x:
                r["vol_ms"] = _num(1000 * (nx.t_on - c.t_off))
                r["longueur_m"] = _num(nx.x - c.x, 2)
                r["frequence_hz"] = _num(1 / st, 2)
                r["vitesse_ms"] = _num((nx.x - c.x) / st, 2)
        fx = sign * _nm([kps[fi, km["heel"][si], 0], kps[fi, km["toe"][si], 0]]) / ppm
        hx = sign * _nm([kps[fi, 11, 0], kps[fi, 12, 0]]) / ppm
        r["attaque_m"] = _num(fx - hx, 2)
        r["genou_attaque_deg"] = _num(knee[fi, si], 0)
        seg = knee[fi:fo + 1, si]
        r["genou_flexion_max_deg"] = _num(np.nanmin(seg), 0) if np.isfinite(seg).any() else None
        r["tronc_deg"] = _num(trunk[fi], 0)
        rows.append(r)

    def neighbours(key, k):
        v = [rows[j][key] for j in range(max(0, k - 3), min(len(rows), k + 4))
             if j != k and rows[j][key] is not None and rows[j]["phase"] == rows[k]["phase"]]
        return float(np.median(v)) if len(v) >= 3 else None

    for k, r in enumerate(rows):
        notes = []
        mc, ml, ma, mk = (neighbours(key, k) for key in ("contact_ms", "longueur_m", "attaque_m", "genou_attaque_deg"))
        if r["phase"] == "départ":
            notes = ["Phase de départ : les foulées s'allongent naturellement, pas de comparaison."]
        elif all(m is None for m in (mc, ml, ma, mk)):
            notes = ["Pas assez de foulées voisines pour comparer : valeurs brutes uniquement."]
        else:
            if mc is not None and r["contact_ms"] is not None:
                d = r["contact_ms"] - mc
                if abs(d) > unc_ms:
                    notes.append(f"Contact {'plus long' if d > 0 else 'plus court'} que tes foulées voisines ({d:+.0f} ms).")
            if ml and r["longueur_m"] is not None and abs(r["longueur_m"] - ml) > 0.06 * ml:
                d = r["longueur_m"] - ml
                notes.append(f"Foulée {'plus longue' if d > 0 else 'plus courte'} que tes foulées voisines ({d:+.2f} m).")
            if ma is not None and r["attaque_m"] is not None:
                d = r["attaque_m"] - ma
                if d > 0.08:
                    notes.append("Pied posé plus loin devant le bassin que d'habitude : risque de freinage.")
                elif d < -0.08:
                    notes.append("Pied posé plus près du bassin que d'habitude.")
            if mk is not None and r["genou_attaque_deg"] is not None:
                d = r["genou_attaque_deg"] - mk
                if d > 8:
                    notes.append("Jambe plus tendue à l'attaque que d'habitude.")
                elif d < -8:
                    notes.append("Genou plus fléchi à l'attaque que d'habitude.")
            if not notes:
                notes = ["Foulée régulière, proche de tes foulées voisines."]
        r["commentaires"] = notes
        side = "gauche" if r["pied"] == "G" else "droit"
        parts = [f"Foulée {r['n']}, pied {side}."]
        if r["contact_ms"] is not None:
            parts.append(f"Contact {r['contact_ms']:.0f} millisecondes.")
        notable = [t for t in notes if not t.startswith(("Foulée régulière", "Pas assez", "Phase de départ"))]
        parts += [re.sub(r"\s*\(.*?\)", "", t) for t in notable[:1]]
        r["voix"] = " ".join(parts)
    return rows


def build_bilan(rows, unc_ms):
    out = []
    g = [r["contact_ms"] for r in rows if r["pied"] == "G" and r["contact_ms"] is not None]
    d = [r["contact_ms"] for r in rows if r["pied"] == "D" and r["contact_ms"] is not None]
    if len(g) >= 2 and len(d) >= 2:
        diff = np.mean(g) - np.mean(d)
        if abs(diff) > unc_ms:
            out.append(f"Asymétrie possible : contact moyen gauche {np.mean(g):.0f} ms, droit {np.mean(d):.0f} ms.")
        else:
            out.append("Pas d'asymétrie gauche/droite mesurable au-delà de la marge d'erreur.")
    ls = [r["longueur_m"] for r in rows if r["longueur_m"] is not None]
    if len(ls) >= 4:
        out.append(f"Régularité : la longueur de foulée varie de {100 * np.std(ls) / np.mean(ls):.0f} % d'une foulée à l'autre.")
    return out


# =====================================================================
# 7. Vidéo annotée : montage avec pause et fiche à chaque foulée
# =====================================================================

_FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
_FONTS = {}
# couleurs (RGB pour le texte, BGR pour OpenCV)
_C = {"vert": (80, 220, 90), "orange": (255, 150, 40), "cyan": (60, 200, 255), "blanc": (255, 255, 255),
      "rouge": (240, 70, 70), "gris": (170, 178, 190), "jaune": (255, 214, 10)}
_BG = (26, 20, 18)


def _bgr(c):
    return (c[2], c[1], c[0])


def _font(size):
    size = max(8, int(size))
    if size not in _FONTS:
        f = None
        for p in _FONT_PATHS:
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:  # noqa
                pass
        if f is None:
            try:
                f = ImageFont.load_default(size=size)
            except Exception:  # noqa
                f = ImageFont.load_default()
        _FONTS[size] = f
    return _FONTS[size]


def _wrap(txt, size, maxw):
    f, lines, cur = _font(size), [], ""
    for wd in txt.split():
        t = (cur + " " + wd).strip()
        if f.getlength(t) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def _flush_text(canvas, ops):
    img = Image.fromarray(np.ascontiguousarray(canvas[:, :, ::-1]))
    dr = ImageDraw.Draw(img)
    for x, y, txt, size, color, anchor, stroke in ops:
        dr.text((x, y), txt, font=_font(size), fill=color, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))
    canvas[:] = np.asarray(img)[:, :, ::-1]


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg introuvable : impossible d'encoder la vidéo annotée.")


def _to_frame(Tinv, pts):
    p = np.asarray(pts, float)
    return p @ Tinv[:2, :2].T + Tinv[:2, 2]


def _fmt(v, nd, unit=""):
    return "--" if v is None else f"{v:.{nd}f}{unit}"


def render_video(video_path, ctx, out_path, playback="slow", progress=lambda f: None, pauses=True):
    """Relit la vidéo et dessine : squelette, angles, appuis, empreintes au sol, bandeau de métriques en bas.
    À chaque foulée (si pauses=True) l'image se fige et la fiche complète s'affiche avec le commentaire.
    Remplit stride['t_out'] / stride['pause_s'] (temps dans la vidéo produite) pour la voix de la page."""
    n, fps, km = ctx["n"], ctx["fps"], ctx["km"]
    kps, T, sign, ppm, yg = ctx["kps"], ctx["T"], ctx["sign"], ctx["ppm"], ctx["y_ground"]
    contacts, strides = ctx["contacts"], ctx["strides"]
    step = 1 if playback == "slow" else max(1, int(round(fps / 30.0)))
    out_fps = 30.0 if playback == "slow" else fps / step

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    if not ok:
        raise ValueError("Relecture de la vidéo impossible.")
    h0, w0 = frame.shape[:2]
    w, h = (MAX_WIDTH, int(h0 * MAX_WIDTH / w0)) if w0 > MAX_WIDTH else (w0, h0)
    w, h = w - w % 2, h - h % 2
    sc = w / 960.0
    strip_h = int(round(250 * sc / 2)) * 2
    H = h + strip_h
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    th = max(1, int(round(2 * sc)))

    proc = subprocess.Popen(
        [_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{H}",
         "-r", f"{out_fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)

    edges = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10)]
    leg = {}
    for s_ in (0, 1):
        leg[s_] = [(km["hip"][s_], km["knee"][s_]), (km["knee"][s_], km["ankle"][s_]),
                   (km["ankle"][s_], km["heel"][s_]), (km["ankle"][s_], km["toe"][s_]), (km["heel"][s_], km["toe"][s_])]
        edges += leg[s_]
    left_set = {5, 7, 9, 11, 13, 15, km["heel"][0], km["toe"][0]}
    shown_pts = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, *km["heel"], *km["toe"])
    last_ct = {"G": None, "D": None}
    names = ("G", "D")

    def compose(i, frame, stride=None):
        t = i / fps
        ops = []
        Tinv = np.linalg.inv(T[i])
        P = _to_frame(Tinv, kps[i])
        ok_pt = ~np.isnan(P).any(axis=1)

        # empreintes collées au sol + longueur de foulée
        prev_pt, prev_c = None, None
        for c in [c for c in contacts if c.t_on <= t]:
            pt = _to_frame(Tinv, [c.x * sign * ppm, yg])
            if np.isfinite(pt).all() and 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), int(7 * sc), _bgr(_C["orange"] if c.foot == "G" else _C["cyan"]), -1, cv2.LINE_AA)
            if prev_pt is not None and np.isfinite(pt).all() and np.isfinite(prev_pt).all() and 0 <= prev_pt[0] < w and 0 <= pt[0] < w:
                y2 = int(min(prev_pt[1], pt[1]) + 14 * sc)
                cv2.line(frame, (int(prev_pt[0]), y2), (int(pt[0]), y2), (255, 255, 255), 1, cv2.LINE_AA)
                ops.append((int((prev_pt[0] + pt[0]) / 2), y2 + int(4 * sc), f"{abs(c.x - prev_c.x):.2f} m", 17 * sc, _C["blanc"], "ma", 2))
            prev_pt, prev_c = pt, c

        # squelette
        for a, b in edges:
            if ok_pt[a] and ok_pt[b]:
                col = _C["orange"] if (a in left_set and b in left_set) else (
                    _C["blanc"] if (a in (5, 6, 11, 12) and b in (5, 6, 11, 12)) else _C["cyan"])
                cv2.line(frame, tuple(P[a].astype(int)), tuple(P[b].astype(int)), _bgr(col), th + 1, cv2.LINE_AA)
        for j in shown_pts:
            if ok_pt[j]:
                cv2.circle(frame, tuple(P[j].astype(int)), int(3 * sc) + 1, (255, 255, 255), -1, cv2.LINE_AA)

        # phase + appui en cours (appuis comptés seulement pendant la course)
        ph = int(ctx["phase"][i])
        in_c = None
        for c in contacts:
            if c.t_on <= t < c.t_off:
                in_c = c
            if c.t_off <= t:
                last_ct[c.foot] = c
        base = ("APPUI " + in_c.foot) if in_c is not None else "VOL"
        phase = {0: "BLOCS / IMMOBILE", 1: "DÉPART · " + base, 2: base, 3: "FIN DE COURSE"}[ph]
        if in_c is not None and ph in (1, 2):
            s_ = names.index(in_c.foot)
            pf = (P[km["heel"][s_]] + P[km["toe"][s_]]) / 2
            if np.isfinite(pf).all():
                cv2.circle(frame, tuple(pf.astype(int)), int(16 * sc), _bgr(_C["vert"]), th + 1, cv2.LINE_AA)

        # angles de genou
        for s_ in (0, 1):
            k_, a_ = km["knee"][s_], ctx["knee"][i, s_]
            if ok_pt[k_] and not np.isnan(a_):
                ops.append((int(P[k_][0] + 9 * sc), int(P[k_][1] - 10 * sc), f"{a_:.0f}°", 19 * sc, _C["orange"] if s_ == 0 else _C["cyan"], "la", 2))

        y0 = h
        card_ops = []
        if stride is not None:
            # ---- géométrie de l'attaque, sur la vidéo
            s_ = 0 if stride["pied"] == "G" else 1
            for a, b in leg[s_]:
                if ok_pt[a] and ok_pt[b]:
                    cv2.line(frame, tuple(P[a].astype(int)), tuple(P[b].astype(int)), _bgr(_C["vert"]), th + 3, cv2.LINE_AA)
            hx = np.nanmean([kps[i, 11, 0], kps[i, 12, 0]])
            fxr = np.nanmean([kps[i, km["heel"][s_], 0], kps[i, km["toe"][s_], 0]])
            if np.isfinite(hx) and np.isfinite(fxr):
                a_pt, b_pt = _to_frame(Tinv, [hx, yg]), _to_frame(Tinv, [fxr, yg])
                hp = (P[11] + P[12]) / 2
                if np.isfinite(hp).all():
                    cv2.line(frame, tuple(hp.astype(int)), tuple(a_pt.astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(frame, tuple(a_pt.astype(int)), tuple(b_pt.astype(int)), _bgr(_C["vert"]), th + 3, cv2.LINE_AA)
                cv2.circle(frame, tuple(b_pt.astype(int)), int(9 * sc), _bgr(_C["vert"]), -1, cv2.LINE_AA)
                if stride["attaque_m"] is not None:
                    ops.append((int((a_pt[0] + b_pt[0]) / 2), int(max(a_pt[1], b_pt[1]) + 8 * sc),
                                f"attaque {stride['attaque_m']:+.2f} m", 20 * sc, _C["vert"], "ma", 3))
            ops.append((int(10 * sc), int(10 * sc), f"FOULÉE {stride['n']}", 30 * sc, _C["jaune"], "la", 3))

            # ---- fiche dans le bandeau du bas
            col = _C["orange"] if stride["pied"] == "G" else _C["cyan"]
            side = "GAUCHE" if stride["pied"] == "G" else "DROIT"
            card_ops.append((int(14 * sc), y0 + int(8 * sc), f"FOULÉE {stride['n']} · PIED {side}", 27 * sc, col, "la", 0))
            tiles = [("CONTACT", _fmt(stride["contact_ms"], 0, " ms"), f"±{stride['incertitude_ms']}"),
                     ("VOL", _fmt(stride["vol_ms"], 0, " ms"), f"±{stride['incertitude_ms']}"),
                     ("LONGUEUR", _fmt(stride["longueur_m"], 2, " m"), ""),
                     ("FRÉQUENCE", _fmt(stride["frequence_hz"], 2, " Hz"), ""),
                     ("VITESSE", _fmt(stride["vitesse_ms"], 1, " m/s"), ""),
                     ("ATTAQUE", _fmt(stride["attaque_m"], 2, " m"), "pied/bassin"),
                     ("GENOU ATTAQUE", _fmt(stride["genou_attaque_deg"], 0, "°"),
                      "" if stride["genou_flexion_max_deg"] is None else f"min {stride['genou_flexion_max_deg']:.0f}°"),
                     ("TRONC", _fmt(stride["tronc_deg"], 0, "°"), "")]
            cw = w / 4.0
            for k_, (lab, val, sub) in enumerate(tiles):
                cx, cy = int(14 * sc + (k_ % 4) * cw), y0 + int((46 + (k_ // 4) * 62) * sc)
                card_ops.append((cx, cy, lab, 13 * sc, _C["gris"], "la", 0))
                card_ops.append((cx, cy + int(15 * sc), val, 30 * sc, _C["blanc"], "la", 0))
                if sub:
                    vx = cx + _font(30 * sc).getlength(val) + 6 * sc
                    card_ops.append((int(vx), cy + int(28 * sc), sub, 13 * sc, _C["gris"], "la", 0))
            lines = _wrap(" ".join(stride["commentaires"][:2]), 20 * sc, w - 28 * sc)[:2]
            for k_, ln in enumerate(lines):
                card_ops.append((int(14 * sc), y0 + int((176 + k_ * 26) * sc), ln, 20 * sc, _C["jaune"], "la", 0))
        else:
            v, tr = ctx["speed"][i], ctx["trunk"][i]
            gt = f"{1000 * last_ct['G'].duration:.0f} ms" if last_ct["G"] else "--"
            dt = f"{1000 * last_ct['D'].duration:.0f} ms" if last_ct["D"] else "--"
            seen = [c for c in contacts if c.t_on <= t]
            fq = f"{1 / np.median(np.diff([c.t_on for c in seen])):.2f} Hz" if len(seen) >= 3 else "--"
            pcol = _C["gris"] if ph in (0, 3) else (_C["vert"] if in_c else _C["rouge"])
            card_ops.append((int(14 * sc), y0 + int(10 * sc), phase, 34 * sc, pcol, "la", 0))
            card_ops.append((int(w - 14 * sc), y0 + int(16 * sc), f"{t:5.2f} s", 22 * sc, _C["gris"], "ra", 0))
            tiles = [("VITESSE", f"{v:.1f} m/s" if v == v else "--"), ("CONTACT G", gt), ("CONTACT D", dt),
                     ("FRÉQUENCE", fq), ("TRONC", f"{tr:.0f}°" if tr == tr else "--")]
            cw = w / 5.0
            for k_, (lab, val) in enumerate(tiles):
                cx, cy = int(14 * sc + k_ * cw), y0 + int(86 * sc)
                card_ops.append((cx, cy, lab, 14 * sc, _C["gris"], "la", 0))
                card_ops.append((cx, cy + int(18 * sc), val, 34 * sc, _C["blanc"], "la", 0))

        canvas = np.empty((H, w, 3), np.uint8)
        canvas[:h] = frame
        canvas[h:] = _BG
        _flush_text(canvas, ops + card_ops)
        return canvas

    outn, ptr, i = 0, 0, 0
    try:
        while i < n:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                if ptr < len(strides) and strides[ptr]["frame"] <= i:
                    s = strides[ptr]
                    ptr += 1
                    s["t_out"] = round(outn / out_fps, 2)
                    if pauses:
                        dur = float(min(6.0, max(3.0, len(s["voix"]) / 16.0 + 0.8)))
                        s["pause_s"] = round(dur, 1)
                        buf = compose(i, frame.copy(), s).tobytes()
                        for _ in range(int(round(dur * out_fps))):
                            proc.stdin.write(buf)
                        outn += int(round(dur * out_fps))
                        i += 1
                        continue
                    s["pause_s"] = 0.0
                proc.stdin.write(compose(i, frame).tobytes())
                outn += 1
            i += 1
            if i % 10 == 0:
                progress(i / n)
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:  # noqa
            pass
        proc.wait()
    for s in strides:                      # foulées sans image (hors vidéo) : pas de chapitre vocal
        s.setdefault("t_out", None)
        s.setdefault("pause_s", 0.0)
    if proc.returncode != 0:
        raise RuntimeError("Échec de l'encodage de la vidéo annotée.")
    progress(1.0)


def parse_range(header, size):
    """En-tête HTTP Range -> (start, end) inclusifs, ou None si absent/invalide. Nécessaire pour lire la vidéo sur iPhone."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    a, _, b = header[6:].partition("-")
    try:
        if a == "":
            start, end = max(0, size - int(b)), size - 1
        else:
            start, end = int(a), (int(b) if b else size - 1)
    except ValueError:
        return None
    end = min(end, size - 1)
    return (start, end) if 0 <= start <= end else "invalid"
