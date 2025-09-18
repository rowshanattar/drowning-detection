import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np

# ---------------------------------------------
# SAM (Segment Anything) vorbereiten
# ---------------------------------------------
# Annahme: sam_vit_h.pth liegt lokal im Arbeitsverzeichnis.
sam_checkpoint = "sam_vit_h.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# SAM-Modell laden und auf das passende Device schieben
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Streamlit-Überschrift
st.title("🎯 DeepSORT-basierte Ertrinkungserkennung mit SAM + YOLOv8 + DeepSort")

# ---------------------------------------------
# Alarm-Logik: einfache Unsichtbarkeits-Heuristik
# ---------------------------------------------
class DrowningDetector:
    """
    Sehr einfache Heuristik:
    - Merkt sich, wann eine Track-ID zuletzt gesehen wurde (Frame-Index).
    - Wenn eine Track-ID länger als threshold_frames nicht gesehen wurde,
      erhöht sich ein 'missing_count'. Ab 'max_warnings' wird ein Alarm gesetzt.
    Hinweise:
    - Diese Logik modelliert nur 'Unsichtbarkeit', nicht echtes Ertrinkungsverhalten.
    - FPS muss zur tatsächlichen Video-FPS passen!
    """
    def __init__(self, fps=30, threshold_seconds=2, max_warnings=1):
        self.fps = fps
        self.threshold_frames = int(threshold_seconds * fps)
        self.max_warnings = max_warnings
        self.last_seen = {}        # track_id -> letzter Frame, in dem die ID gesehen wurde
        self.last_position = {}    # track_id -> letzte bekannte Position (cx, cy)
        self.missing_counts = {}   # track_id -> wie oft 'unsichtbar > Schwelle' passiert ist
        self.alarms = set()        # track_ids, für die bereits Alarm ausgegeben wurde

    def update(self, current_frame_index, visible_objects):
        """
        visible_objects: Liste von (track_id, (cx, cy)) für IDs, die im aktuellen Frame sichtbar sind.
        Rückgabe: Liste neuer Alarme (track_ids), die in DIESEM Frame ausgelöst wurden.
        """
        alerts = []
        current_ids = set()

        # 1) Sichtbare IDs im aktuellen Frame registrieren
        for track_id, position in visible_objects:
            track_id = int(track_id)
            current_ids.add(track_id)
            self.last_seen[track_id] = current_frame_index
            self.last_position[track_id] = position

        # 2) Für alle bekannten IDs prüfen, ob sie aktuell fehlen
        for track_id in list(self.last_seen.keys()):
            if track_id not in current_ids:
                missing_duration = current_frame_index - self.last_seen[track_id]
                # Wenn länger als Schwelle nicht gesehen -> Missing-Count erhöhen
                if missing_duration > self.threshold_frames:
                    self.missing_counts[track_id] = self.missing_counts.get(track_id, 0) + 1
                    # Zur Entzerrung: 'last_seen' auf aktuellen Frame setzen,
                    # damit nicht in jedem folgenden Frame sofort wieder erhöht wird
                    self.last_seen[track_id] = current_frame_index
                    # Ab max_warnings -> einmaliger Alarm (pro ID)
                    if self.missing_counts[track_id] >= self.max_warnings:
                        if track_id not in self.alarms:
                            self.alarms.add(track_id)
                            alerts.append(track_id)
        return alerts

# ---------------------------------------------
# Streamlit: Video hochladen
# ---------------------------------------------
uploaded_video = st.file_uploader("🎥 Video hochladen", type=["mp4", "avi"])

if uploaded_video is not None:
    # Temporäre Datei auf Platte (OpenCV braucht einen Pfad)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    tfile.close()  # Datei schließen, damit andere Prozesse sie lesen können
    video_path = tfile.name

    if st.button("▶️ Run Object Tracking"):
        st.text("Verarbeitung läuft...")

        # Modelle/Tracker instanziieren
        yolo = YOLO("yolov8l.pt")  # weder zu einfach noch zu kompliziert
        # DeepSORT-Parameter:
        # - max_age: wie viele Frames darf eine ID ohne Detektion "überleben"
        # - n_init: wie viele Bestätigungen nötig, bis eine Spur als "confirmed" gilt
        # - max_cosine_distance: Schwellwert für ReID-Ähnlichkeit
        deepsort = DeepSort(max_age=90, n_init=5, max_cosine_distance=0.4)

        # Alarm-Heuristik (FPS-Schätzung anpassen, falls Video-FPS abweichen)
        drowning_detector = DrowningDetector(fps=30, threshold_seconds=2)
        frame_count = 0
        water_box = None  # (x_min, y_min, x_max, y_max)

        # Video I/O
        cap = cv2.VideoCapture(video_path)
        output_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_h, frame_w = frame.shape[:2]

            # ---------------------------------------------
            # SAM: nur im ersten Frame (Performance + Robustheit)
            # ---------------------------------------------
            if frame_count == 1:
                predictor.set_image(frame)
                # Minimaler Prompt: ein Punkt grob im Wasserbereich in den Videos( Annahme: dieser Punkt liegt immer im Wasser)
                input_point = np.array([[frame_w // 2, int(frame_h * 0.6)]])
                input_label = np.array([1])

                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )
                # Heuristik: nimm die erste Maske (könnte man durch Auswahl der größten Fläche robuster machen)
                water_mask = masks[0]

                # Bounding Box des Wasserbereichs aus der Maske ableiten
                ys, xs = np.where(water_mask)
                x_min, x_max = np.min(xs), np.max(xs)
                y_min, y_max = np.min(ys), np.max(ys)
                water_box = (x_min, y_min, x_max, y_max)

            # ---------------------------------------------
            # YOLO: Personendetektion
            # ---------------------------------------------
            results = yolo(frame, imgsz=960)
            detections = []  # für DeepSORT: ([x, y, w, h], conf, label)

            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    label = result.names[int(cls)]
                    # Nur Personen und ausreichend Konfidenz
                    if label != "person" or conf < 0.5:
                        continue

                    x = int(x1); y = int(y1)
                    w = int(x2 - x1); h = int(y2 - y1)
                    cx = x + w // 2
                    cy = y + h // 2

                    # Nur Personen innerhalb des Wasserrechtecks berücksichtigen
                    if water_box:
                        x_min, y_min, x_max, y_max = water_box
                        if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                            continue

                    detections.append(([x, y, w, h], conf, label))

            # ---------------------------------------------
            # DeepSORT: Tracking-Update
            # ---------------------------------------------
            tracks = deepsort.update_tracks(detections, frame=frame)
            print(f"[Frame {frame_count}] YOLO: {len(detections)} Personen – DeepSORT: {len(tracks)} Tracks")

            visible_objects = []

            # Visualisierung + Sammeln sichtbarer Track-Zentren
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = int(track.track_id)
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                visible_objects.append((track_id, (center_x, center_y)))

                # Bounding-Box + Track-ID ins Frame zeichnen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ---------------------------------------------
            # Alarm-Update (Unsichtbarkeits-Heuristik)
            # ---------------------------------------------
            alerts = drowning_detector.update(
                current_frame_index=frame_count,
                visible_objects=visible_objects
            )

            # Aktive Alarme (rot) einblenden
            for i, alert_id in enumerate(drowning_detector.alarms):
                try:
                    pos = drowning_detector.last_position[alert_id]
                    line_offset = 50 + 30 * i
                    cv2.putText(frame, f" ERTRINKUNG ID {alert_id} bei Position {pos}!",
                                (30, line_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except Exception:
                    # Falls Position ausnahmsweise nicht vorliegt
                    pass

            # Wasserrechteck (blau) einzeichnen
            if water_box:
                x_min, y_min, x_max, y_max = water_box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # ---------------------------------------------
            # Videoausgabe initialisieren und schreiben
            # ---------------------------------------------
            if out is None:
                # Annahme: 20 FPS (kann an Video-FPS angepasst werden)
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            out.write(frame)

        # Ressourcen freigeben
        cap.release()
        if out is not None:
            out.release()

        st.success("Fertig! 🎉")
        st.video(output_path)

    # UI: Reset
    if st.button("❌ Exit / Reset"):
        st.experimental_rerun()
