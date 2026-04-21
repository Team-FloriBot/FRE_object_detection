import cv2
import os
import random
import numpy as np

DATASET_DIR = "raw_images/train"
BG_DIR = "backgrounds"
OUT_IMG = "output/images"
OUT_LABEL = "output/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LABEL, exist_ok=True)

# Unterstützte Bildformate erweitern
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")

def load_polygon_label(path):
    with open(path, "r") as f:
        line = f.readline().strip().split()
        cls = int(line[0])
        coords = list(map(float, line[1:]))
        polygon = np.array(coords).reshape(-1, 2)
    return cls, polygon

def polygon_to_mask(img_shape, polygon):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = polygon.astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def random_place(bg, obj, mask):
    bg_h, bg_w = bg.shape[:2]
    obj_h, obj_w = obj.shape[:2]

    if obj_h >= bg_h or obj_w >= bg_w:
        scale = min(bg_h / obj_h, bg_w / obj_w) * 0.5
        obj = cv2.resize(obj, None, fx=scale, fy=scale)
        mask = cv2.resize(mask, None, fx=scale, fy=scale)
        obj_h, obj_w = obj.shape[:2]

    x = random.randint(0, bg_w - obj_w)
    y = random.randint(0, bg_h - obj_h)

    roi = bg[y:y+obj_h, x:x+obj_w]
    roi[mask > 0] = obj[mask > 0]
    bg[y:y+obj_h, x:x+obj_w] = roi

    return x, y, obj_h, obj_w

def transform_polygon(polygon, x_offset, y_offset, scale_x, scale_y):
    poly = polygon.copy()
    poly[:, 0] = poly[:, 0] * scale_x + x_offset
    poly[:, 1] = poly[:, 1] * scale_y + y_offset
    return poly

def normalize_polygon(poly, img_w, img_h):
    poly_norm = poly.copy()
    poly_norm[:, 0] /= img_w
    poly_norm[:, 1] /= img_h
    return poly_norm

def random_place_no_overlap(bg, obj, mask, total_mask, max_tries=50):
    bg_h, bg_w = bg.shape[:2]
    obj_h, obj_w = obj.shape[:2]

    # Falls Objekt zu groß für Hintergrund, runterskalieren
    if obj_h >= bg_h or obj_w >= bg_w:
        scale = min(bg_h / obj_h, bg_w / obj_w) * 0.9
        obj = cv2.resize(obj, None, fx=scale, fy=scale)
        mask = cv2.resize(mask, None, fx=scale, fy=scale)
        obj_h, obj_w = obj.shape[:2]

    for _ in range(max_tries):
        # Zufällige Position wählen
        x = random.randint(0, bg_w - obj_w)
        y = random.randint(0, bg_h - obj_h)

        # Prüfen, ob an dieser Stelle bereits etwas liegt
        roi_mask = total_mask[y:y+obj_h, x:x+obj_w]
        
        # bitwise_and prüft, ob die neue Maske die alte überlappt
        overlap = cv2.bitwise_and(roi_mask, mask)
        
        if np.sum(overlap) == 0:
            # Kein Überlapp! Jetzt platzieren
            roi_bg = bg[y:y+obj_h, x:x+obj_w]
            roi_bg[mask > 0] = obj[mask > 0]
            
            # Die Belegungsmaske aktualisieren
            total_mask[y:y+obj_h, x:x+obj_w] = cv2.bitwise_or(roi_mask, mask)
            
            return x, y, obj_h, obj_w, True
    
    # Falls nach max_tries kein Platz gefunden wurde
    return 0, 0, 0, 0, False

# --- CONFIGURATION ---
NUM_GENERATED_IMAGES = 100       # Wie viele Bilder insgesamt erstellt werden sollen
OBJS_PER_IMAGE = (2, 6)         # Zufällige Anzahl (Min, Max) an Objekten pro Bild
SCALE_RANGE = (0.4, 0.8)        # Skalierungsfaktor relativ zum Hintergrund (Min, Max)
ROTATION_RANGE = (-20, 20)      # Drehung in Grad
FLIP_PROB = 0.5                 # 50% Chance für horizontales Spiegeln
BRIGHTNESS_RANGE = (0.7, 1.3)   # Helligkeits-Augmentation (0.7 = dunkler, 1.3 = heller)
BLUR_PROB = 0.2                 # Chance für leichte Unschärfe (fokussiert vs. unfokussiert)
# ---------------------

def augment_object(obj, mask, polygon_local):
    # 1. Zufällige Skalierung
    scale = random.uniform(*SCALE_RANGE)
    obj = cv2.resize(obj, None, fx=scale, fy=scale)
    mask = cv2.resize(mask, None, fx=scale, fy=scale)
    polygon_local = polygon_local * scale

    # 2. Zufällige Spiegelung (Horizontal)
    if random.random() < FLIP_PROB:
        obj = cv2.flip(obj, 1)
        mask = cv2.flip(mask, 1)
        h, w = obj.shape[:2]
        polygon_local[:, 0] = w - polygon_local[:, 0]

    # 3. Zufällige Rotation
    angle = random.uniform(*ROTATION_RANGE)
    h, w = obj.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Bild und Maske rotieren
    obj = cv2.warpAffine(obj, M, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    
    # Polygon-Punkte rotieren
    ones = np.ones(shape=(len(polygon_local), 1))
    points_ones = np.concatenate([polygon_local, ones], axis=1)
    polygon_local = M.dot(points_ones.T).T

    # 4. Helligkeit anpassen
    brightness = random.uniform(*BRIGHTNESS_RANGE)
    obj = cv2.convertScaleAbs(obj, alpha=brightness, beta=0)

    return obj, mask, polygon_local

# Hauptschleife zur Generierung
for i in range(NUM_GENERATED_IMAGES):
    print(f"Generiere Bild {i+1}/{NUM_GENERATED_IMAGES}...")
    # Hintergrund wählen und vorbereiten
    bg_name = random.choice(os.listdir(BG_DIR))
    bg = cv2.imread(os.path.join(BG_DIR, bg_name))
    bg = cv2.resize(bg, (640, 640))
    bg_h, bg_w = bg.shape[:2]
    
    labels_to_save = []

    # 1. Belegungsmaske für dieses Bild initialisieren (alles schwarz)
    total_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    labels_to_save = []
    
    # Zufällige Anzahl an Objekten platzieren
    num_objs = random.randint(*OBJS_PER_IMAGE)
    for _ in range(num_objs):
        # Zufälliges Quellbild wählen
        img_name = random.choice([f for f in os.listdir(os.path.join(DATASET_DIR, "images")) if f.lower().endswith(VALID_EXT)])
        
        if not img_name.lower().endswith(VALID_EXT):
            continue

        img_path = os.path.join(DATASET_DIR, "images", img_name)
        label_path = os.path.join(DATASET_DIR, "labels", img_name.rsplit(".", 1)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        cls, polygon_norm = load_polygon_label(label_path)
        polygon = polygon_norm.copy()
        polygon[:, 0] *= w
        polygon[:, 1] *= h

        mask = polygon_to_mask(img.shape, polygon)
        obj = cv2.bitwise_and(img, img, mask=mask)

        x_min, y_min = polygon.min(axis=0).astype(int)
        x_max, y_max = polygon.max(axis=0).astype(int)

        obj_crop = obj[y_min:y_max, x_min:x_max]
        mask_crop = mask[y_min:y_max, x_min:x_max]

        polygon_local = polygon - np.array([x_min, y_min])
        
        # Augmentation anwenden
        aug_obj, aug_mask, aug_poly = augment_object(obj_crop, mask_crop, polygon_local)
        
        # 2. Platzieren mit Überlappungs-Check
        x_off, y_off, final_h, final_w, success = random_place_no_overlap(
            bg, aug_obj, aug_mask, total_mask
        )
        
        if success:
            # Nur speichern, wenn das Objekt wirklich platziert wurde
            final_poly = aug_poly + np.array([x_off, y_off])
            norm_poly = normalize_polygon(final_poly, bg_w, bg_h)
            labels_to_save.append((cls, norm_poly))
        else:
            print(f"Bild {i}: Kein Platz für ein weiteres Objekt gefunden.")

    # Speichern
    out_name = f"synth_{i}.jpg"
    cv2.imwrite(os.path.join(OUT_IMG, out_name), bg)
    with open(os.path.join(OUT_LABEL, f"synth_{i}.txt"), "w") as f:
        for c, poly in labels_to_save:
            poly_str = " ".join([f"{coord[0]:.6f} {coord[1]:.6f}" for coord in poly])
            f.write(f"{c} {poly_str}\n")
