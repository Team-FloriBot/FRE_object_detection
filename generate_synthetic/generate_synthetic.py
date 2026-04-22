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

def random_place_no_overlap(bg, obj, mask, polygon, total_mask, max_tries=50):
    bg_h, bg_w = bg.shape[:2]
    obj_h, obj_w = obj.shape[:2]
    poly = None if polygon is None else polygon.copy()

    # Falls Objekt zu groß für Hintergrund, runterskalieren
    if obj_h >= bg_h or obj_w >= bg_w:
        scale = min(bg_h / obj_h, bg_w / obj_w) * 0.9
        obj = cv2.resize(obj, None, fx=scale, fy=scale)
        mask = cv2.resize(mask, None, fx=scale, fy=scale)
        if poly is not None:
            poly = poly * scale
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
            return x, y, obj_h, obj_w, poly, True
    
    # Falls nach max_tries kein Platz gefunden wurde
    return 0, 0, 0, 0, poly, False

# --- CONFIGURATION ---
NUM_GENERATED_IMAGES = 200       # Wie viele Bilder insgesamt erstellt werden sollen
OBJS_PER_IMAGE = (2, 6)         # Zufällige Anzahl (Min, Max) an Objekten pro Bild
SCALE_RANGE = (0.3, 0.8)        # Skalierungsfaktor relativ zum Hintergrund (Min, Max)
ROTATION_RANGE = (-10, 10)      # Drehung in Grad
FLIP_PROB = 0.5                 # 50% Chance für horizontales Spiegeln
BRIGHTNESS_RANGE = (0.6, 1.4)   # Helligkeits-Augmentation (0.7 = dunkler, 1.3 = heller)
BLUR_PROB = 0.2                 # Chance für leichte Unschärfe (fokussiert vs. unfokussiert)

# --- OUTDOOR AUGMENTATION CONFIG ---
MOTION_BLUR_PROB = 0.25
SHADOW_PROB = 0.30
COLOR_TEMP_PROB = 0.30
CONTRAST_PROB = 0.30
JPEG_ARTIFACT_PROB = 0.30

# --- BACKGROUND AUGMENTATION CONFIG ---
BG_SHADOW_PROB = 0.35          # Schatten auf Hintergrund
BG_LIGHT_PROB = 0.30           # Lichtflecken / Sonne
BG_COLOR_TEMP_PROB = 0.30      # Farbtemperatur (warm/kalt)
BG_CONTRAST_PROB = 0.30        # Kontrastvariation
BG_JPEG_ARTIFACT_PROB = 0.25   # JPEG-Kompression
BG_HAZE_PROB = 0.20            # leichter Dunst / Nebel

# ---------------------

# ---------------------------------------------------------
# Train/Val/Test Split konfigurieren
# ---------------------------------------------------------
TRAIN_SPLIT = 0.7   # 70%
VAL_SPLIT = 0.2     # 20%
TEST_SPLIT = 0.1    # 10%

assert abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) < 1e-6, \
    "Splits müssen zusammen 1.0 ergeben!"

# Zielordner erstellen
for split in ["train", "val", "test"]:
    os.makedirs(f"output/images/{split}", exist_ok=True)
    os.makedirs(f"output/labels/{split}", exist_ok=True)

def choose_split():
    r = random.random()
    if r < TRAIN_SPLIT:
        return "train"
    elif r < TRAIN_SPLIT + VAL_SPLIT:
        return "val"
    else:
        return "test"
    
def augment_background(bg):
    h, w = bg.shape[:2]

    # 1. Schattenwurf
    if random.random() < BG_SHADOW_PROB:
        shadow = np.zeros_like(bg)
        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h
        cv2.line(shadow, (x1, y1), (x2, y2), (0, 0, 0), random.randint(80, 200))
        bg = cv2.addWeighted(bg, 1, shadow, -0.4, 0)

    # 2. Lichtflecken (Sonne)
    if random.random() < BG_LIGHT_PROB:
        light = np.zeros_like(bg)
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(80, 200)
        cv2.circle(light, (cx, cy), radius, (255, 255, 255), -1)
        bg = cv2.addWeighted(bg, 1, light, 0.25, 0)

    # 3. Farbtemperatur (warm/kalt)
    if random.random() < BG_COLOR_TEMP_PROB:
        shift = random.randint(-25, 25)
        bg = cv2.add(bg, np.array([shift, shift//2, -shift]).astype(np.int8))

    # 4. Kontrastvariation
    if random.random() < BG_CONTRAST_PROB:
        alpha = random.uniform(0.8, 1.3)
        bg = cv2.convertScaleAbs(bg, alpha=alpha, beta=0)

    # 5. JPEG-Kompression
    if random.random() < BG_JPEG_ARTIFACT_PROB:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 90)]
        _, enc = cv2.imencode('.jpg', bg, encode_param)
        bg = cv2.imdecode(enc, 1)

    # 6. Leichter Dunst / Nebel
    if random.random() < BG_HAZE_PROB:
        haze = np.full_like(bg, random.randint(150, 200))
        alpha = random.uniform(0.05, 0.15)
        bg = cv2.addWeighted(bg, 1 - alpha, haze, alpha, 0)

    
    return bg

def augment_object(obj, mask, polygon_local):
    # 1. Skalierung
    scale = random.uniform(*SCALE_RANGE)
    obj = cv2.resize(obj, None, fx=scale, fy=scale)
    mask = cv2.resize(mask, None, fx=scale, fy=scale)

    # 2. Horizontal Flip
    if random.random() < FLIP_PROB:
        obj = cv2.flip(obj, 1)
        mask = cv2.flip(mask, 1)

    # 3. Rotation
    angle = random.uniform(*ROTATION_RANGE)
    h, w = obj.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    obj = cv2.warpAffine(obj, M, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # Maske wieder binär machen
    mask = (mask > 127).astype(np.uint8) * 255

    # 4. Helligkeit
    brightness = random.uniform(*BRIGHTNESS_RANGE)
    obj = cv2.convertScaleAbs(obj, alpha=brightness, beta=0)

    # --- OUTDOOR AUGS NUR AUF OBJ ---
    if random.random() < MOTION_BLUR_PROB:
        k = random.choice([3, 5, 7])
        angle = random.uniform(-10, 10)
        M_blur = cv2.getRotationMatrix2D((k/2, k/2), angle, 1)
        kernel = np.diag(np.ones(k))
        kernel = cv2.warpAffine(kernel, M_blur, (k, k))
        kernel = kernel / k
        obj = cv2.filter2D(obj, -1, kernel)

    if random.random() < SHADOW_PROB:
        shadow = np.zeros_like(obj)
        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h
        cv2.line(shadow, (x1, y1), (x2, y2), (0, 0, 0), random.randint(50, 150))
        obj = cv2.addWeighted(obj, 1, shadow, -0.5, 0)

    if random.random() < COLOR_TEMP_PROB:
        shift = random.randint(-20, 20)
        obj = cv2.add(obj, np.array([shift, shift//2, -shift]).astype(np.int8))

    if random.random() < CONTRAST_PROB:
        alpha = random.uniform(0.8, 1.3)
        obj = cv2.convertScaleAbs(obj, alpha=alpha, beta=0)

    if random.random() < JPEG_ARTIFACT_PROB:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 90)]
        _, enc = cv2.imencode('.jpg', obj, encode_param)
        obj = cv2.imdecode(enc, 1)

    # --- Polygon jetzt AUS DER MASKE holen ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return obj, mask, None  # nichts gefunden

    # größte Kontur nehmen
    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt.reshape(-1, 2).astype(np.float32)

    return obj, mask, cnt




# Hauptschleife zur Generierung
for i in range(NUM_GENERATED_IMAGES):
    print(f"Generiere Bild {i+1}/{NUM_GENERATED_IMAGES}...")
    # Hintergrund wählen und vorbereiten
    bg_name = random.choice(os.listdir(BG_DIR))
    bg = cv2.imread(os.path.join(BG_DIR, bg_name))
    bg = cv2.resize(bg, (640, 640))

    # Hintergrund augmentieren
    bg = augment_background(bg)

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

        ys, xs = np.where(mask > 0)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()


        obj_crop = obj[y_min:y_max + 1, x_min:x_max + 1]
        mask_crop = mask[y_min:y_max + 1, x_min:x_max + 1]

        polygon_local = polygon.copy()
        polygon_local[:, 0] -= x_min
        polygon_local[:, 1] -= y_min

        # Augmentation anwenden – polygon_local wird ignoriert
        aug_obj, aug_mask, aug_poly = augment_object(obj_crop, mask_crop, polygon_local)

        if aug_poly is None:
            continue

        
        # 2. Platzieren mit Überlappungs-Check
        x_off, y_off, final_h, final_w, placed_poly, success = random_place_no_overlap(
            bg, aug_obj, aug_mask, aug_poly, total_mask
        )
        
        if success:
            # Polygon an die globale Position verschieben
            final_poly = placed_poly + np.array([x_off, y_off])

            # --- WICHTIG: Polygon clampen, damit keine Werte > Bildgröße entstehen ---
            final_poly[:, 0] = np.clip(final_poly[:, 0], 0, bg_w - 1)
            final_poly[:, 1] = np.clip(final_poly[:, 1], 0, bg_h - 1)

            # Normalisieren (jetzt garantiert <= 1)
            norm_poly = normalize_polygon(final_poly, bg_w, bg_h)

            labels_to_save.append((cls, norm_poly))
        else:
            print(f"Bild {i}: Kein Platz für ein weiteres Objekt gefunden.")


    # Speichern
    split = choose_split()

    out_name = f"synth_{i}.jpg"

    # Bild speichern
    cv2.imwrite(os.path.join(f"output/images/{split}", out_name), bg)

    # Label speichern
    with open(os.path.join(f"output/labels/{split}", f"synth_{i}.txt"), "w") as f:
        for c, poly in labels_to_save:
            poly_str = " ".join([f"{coord[0]:.6f} {coord[1]:.6f}" for coord in poly])
            f.write(f"{c} {poly_str}\n")

