import cv2
import os
import random
import numpy as np

OBJECT_DIR = "object_images/train"
BG_DIR_OUT = "backgrounds/outside_context"
BG_DIR_TAR = "backgrounds/target_context"
OUT_IMG = "dataset/images"
OUT_LABEL = "dataset/labels"

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
    # bg: uint8 HxWx3, obj: uint8 HxWx3, mask: uint8 HxW (0/255), total_mask: uint8 HxW (0/255)
    if obj is None or mask is None:
        return 0, 0, 0, 0, polygon, False

    bg_h, bg_w = bg.shape[:2]
    obj_h, obj_w = obj.shape[:2]
    poly = None if polygon is None else polygon.copy()

    # Falls Objekt zu groß, runterskalieren (sicher)
    if obj_h >= bg_h or obj_w >= bg_w:
        scale = min((bg_h - 1) / obj_h, (bg_w - 1) / obj_w) * 0.9
        if scale <= 0:
            return 0, 0, 0, 0, poly, False
        interp_resize = cv2.INTER_AREA
        obj = cv2.resize(obj, None, fx=scale, fy=scale, interpolation=interp_resize)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if poly is not None:
            poly *= scale
        obj_h, obj_w = obj.shape[:2]

    for _ in range(max_tries):
        x = random.randint(0, bg_w - obj_w)
        y = random.randint(0, bg_h - obj_h)

        roi_mask = total_mask[y:y+obj_h, x:x+obj_w]

        # --- Safety: mask in das erwartete Format bringen (uint8, single channel, gleiche Größe) ---
        mask_u = mask

        # Falls Maske float (0..1) -> in 0..255 uint8 konvertieren
        if mask_u.dtype == np.float32 or mask_u.dtype == np.float64:
            if mask_u.max() <= 1.0:
                mask_u = (mask_u * 255.0).astype(np.uint8)
            else:
                mask_u = mask_u.astype(np.uint8)

        # Falls Maske 3 Kanäle hat, auf ersten Kanal reduzieren
        if mask_u.ndim == 3 and mask_u.shape[2] > 1:
            mask_u = mask_u[..., 0]

        # Sicherstellen, dass Maske uint8 ist
        if mask_u.dtype != np.uint8:
            mask_u = mask_u.astype(np.uint8)

        # Falls die Größen nicht übereinstimmen, Maske auf roi_mask-Größe skalieren (INTER_NEAREST)
        if mask_u.shape != roi_mask.shape:
            mask_u = cv2.resize(mask_u, (roi_mask.shape[1], roi_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Jetzt ist mask_u uint8 und hat dieselbe Form wie roi_mask → bitwise_and ist sicher
        overlap = cv2.bitwise_and(roi_mask, mask_u)


    if np.sum(overlap) == 0:

        roi_bg = bg[y:y+obj_h, x:x+obj_w].astype(np.float32)

        mask_u = mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        eroded = cv2.erode(mask_u, kernel, iterations=1)
        dilate = cv2.dilate(mask_u, kernel, iterations=1)

        border = dilate - eroded  # 2 Pixel breiter Rand
        border_blur = cv2.GaussianBlur(border.astype(np.float32), (5,5), 2)

        # Soft mask
        mask_soft = np.clip(eroded.astype(np.float32) + border_blur, 0, 255).astype(np.uint8)
        alpha = (mask_soft.astype(np.float32) / 255.0)[..., None]

        # Schwarze Pixel finden
        gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
        black = gray < 100

        # Nur dort schwarze Pixel ersetzen, wo die Maske weich ist
        # (mask_soft < 255 bedeutet: Übergangsbereich)
        soft_region = mask_soft < 255
        black_soft = np.logical_and(black, soft_region)

        obj = obj.astype(np.float32)
        obj[black_soft] = roi_bg[black_soft]

        # Blending
        obj_f = obj.astype(np.float32)
        blended = (alpha * obj_f + (1 - alpha) * roi_bg).astype(np.uint8)

        bg[y:y+obj_h, x:x+obj_w] = blended

        total_mask[y:y+obj_h, x:x+obj_w] = cv2.bitwise_or(
            roi_mask, (mask > 127).astype(np.uint8) * 255
        )

        return x, y, obj_h, obj_w, poly, True


    return 0, 0, 0, 0, poly, False
    
# --- CONFIGURATION ---
NUM_GENERATED_IMAGES = 200       # Wie viele Bilder insgesamt erstellt werden sollen

OBJS_PER_IMAGE = (1, 6)         # Zufällige Anzahl (Min, Max) an Objekten pro Bild
SCALE_RANGE = (0.2, 0.8)        # Skalierungsfaktor relativ zum Hintergrund (Min, Max)
ROTATION_RANGE = (-20, 20)      # Drehung in Grad
FLIP_PROB = 0.5                 # 50% Chance für horizontales Spiegeln
BRIGHTNESS_RANGE = (0.6, 1.4)   # Helligkeits-Augmentation (0.7 = dunkler, 1.3 = heller)
BLUR_PROB = 0.2                 # Chance für leichte Unschärfe (fokussiert vs. unfokussiert)

# --- OUTDOOR AUGMENTATION CONFIG ---
MOTION_BLUR_PROB = 0.1
SHADOW_PROB = 0.15
COLOR_TEMP_PROB = 0.15
CONTRAST_PROB = 0.15
JPEG_ARTIFACT_PROB = 0.1

# --- BACKGROUND AUGMENTATION CONFIG ---
BG_DISTRIBUTION = {
    "target_context": 0.8,          # Anteil von Hintergründen, die zum Zielkontext passen (z.B. Straßen, Parkplätze)
    "outside_context": 0.2,         # Anteil von Hintergründen, die nicht zum Zielkontext passen (z.B. Innenräume, Natur)
}
OBJECT_DISTRIBUTION = {
    "target_context": 0.85,          # Anteil BG_DISTRIBUTION von Objekten im Zielkontext
    "outside_context": 0.15,         # Anteil von BG_DISTRIBUTION von Objekten im Außenkontext
}

BG_SHADOW_PROB = 0.15
BG_LIGHT_PROB = 0.10
BG_COLOR_TEMP_PROB = 0.10
BG_CONTRAST_PROB = 0.10
BG_JPEG_ARTIFACT_PROB = 0.10
BG_HAZE_PROB = 0.05


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
    os.makedirs(f"dataset/images/{split}", exist_ok=True)
    os.makedirs(f"dataset/labels/{split}", exist_ok=True)

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

def scale_object_relative_to_bg(obj, bg_w, bg_h, scale):
    h, w = obj.shape[:2]

    # Objekt ist höher als breit → Höhe dominiert
    if h >= w:
        target_h = scale * bg_h
        factor = target_h / h
    else:
        target_w = scale * bg_w
        factor = target_w / w

    # Mindestgröße absichern
    factor = max(factor, max(1/h, 1/w))

    
    return factor


def augment_object(obj, mask, polygon_local):
    # 1. Skalierung
    scale = scale_object_relative_to_bg(
        obj, 640, 480, random.uniform(*SCALE_RANGE)
    )

    obj = cv2.resize(obj, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)



    # 2. Horizontal Flip
    if random.random() < FLIP_PROB:
        obj = cv2.flip(obj, 1)
        mask = cv2.flip(mask, 1)

    h, w = obj.shape[:2]
    angle = random.uniform(*ROTATION_RANGE)

    # neue Größe berechnen
    rad = np.deg2rad(angle)
    new_w = int(abs(np.sin(rad) * h) + abs(np.cos(rad) * w))
    new_h = int(abs(np.sin(rad) * w) + abs(np.cos(rad) * h))

    # Rotationsmatrix mit Center-Shift
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # ❗ WICHTIG: KEIN Schwarz reinziehen, aber auch KEIN frühzeitiges Mask-Cutting
    obj = cv2.warpAffine(
        obj, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    mask = cv2.warpAffine(
        mask, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 1. Binäre Maske
    mask_bin = (mask > 127).astype(np.uint8)

    # 2. Rand extrahieren
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_eroded = cv2.erode(mask_bin, kernel, iterations=1)
    mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    border = cv2.subtract(mask_dilated, mask_eroded)   # nur Randpixel

    # 3. Schwarze Pixel finden
    obj_vis = obj.astype(np.uint8) if obj.dtype != np.uint8 else obj
    gray = cv2.cvtColor(obj_vis, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < 30).astype(np.uint8)

    # 4. Nur schwarze Randpixel entfernen
    edge_artifacts = cv2.bitwise_and(black_mask, border)
    mask_bin[edge_artifacts == 1] = 0

    # 5. Glätten
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)



    # 6. Helligkeit (wie vorher)
    brightness = random.uniform(*BRIGHTNESS_RANGE)
    obj = cv2.convertScaleAbs(obj * brightness)

    # --- OUTDOOR AUGS NUR AUF OBJ ---

    if random.random() < MOTION_BLUR_PROB:
        k = random.choice([3, 5, 7])
        angle_blur = random.uniform(-10, 10)

        M_blur = cv2.getRotationMatrix2D((k / 2, k / 2), angle_blur, 1)
        kernel = np.diag(np.ones(k))
        kernel = cv2.warpAffine(kernel, M_blur, (k, k))
        kernel = kernel / k

        obj = cv2.filter2D(obj, -1, kernel)

    if random.random() < SHADOW_PROB:
        shadow = np.zeros_like(obj)

        x1, y1 = random.randint(0, max(0, w - 1)), 0
        x2, y2 = random.randint(0, max(0, w - 1)), h

        cv2.line(
            shadow,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            random.randint(50, 150)
        )

        obj = cv2.addWeighted(obj, 1.0, shadow, -0.4, 0)

    if random.random() < COLOR_TEMP_PROB:
        shift = random.randint(-20, 20)
        obj = cv2.add(
            obj,
            np.array([shift, shift // 2, -shift]).astype(np.int8)
        )

    if random.random() < CONTRAST_PROB:
        alpha = random.uniform(0.8, 1.3)
        obj = cv2.convertScaleAbs(obj, alpha=alpha, beta=0)

    # sicherstellen, dass die Maske uint8 ist (0/255)
    mask_for_contours = (mask_bin * 255).astype(np.uint8) if mask_bin.dtype != np.uint8 else mask_bin

    # Konturen aus sauberer Binärmaske holen
    contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) == 0:
        return obj, mask_bin * 255, None

    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt.reshape(-1, 2).astype(np.float32)

    return obj, mask_bin * 255, cnt


def resize_and_center_crop(img, target_width=640, target_height=480):
    h, w = img.shape[:2]
    scale = min(w / target_width, h / target_height)

    start_y = int(max(0, (h - target_height*scale) // 2))
    start_x = int(max(0, (w - target_width*scale) // 2))
    cropped_image = img[start_y: max(h,start_y + int(target_height*scale)), start_x: max(w, start_x + int(target_width*scale))]

    final_img = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)


    return final_img



# Hauptschleife zur Generierung
for i in range(NUM_GENERATED_IMAGES):
    print(f"Generiere Bild {i+1}/{NUM_GENERATED_IMAGES}...")
    put_object_on_image = False
    if random.random() < BG_DISTRIBUTION["target_context"]:
        bg_dir = BG_DIR_TAR
        if random.random() < OBJECT_DISTRIBUTION["target_context"]:
            put_object_on_image = True
    else:
        bg_dir = BG_DIR_OUT
        if random.random() < OBJECT_DISTRIBUTION["outside_context"]:
            put_object_on_image = True

    # Hintergrund wählen und vorbereiten
    bg_name = random.choice(os.listdir(bg_dir))
    bg = cv2.imread(os.path.join(bg_dir, bg_name))
    bg = cv2.imread(os.path.join(bg_dir, bg_name))

    if bg is None:
        print("❌ Hintergrund konnte nicht geladen werden:", bg_name)
        continue

    if bg.shape[0] < 200 or bg.shape[1] < 200:
        print("❌ Hintergrund zu klein oder korrupt:", bg_name, bg.shape)
        continue

    bg = resize_and_center_crop(bg, 640, 480) # Für Querformat

    # Hintergrund augmentieren
    bg = augment_background(bg)

    bg_h, bg_w = bg.shape[:2]
    
    labels_to_save = []

    # 1. Belegungsmaske für dieses Bild initialisieren (alles schwarz)
    total_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    labels_to_save = []

    if put_object_on_image:

        # Zufällige Anzahl an Objekten platzieren
        num_objs = random.randint(*OBJS_PER_IMAGE)
        for _ in range(num_objs):
            # Zufälliges Quellbild wählen
            img_name = random.choice([f for f in os.listdir(os.path.join(OBJECT_DIR, "images")) if f.lower().endswith(VALID_EXT)])
            
            if not img_name.lower().endswith(VALID_EXT):
                continue

            img_path = os.path.join(OBJECT_DIR, "images", img_name)
            label_path = os.path.join(OBJECT_DIR, "labels", img_name.rsplit(".", 1)[0] + ".txt")

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
            if len(ys) == 0 or len(xs) == 0:
                print("❌ Maske leer, Objekt wird übersprungen")
                continue

            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()


            obj_crop = obj[y_min:y_max + 1, x_min:x_max + 1]
            mask_crop = mask[y_min:y_max + 1, x_min:x_max + 1]

            # optional: entferne 1px Rand am Crop, falls Originalbilder Randartefakte haben
            if obj_crop.shape[0] > 2 and obj_crop.shape[1] > 2:
                obj_crop, mask_crop = obj_crop[1:-1, 1:-1], mask_crop[1:-1, 1:-1]


            if obj_crop.size == 0 or mask_crop.size == 0:
                print("❌ Crop ist leer:", obj_crop.shape, mask_crop.shape)
                continue


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
    cv2.imwrite(os.path.join(f"dataset/images/{split}", out_name), bg)

    # Label speichern
    with open(os.path.join(f"dataset/labels/{split}", f"synth_{i}.txt"), "w") as f:
        for c, poly in labels_to_save:
            poly_str = " ".join([f"{coord[0]:.6f} {coord[1]:.6f}" for coord in poly])
            f.write(f"{c} {poly_str}\n")

