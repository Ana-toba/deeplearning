# utils.py

from config import NUMERIC_CLASSES
import cv2

# =====================================================
# ORDENAR
# =====================================================

def sort_left_to_right(items):

    return sorted(
        items,
        key=lambda x: x["x1"]
    )

# =====================================================
# EXTRAER REGION
# =====================================================

def extract_region(
    model,
    image,
    region_name
):

    results = model(
        image,
        verbose=False
    )

    for box in results[0].boxes:

        cls = int(box.cls[0])

        class_name = model.names[cls]

        if class_name.lower() == region_name.lower():

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = image[
                y1:y2,
                x1:x2
            ]

            return {

                "crop": crop,

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "confidence": float(
                    box.conf[0]
                )
            }

    return None

# =====================================================
# AGRUPAR NUMEROS POR FILAS
# =====================================================

def group_numbers_by_rows(
    detections,
    y_tolerance=10
):

    numeric_detections = []

    # =====================================
    # SOLO NUMERICOS
    # =====================================

    for d in detections:

        if d["label"] in NUMERIC_CLASSES:

            center_y = (
                d["y1"] + d["y2"]
            ) / 2

            d["center_y"] = center_y

            numeric_detections.append(d)

    # =====================================
    # ORDENAR VERTICALMENTE
    # =====================================

    numeric_detections.sort(
        key=lambda x: x["center_y"]
    )

    rows = []

    # =====================================
    # AGRUPAR FILAS
    # =====================================

    for det in numeric_detections:

        added = False

        for row in rows:

            row_y = sum(
                r["center_y"]
                for r in row
            ) / len(row)

            # =================================
            # MISMA FILA
            # =================================

            if abs(
                det["center_y"] - row_y
            ) < y_tolerance:

                duplicate = False

                # =============================
                # EVITAR DUPLICADOS
                # =============================

                for existing in row:

                    dx = abs(
                        existing["x1"] - det["x1"]
                    )

                    dy = abs(
                        existing["y1"] - det["y1"]
                    )

                    same_label = (

                        existing["label"]
                        == det["label"]
                    )

                    if (

                        dx < 10
                        and
                        dy < 10
                        and
                        same_label
                    ):

                        duplicate = True
                        break

                if not duplicate:

                    row.append(det)

                added = True

                break

        # =====================================
        # NUEVA FILA
        # =====================================

        if not added:

            rows.append([det])

    # =====================================
    # ORDENAR IZQ -> DER
    # =====================================

    for row in rows:

        row.sort(
            key=lambda x: x["x1"]
        )

    # =====================================
    # DEBUG
    # =====================================

    print("\nROWS:")

    for i, row in enumerate(rows):

        text = "".join(
            [r["label"] for r in row]
        )

        print(
            f"ROW {i}: {text}"
        )

    return rows

# =====================================================
# DIBUJAR DETECCIONES
# =====================================================

def draw_detections(
    image,
    detections,
    window_name
):

    draw_image = image.copy()

    for d in detections:

        x1 = d["x1"]
        y1 = d["y1"]
        x2 = d["x2"]
        y2 = d["y2"]

        label = d["label"]

        cv2.rectangle(
            draw_image,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.putText(
            draw_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow(
        window_name,
        draw_image
    )