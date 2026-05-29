from ultralytics import YOLO
import cv2, requests, depthai as dai, base64, time

from config import LS_CLASS_MAP, RS_CLASS_MAP, GLOBAL_MODEL_PATH, LS_MODEL_PATH, RS_MODEL_PATH
from utils import extract_region, group_numbers_by_rows
from postprocess import reconstruct_values, get_rs_parameters, detect_mode

API_URL = "http://127.0.0.1:5000/Mediciones/create"

global_model = YOLO(GLOBAL_MODEL_PATH)
ls_model = YOLO(LS_MODEL_PATH)
rs_model = YOLO(RS_MODEL_PATH)

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)

cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setVideoSize(1920,1080)
cam.setFps(30)
cam.setInterleaved(False)

q_video = cam.video.createOutputQueue(maxSize=4,blocking=False)

pipeline.start()

current_mode = "ASV"

print("OAK-D iniciada correctamente")

def image_to_base64(img):
    if img is None: return None
    _, buffer = cv2.imencode(".jpg",img)
    return base64.b64encode(buffer).decode("utf-8")

try:
    requests.delete("http://127.0.0.1:5000/Mediciones/clear",timeout=5)
    print("Mediciones antiguas eliminadas")
except Exception as e:
    print("No se pudo limpiar BD:",e)


while True:

    frame_start = time.time()

    # =================================================
    # LEER FRAME OAK-D
    # =================================================

    msg = q_video.get()

    if msg is None:

        continue

    frame = msg.getCvFrame()

    display_frame = frame.copy()

    # =================================================
    # VARIABLES
    # =================================================

    ls_values = None
    rs_values = None
    gs_base64 = None
    gi_base64 = None
    ginf_base64 = None

    # =================================================
    # LS
    # =================================================

    ls_data = extract_region(global_model,frame,"LS")

    if ls_data is not None:

        ls_crop = ls_data["crop"]
        results = ls_model(ls_crop,verbose=False)
        ls_detections = []

        for box in results[0].boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            if conf < 0.5:

                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            detection = {

                "label":
                    LS_CLASS_MAP[cls],

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2
            }

            ls_detections.append(
                detection
            )

            cv2.rectangle(

                ls_crop,

                (x1, y1),

                (x2, y2),

                (0,255,0),

                2
            )

            cv2.putText(

                ls_crop,

                LS_CLASS_MAP[cls],

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0,255,0),

                2
            )
        
        # =================================================
        # GS
        # =================================================

        gs_data = extract_region(

            global_model,

            frame,

            "GS"
        )

        if gs_data is not None:
            

            gs_crop = gs_data["crop"]
            gs_crop = cv2.resize(gs_crop,(600, 120))
            gs_base64 = image_to_base64(
                gs_crop
            )

            cv2.imshow(
                "GS",
                gs_crop
            )

        # =================================================
        # GI
        # =================================================

        gi_data = extract_region(

            global_model,

            frame,

            "GI"
        )

        if gi_data is not None:

            gi_crop = gi_data["crop"]
            gi_crop = cv2.resize(gi_crop,(600, 120))

            gi_base64 = image_to_base64(
                gi_crop
            )

            cv2.imshow(
                "GI",
                gi_crop
            )

        # =================================================
        # GINF
        # =================================================

        ginf_data = extract_region(

            global_model,

            frame,

            "GInf"
        )

        if ginf_data is not None:

            ginf_crop = ginf_data["crop"]
            ginf_crop = cv2.resize(ginf_crop,(600, 120))

            ginf_base64 = image_to_base64(
                ginf_crop
            )

            cv2.imshow(
                "GInf",
                ginf_crop
            )

        # =========================================
        # DETECTAR MODO
        # =========================================

        detected_mode = detect_mode(
            ls_detections
        )

        if detected_mode is not None:

            current_mode = detected_mode

        # =========================================
        # FILAS
        # =========================================

        ls_rows = group_numbers_by_rows(

            ls_detections,

            y_tolerance=25
        )

        # =========================================
        # DEBUG
        # =========================================

        print("\nROWS LS:")

        for i, row in enumerate(ls_rows):

            text = "".join(
                [r["label"] for r in row]
            )

            print(
                f"ROW {i}: {text}"
            )

        # =========================================
        # PARAMETROS LS
        # =========================================

        ls_parameters = [

            "ppico",

            "volminesp",

            "vte",

            "ftotal"
        ]

        # =========================================
        # RECONSTRUIR
        # =========================================

        ls_values = reconstruct_values(

            ls_rows,

            ls_parameters
        )

        print("\nLS:")

        print(
            f"Modo detectado: {current_mode}"
        )

        for k, v in ls_values.items():

            print(
                f"{k}: {v['formatted']}"
            )

        # =========================================
        # MOSTRAR TEXTO
        # =========================================

        y_text = 30

        for k, v in ls_values.items():

            text = (
                f"{k}: {v['formatted']}"
            )

            cv2.putText(

                display_frame,

                text,

                (20, y_text),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0,255,0),

                2
            )

            y_text += 30

        cv2.putText(

            display_frame,

            f"Modo: {current_mode}",

            (20, 160),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0,255,255),

            2
        )

        cv2.imshow(
            "LS Detection",
            ls_crop
        )

    # =================================================
    # RS
    # =================================================

    rs_data = extract_region(

        global_model,

        frame,

        "RS"
    )

    if rs_data is not None:

        rs_crop = rs_data["crop"]

        results = rs_model(

            rs_crop,

            verbose=False
        )

        rs_detections = []

        for box in results[0].boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            if conf < 0.55:

                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            detection = {

                "label":
                    RS_CLASS_MAP[cls],

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2
            }

            rs_detections.append(
                detection
            )

            cv2.rectangle(

                rs_crop,

                (x1, y1),

                (x2, y2),

                (255,0,0),

                2
            )

            cv2.putText(

                rs_crop,

                RS_CLASS_MAP[cls],

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255,0,0),

                2
            )

        rs_rows = group_numbers_by_rows(

            rs_detections,

            y_tolerance=25
        )

        print("\nROWS RS:")

        for i, row in enumerate(rs_rows):

            text = "".join(
                [r["label"] for r in row]
            )

            print(
                f"ROW {i}: {text}"
            )

        rs_parameters = get_rs_parameters(
            current_mode
        )

        rs_values = reconstruct_values(

            rs_rows,

            rs_parameters
        )

        print("\nRS:")

        for k, v in rs_values.items():

            print(
                f"{k}: {v['formatted']}"
            )

        y_text = 180

        for k, v in rs_values.items():

            text = (
                f"{k}: {v['formatted']}"
            )

            cv2.putText(

                display_frame,

                text,

                (20, y_text),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255,0,0),

                2
            )

            y_text += 30

        cv2.imshow(
            "RS Detection",
            rs_crop
        )

        # =================================================
    # REALTIME + DATABASE
    # =================================================

    if (

        ls_values is not None

        and

        rs_values is not None

        and

        len(rs_values) > 0

    ):

        dynamic_parameter = list(
            rs_values.keys()
        )[0]

        # =========================================
        # DATOS REALTIME (FRONT)
        # =========================================

        realtime_data = {

            "modo":
                current_mode,

            "ppico":
                ls_values.get(
                    "ppico",
                    {"formatted": "--"}
                )["formatted"],

            "volMinEsp":
                ls_values.get(
                    "volminesp",
                    {"formatted": "--"}
                )["formatted"],

            "vte":
                ls_values.get(
                    "vte",
                    {"formatted": "--"}
                )["formatted"],

            "fTotal":
                ls_values.get(
                    "ftotal",
                    {"formatted": "--"}
                )["formatted"],

            "nombreParametro":
                dynamic_parameter,

            "valorParametro":
                rs_values.get(
                    dynamic_parameter,
                    {"formatted": "--"}
                )["formatted"],

            "peep":
                rs_values.get(
                    "PEEP/CPAP",
                    {"formatted": "--"}
                )["formatted"],

            "oxigeno":
                rs_values.get(
                    "Oxigeno",
                    {"formatted": "--"}
                )["formatted"],
            
            "gsImage":
                gs_base64,

            "giImage":
                gi_base64,

            "ginfImage":
                ginf_base64
        }

        print("\nREALTIME:")
        print(realtime_data)
        print(
            "GS:",
            gs_base64 is not None
        )

        print(
            "GI:",
            gi_base64 is not None
        )

        print(
            "GINF:",
            ginf_base64 is not None
        )

        # =========================================
        # ENVIAR AL FRONT
        # =========================================

        try:

            requests.post(

                "http://127.0.0.1:5000/realtime/update",

                json=realtime_data,

                timeout=1
            )

        except Exception as e:

            print(
                "ERROR REALTIME:",
                e
            )

        def safe_float(value):

            if value in [None, "--", ""]:

                return None

            return float(
                str(value).replace(",", ".")
            )

        # =========================================
        # GUARDAR EN BD
        # =========================================

        try:

            measurement_data = {

                "patient_id": 1,

                "ventilator_id": 1,

                "modo": current_mode,

                "Ppico": safe_float(
                    realtime_data["ppico"]
                ),

                "VolMinEsp": safe_float(
                    realtime_data["volMinEsp"]
                ),

                "VTE": safe_float(
                    realtime_data["vte"]
                ),

                "fTotal": safe_float(
                    realtime_data["fTotal"]
                ),

                "parametro_modo": safe_float(
                    realtime_data["valorParametro"]
                ),

                "PEEP": safe_float(
                    realtime_data["peep"]
                ),

                "Oxigeno": safe_float(
                    realtime_data["oxigeno"]
                )
            }

            requests.post(

                "http://127.0.0.1:5000/Mediciones/create",

                json=measurement_data,

                timeout=1
            )
            latency_end = time.time()
            latency_ms = (latency_end -frame_start) * 1000

            print(
                f"Latencia: {latency_ms:.2f} ms"
            )

        except Exception as e:

            print(
                "ERROR GUARDANDO:",
                e
            )
    frame_end = time.time()
    frame_time = (frame_end -frame_start)
    fps = 1 / frame_time
    print(
        f"FPS: {fps:.2f}"
    )
    cv2.imshow(
        "Video",
        display_frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):

        break

cv2.destroyAllWindows()