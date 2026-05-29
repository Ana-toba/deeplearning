# config.py

LS_CLASS_MAP = {
    0: '0',   
    1: '1', 
    2: '2', 
    3: '3', 
    4: '4', 
    5: '5', 
    6: '6', 
    7: '7', 
    8: '8', 
    9: '9', 
    10: 'asv', 
    11: 'person', 
    12: 'ppico', 
    13: 'cmH2O', 
    14: 'coma', 
    15: 'volminesp', 
    16: 'l/min', 
    17: 'vte', 
    18: 'ml', 
    19: 'ftotal', 
    20: 'c/min', 
    21: 'spo2', 
    22: 'pcv+', 
    23: 'simv+', 
    24: 'psimv+'
}


RS_CLASS_MAP = {

    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "modo",
    11: "porcentaje",
    12: "volmin",
    13: "vt",
    14: "cmH2O",
    15: "peep",
    16: "oxigeno",
    17: "controles",
    18: "alarmas",
    19: "iconos"
}

# modo rs c1

MODE_MAPPING = {

    "ASV": "%VolMin",
    "SIMV+": "Vt",
    "PCV+": "Pcontrol",
    "PSIMV+": "Pinsp"
}

NUMERIC_CLASSES = ["0","1","2","3","4","5","6","7","8","9",","]

# rangos

RANGES = {

    "ppico": (0, 80),
    "volminesp": (0, 30),
    "vte": (0, 2000),
    "ftotal": (0, 80),
    "%VolMin": (0, 400),
    "PEEP/CPAP": (0, 30),
    "Oxigeno": (21, 100),
    "Vt": (0, 2000),
    "Pcontrol": (0, 80),
    "Pinsp": (0, 80)
}

# modelos

LS_MODEL_PATH = (
    "project/models/modelo_lsf/weights/best.pt"
)

RS_MODEL_PATH = (
    "project/models/modelo_rsf/weights/best.pt"
)

GLOBAL_MODEL_PATH = (
    "project/models/modelo_final/weights/best.pt"
)