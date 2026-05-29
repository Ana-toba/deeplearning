from config import (
    RANGES,
    MODE_MAPPING
)

# =====================================================
# LIMPIAR NUMERO
# =====================================================

def clean_number(text):

    text = text.replace(",", ".")

    if text.count(".") > 1:

        first = text.find(".")

        text = (
            text[:first + 1]
            + text[first + 1:].replace(".", "")
        )

    if text == "":
        return None

    try:
        return float(text)

    except:
        return None


# =====================================================
# VALIDAR VALORES
# =====================================================

def validate_value(
    parameter,
    value
):

    if value is None:
        return None

    low, high = RANGES[parameter]

    if low <= value <= high:
        return value

    # =========================================
    # CORRECCION VOLMINESP
    # =========================================

    if parameter == "volminesp":

        # 104 -> 10.4

        if 100 <= value <= 300:

            corrected = value / 10

            if low <= corrected <= high:

                return corrected

        # 67 -> 6.7

        if 30 < value < 100:

            corrected = value / 10

            if low <= corrected <= high:

                return corrected

        # 1004 -> 10.04

        if 1000 <= value < 10000:

            corrected = value / 100

            if low <= corrected <= high:

                return corrected

    return None


# =====================================================
# FORMATEAR
# =====================================================

def format_value(
    parameter,
    value
):

    if value is None:
        return "--"

    if parameter == "volminesp":

        return (
            f"{value:.1f}"
            .replace(".", ",")
        )

    if isinstance(value, float):

        if value.is_integer():
            return str(int(value))

    return str(value).replace(".", ",")


# =====================================================
# DETECTAR MODO
# =====================================================

def detect_mode(ls_detections):

    for d in ls_detections:

        label = d["label"].lower()

        if label == "asv":
            return "ASV"

        elif label == "simv+":
            return "SIMV+"

        elif label == "pcv+":
            return "PCV+"

        elif label == "psimv+":
            return "PSIMV+"

    return None


# =====================================================
# RECONSTRUIR
# =====================================================

def reconstruct_values(
    rows,
    parameter_order
):

    final_values = {}

    for parameter, row in zip(
        parameter_order,
        rows
    ):

        text = "".join(
            [r["label"] for r in row]
        )

        value = clean_number(text)

        value = validate_value(
            parameter,
            value
        )

        final_values[parameter] = {

            "raw": value,

            "formatted": format_value(
                parameter,
                value
            )
        }

    return final_values


# =====================================================
# PARAMETROS RS
# =====================================================

def get_rs_parameters(
    current_mode
):

    dynamic_parameter = MODE_MAPPING[current_mode]

    return [

        dynamic_parameter,

        "PEEP/CPAP",

        "Oxigeno"
    ]