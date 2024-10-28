SETTINGS_DEFAULT = {
    "IMX_CONFIG_SSDMOBILENET": {
        "NETWORK_FPK_PATH": "networks/imx500_network_yolov8n_pose.fpk",
        "SWAP_TENSORS": False,
        "LABELS": ["Person"],
    },
    "FRAME_RATE": 30,
    "PICAMERA_CONTROLS": {"FrameRate": 30},
    "IMX_CONFIG_SELECTOR": "IMX_CONFIG_SSDMOBILENET",
    "BYTE_TRACKER": {
        "TRACK_THRESH": 0.60,
        "TRACK_BUFFER": 240,
        "MATCH_THRESH": 0.6,
        "ASPECT_RATIO_THRESH": 3.0,
        "MIN_BOX_AREA": 1.0,
        "MOT20": False,
    },
    "CONFIDENCE_THRESHOLD": 0.6,
    "OVERLAP_DETECTION": {"OVERLAP_BOX_SIZE_PX": 20, "MODIFIED_BBOX": False},
    "EVENT_DETECTION": {"MAX_MISSING_OVERLAP": 120, "MAX_MISSING_TRACKER": 60, "MIN_OVERLAP_THRESHOLD": 0.5, "HYSTERESIS": 0.4},
    "CLOUD": False,
    "SAVE_IMAGE": False,
    "EMIT_DETAILED_RESULT": True,
    "MINIMUM_UPTIME": 15,
    "ANNOTATION": {"BYTE_TRACKER": False, "OBJECT_OVERLAY": False, "OBJECT_EVENT": True},
}


an_colors = [["#ff00ff", "#002200", "#000066"], ["#ff00ff", "#002200", "#000066"]]
