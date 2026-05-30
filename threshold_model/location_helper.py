import json
import os

LOCATION_FILE = "location.json"

def get_current_location_text():
    if os.path.exists(LOCATION_FILE):
        try:
            with open(LOCATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "city": data.get("city", "Browser Detected"),
                "region": data.get("region", ""),
                "country": data.get("country", "India"),
                "coordinates": data.get("coordinates", ""),
                "map_link": data.get("map_link", "")
            }
        except Exception:
            pass

    return {
        "city": "Unknown City",
        "region": "",
        "country": "India",
        "coordinates": "",
        "map_link": ""
    }