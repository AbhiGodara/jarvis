import logging
import webbrowser
import requests
import geocoder
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from commands.registry import command

logger = logging.getLogger(__name__)


def _get_ip_location() -> dict:
    """Helper to fetch city, region, and country based on external IP geolocation."""
    try:
        ip_resp = requests.get("https://api.ipify.org", timeout=5)
        ip_resp.raise_for_status()
        ip_add = ip_resp.text.strip()
        
        geo_resp = requests.get(f"https://get.geojs.io/v1/ip/geo/{ip_add}.json", timeout=5)
        geo_resp.raise_for_status()
        data = geo_resp.json()
        
        return {
            "city": data.get("city", ""),
            "region": data.get("region", ""),
            "country": data.get("country", ""),
            "latitude": float(data.get("latitude", 0.0)),
            "longitude": float(data.get("longitude", 0.0))
        }
    except Exception as e:
        logger.warning(f"Failed to get location via geojs: {e}. Trying geocoder fallback.")
        try:
            g = geocoder.ip("me")
            if g.latlng:
                return {
                    "city": g.city or "",
                    "region": g.state or "",
                    "country": g.country or "",
                    "latitude": g.latlng[0],
                    "longitude": g.latlng[1]
                }
        except Exception as ex:
            logger.error(f"Fallback geocoder failed: {ex}")
        return {}


@command(keywords=["where is", "how far is", "current location", "where am i", "locate"])
def location_services(text: str) -> str:
    """Handle geolocation, distance calculations, and open map locations."""
    lower_text = text.lower()

    # --- 1. Current Location Query ---
    if any(kw in lower_text for kw in ["current location", "where am i"]):
        loc_data = _get_ip_location()
        if not loc_data or not loc_data.get("city"):
            return "I'm sorry, I was unable to determine your current location."
        
        return (
            f"According to your IP address, you are currently in {loc_data['city']}, "
            f"which is in the region of {loc_data['region']}, in {loc_data['country']}."
        )

    # --- 2. Target Location Query ("where is [place]") ---
    elif "where is" in lower_text or "locate" in lower_text:
        # Extract target place name
        place = ""
        for trigger in ["where is", "locate"]:
            if trigger in lower_text:
                place = lower_text.split(trigger, 1)[-1].strip()
                break
        
        if not place:
            return "Which place would you like me to locate?"

        try:
            # Geocode target
            geolocator = Nominatim(user_agent="jarvis_assistant_geocoder")
            location = geolocator.geocode(place, addressdetails=True)
            
            if not location:
                return f"I couldn't find a location matching {place}."

            target_coords = (location.latitude, location.longitude)
            address = location.raw.get("address", {})
            city = address.get("city", address.get("town", address.get("village", "")))
            state = address.get("state", "")
            country = address.get("country", "")

            # Open maps in browser
            map_url = f"http://www.google.com/maps/place/{place.replace(' ', '+')}"
            webbrowser.open(map_url)

            # Calculate distance from current location
            curr_loc = _get_ip_location()
            distance_str = ""
            if curr_loc and curr_loc.get("latitude"):
                curr_coords = (curr_loc["latitude"], curr_loc["longitude"])
                dist = great_circle(curr_coords, target_coords).kilometers
                distance_str = f" It is approximately {round(dist, 1)} kilometers away from your current location."

            loc_description = f"{place.capitalize()}"
            if city or state or country:
                parts = [p for p in [city, state, country] if p]
                loc_description += f" is in " + ", ".join(parts)

            return f"Opening Google Maps for {place}. {loc_description}.{distance_str}"

        except Exception as e:
            logger.error(f"Geocoding failed for {place}: {e}")
            return f"I ran into an issue locating {place} on the map."

    return "Location command was not understood."
