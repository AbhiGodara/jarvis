import math
import logging
import psutil
from commands.registry import command

logger = logging.getLogger(__name__)


def _convert_size(size_bytes: int) -> str:
    """Helper to convert bytes to human-readable size."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    return f"{s} {size_name[i]}"


@command(keywords=["system stats", "cpu usage", "ram usage", "battery level", "system status"])
def get_system_stats(text: str) -> str:
    """Retrieve CPU utilization, memory usage, and battery statistics."""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        
        # Memory info
        mem = psutil.virtual_memory()
        mem_used = _convert_size(mem.used)
        mem_total = _convert_size(mem.total)
        mem_percent = mem.percent

        # Battery info (gracefully handle desktops without batteries)
        battery = psutil.sensors_battery()
        if battery:
            battery_percent = battery.percent
            plugged_in = "plugged in" if battery.power_plugged else "discharging"
            battery_str = f"and the battery is at {battery_percent} percent, currently {plugged_in}."
        else:
            battery_str = "and there is no battery detected."

        response = (
            f"Currently, CPU usage is at {cpu} percent. "
            f"RAM usage is at {mem_percent} percent, with {mem_used} in use out of {mem_total}, "
            f"{battery_str}"
        )
        logger.info(f"System stats fetched: CPU={cpu}%, RAM={mem_percent}%")
        return response

    except Exception as e:
        logger.error(f"Failed to fetch system stats: {e}")
        return "I was unable to retrieve the system statistics."
