"""
Robust, standalone launcher for LocalGPT Pro.
Bypasses Streamlit credential checks and permission locks on Windows.
Waits until the server is actively listening on port 8501 before launching browser.
"""
import os
import sys
import socket
import webbrowser
import threading
import time

# Guarantee working directory is LocalGPT-Pro
DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)
if DIR not in sys.path:
    sys.path.insert(0, DIR)

# Streamlit environment variables
home_dir = os.path.abspath(os.path.join(DIR, "..", ".home"))
os.environ["USERPROFILE"] = home_dir
os.environ["HOME"] = home_dir
os.environ["STREAMLIT_SERVER_SHOW_EMAIL_PROMPT"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
os.environ["PYTHONUNBUFFERED"] = "1"

# Neutralize credentials check to prevent Windows ~/.streamlit PermissionErrors
try:
    import streamlit.runtime.credentials as c
    c.check_credentials = lambda: None
except Exception:
    pass

def open_browser_when_ready():
    """Polls port 8501 and only launches browser once server is actively listening."""
    start_time = time.time()
    while time.time() - start_time < 45:
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", 8501), timeout=0.5):
                print("\n[LocalGPT Pro] Server is ready! Opening browser at http://127.0.0.1:8501 ...", flush=True)
                time.sleep(0.3)
                webbrowser.open("http://127.0.0.1:8501")
                return
        except (OSError, ConnectionRefusedError):
            continue
    # Fallback if connection check timed out
    try:
        webbrowser.open("http://127.0.0.1:8501")
    except Exception:
        pass

from streamlit.web import cli as stcli

if __name__ == "__main__":
    app_path = os.path.join(DIR, "app.py")
    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    
    # Valid CLI arguments for Streamlit run
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--server.headless=false"
    ]
    stcli.main()
