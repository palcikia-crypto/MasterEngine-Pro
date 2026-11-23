"""Run the MasterEngine web panel:  python run.py [port]"""
import sys

import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"MasterEngine panel → http://127.0.0.1:{port}")
    uvicorn.run("masterengine.web:app", host="127.0.0.1", port=port)
