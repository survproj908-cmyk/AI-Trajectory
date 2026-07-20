import os
import uvicorn
from dashboard.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting dashboard server from root main.py on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
