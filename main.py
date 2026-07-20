import os
import uvicorn
from dashboard.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting AI Threat Surveillance System web server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
