import requests
import argparse
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.getcwd())

# Optional: Import TestClient for local-only mode
try:
    from fastapi.testclient import TestClient
    from src.main import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

def predict_remote(url, text):
    """Test against a running server."""
    endpoint = f"{url.rstrip('/')}/api/v1/classify"
    try:
        response = requests.post(endpoint, json={"text": text})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def predict_local(text):
    """Test using TestClient (no server process needed)."""
    if not HAS_FASTAPI:
        return {
            "error": "FastAPI/src not found in path. Cannot run local mode."
        }

    with TestClient(app) as client:
        response = client.post("/api/v1/classify", json={"text": text})
        return response.json()

def main():
    parser = argparse.ArgumentParser(
        description="Ogos AI Urgent Classifier Testing CLI"
    )
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Remote API base URL"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local TestClient (default if no URL provided "
        + "and server not running)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Add project root to sys.path if running locally
    if args.local or args.interactive:
        sys.path.append(os.getcwd())

    if args.interactive:
        print("=== PUPT-OGOS AI Urgency Classifier Interactive Mode ===")
        print("Type 'exit' or 'quit' to stop.")
        while True:
            text = input("\nEnter student concern: ").strip()
            if text.lower() in ['exit', 'quit']:
                break
            if not text:
                continue

            if args.local:
                result = predict_local(text)
            else:
                result = predict_remote(args.url, text)

            print(json.dumps(result, indent=2))
    elif args.text:
        if args.local:
            result = predict_local(args.text)
        else:
            result = predict_remote(args.url, args.text)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
