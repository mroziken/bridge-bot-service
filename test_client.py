import argparse
import json
import time
import sys
from pathlib import Path

import requests


DEFAULT_URL = "http://localhost:8080/bot/decision"


def load_request(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def send_request(url: str, payload: dict) -> dict:
    headers = {
        "Content-Type": "application/json"
    }

    start = time.perf_counter()

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )

    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nHTTP status: {response.status_code}")
    print(f"Latency: {elapsed:.2f} ms")

    try:
        data = response.json()
    except Exception:
        print("Response is not JSON:")
        print(response.text)
        sys.exit(1)

    return data


def validate_response(resp: dict):
    if "error" in resp and resp["error"] is not None:
        print("\nService returned error:")
        print(json.dumps(resp["error"], indent=2))
        return

    if "decision" not in resp:
        print("\nInvalid response: missing decision")
        return

    decision = resp["decision"]

    if decision["type"] == "BID":
        print(f"\nBot bid: {decision['call']}")

    elif decision["type"] == "PLAY":
        print(f"\nBot played: {decision['card']}")

    else:
        print("\nUnknown decision type")

    if resp.get("meta"):
        print("\nMeta:")
        print(json.dumps(resp["meta"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Bridge Bot Service Test Client")

    parser.add_argument(
        "file",
        type=str,
        help="Path to JSON request file"
    )

    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Bot service URL"
    )

    args = parser.parse_args()

    file_path = Path(args.file)

    print(f"Loading request from: {file_path}")
    request_payload = load_request(file_path)

    print("\nRequest:")
    print(json.dumps(request_payload, indent=2))

    print("\nSending request...")

    response = send_request(args.url, request_payload)

    print("\nResponse:")
    print(json.dumps(response, indent=2))

    validate_response(response)


if __name__ == "__main__":
    main()