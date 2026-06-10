"""Заполнение БД тестовыми профилями через API."""

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_data import test_users

BASE_URL = "http://127.0.0.1:8000"


def main():
    for user in test_users:
        response = requests.post(f"{BASE_URL}/users", json=user, timeout=30)
        if response.status_code == 200:
            print(f"OK: {user['user_id']}")
        else:
            print(f"FAIL: {user['user_id']} -> {response.text}")


if __name__ == "__main__":
    main()
