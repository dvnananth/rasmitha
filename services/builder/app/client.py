from __future__ import annotations

import os
from typing import Any

import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class ApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or API_BASE_URL
        self._client = httpx.Client(base_url=self.base_url, timeout=10.0)

    def list_templates(self) -> list[dict[str, Any]]:
        r = self._client.get("/templates")
        r.raise_for_status()
        return r.json()

    def create_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/templates", json=payload)
        r.raise_for_status()
        return r.json()

    def list_modules(self) -> list[dict[str, Any]]:
        r = self._client.get("/modules")
        r.raise_for_status()
        return r.json()

    def create_module(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/modules", json=payload)
        r.raise_for_status()
        return r.json()

    def list_views(self) -> list[dict[str, Any]]:
        r = self._client.get("/views")
        r.raise_for_status()
        return r.json()

    def create_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/views", json=payload)
        r.raise_for_status()
        return r.json()

    def list_actions(self) -> list[dict[str, Any]]:
        r = self._client.get("/actions")
        r.raise_for_status()
        return r.json()

    def create_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/actions", json=payload)
        r.raise_for_status()
        return r.json()

    def list_connections(self) -> list[dict[str, Any]]:
        r = self._client.get("/connections")
        r.raise_for_status()
        return r.json()

    def create_connection(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/connections", json=payload)
        r.raise_for_status()
        return r.json()

