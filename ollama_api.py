"""
A lightweight Python client for the Ollama local/server API.

Features:
- Basic generate() call (non-streaming) that returns parsed JSON from the API.
- generate_stream() generator to consume streaming responses (SSE or chunked).
- Configurable base_url, timeout, and optional API key header.
- Minimal CLI for quick manual testing.

Note: Ollama's API commonly runs at http://localhost:11434 and exposes /api/generate for model inference. This client is defensive about streaming formats and yields raw chunks or parsed JSON when available.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import requests

__all__ = ["OllamaClient"]

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 60


class OllamaAPIError(RuntimeError):
    pass


class OllamaClient:
    """Simple client for interacting with an Ollama server.

    Example:
        client = OllamaClient()
        resp = client.generate("llama2", "Tell me a joke.")

    Streaming usage:
        for chunk in client.generate_stream("llama2", "Say hello repeatedly"):
            print(chunk)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout

        if api_key:
            # Ollama may support an authorization header if configured behind a proxy
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        # Default to JSON responses
        self.session.headers.setdefault("Content-Type", "application/json")

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def generate(
        self,
        model: str,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call the Ollama /api/generate endpoint and return parsed JSON.

        This method makes a blocking request and returns the full result.
        """
        url = self._endpoint("/api/generate")
        payload: Dict[str, Any] = {"model": model, "prompt": prompt}

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop

        payload.update(kwargs)

        logger.debug("POST %s payload=%s", url, payload)

        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Request to Ollama failed")
            raise OllamaAPIError(f"request failed: {exc}") from exc

        try:
            return resp.json()
        except ValueError:
            # Non-JSON response
            raise OllamaAPIError("non-JSON response from Ollama server")

    def generate_stream(
        self,
        model: str,
        prompt: Union[str, List[str]],
        **kwargs: Any,
    ) -> Generator[Union[str, Dict[str, Any]], None, None]:
        """Stream a response from the Ollama /api/generate endpoint.

        Yields raw text chunks or parsed JSON objects when lines are JSON.
        This function is defensive: it supports SSE-like lines (e.g. starting with "data: ")
        and plain chunked text. Consumers should handle both strings and dicts.
        """
        url = self._endpoint("/api/generate")
        payload: Dict[str, Any] = {"model": model, "prompt": prompt}
        payload.update(kwargs)

        logger.debug("POST %s (stream) payload=%s", url, payload)

        try:
            with self.session.post(url, json=payload, stream=True, timeout=self.timeout) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue

                    # Ollama may use SSE style lines like "data: {...}"
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        # Some SSE streams send a single "[DONE]" or similar sentinel
                        if data in ("[DONE]", "done"):
                            break

                        try:
                            yield json.loads(data)
                        except Exception:
                            # If parsing fails, return the raw data string
                            yield data
                    else:
                        # Try to parse JSON chunk, otherwise yield as text
                        try:
                            yield json.loads(line)
                        except Exception:
                            yield line
        except requests.RequestException as exc:
            logger.exception("Streaming request to Ollama failed")
            raise OllamaAPIError(f"streaming request failed: {exc}") from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Ollama API client CLI")
    parser.add_argument("model", help="Model name to use")
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama server base URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = OllamaClient(base_url=args.base_url)

    if args.stream:
        for chunk in client.generate_stream(args.model, args.prompt):
            # print raw chunk or JSON prettified
            if isinstance(chunk, dict):
                print(json.dumps(chunk, ensure_ascii=False))
            else:
                print(chunk)
    else:
        resp = client.generate(args.model, args.prompt)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
