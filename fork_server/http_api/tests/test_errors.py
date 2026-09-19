# coding: utf-8

from __future__ import annotations

import unittest

from fork_server.http_api.errors import (
    error_type_for_status,
    openai_error_payload,
    validation_error_message,
)


class FakeValidationError:
    def __init__(self, errors):
        self._errors = errors

    def errors(self):
        return self._errors


class HttpErrorPayloadTest(unittest.TestCase):
    def test_error_type_for_status_uses_openai_categories(self) -> None:
        self.assertEqual(error_type_for_status(400), "invalid_request_error")
        self.assertEqual(error_type_for_status(401), "authentication_error")
        self.assertEqual(error_type_for_status(429), "rate_limit_error")
        self.assertEqual(error_type_for_status(500), "server_error")
        self.assertEqual(error_type_for_status(504), "timeout_error")

    def test_openai_error_payload_shape(self) -> None:
        payload = openai_error_payload(
            message="Missing file",
            status_code=400,
            param="file",
            code="missing_file",
        )

        self.assertEqual(payload["error"]["message"], "Missing file")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "file")
        self.assertEqual(payload["error"]["code"], "missing_file")

    def test_validation_error_message_extracts_field_param(self) -> None:
        message, param = validation_error_message(
            FakeValidationError(
                [{"loc": ("body", "file"), "msg": "Field required"}]
            )
        )

        self.assertEqual(message, "Invalid request: file: Field required")
        self.assertEqual(param, "file")

    def test_validation_error_message_handles_empty_errors(self) -> None:
        message, param = validation_error_message(FakeValidationError([]))

        self.assertEqual(message, "Invalid request")
        self.assertIsNone(param)


if __name__ == "__main__":
    unittest.main()
