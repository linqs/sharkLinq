"""Unit tests for frontend common response wrapping and route building."""

import http
import unittest

import frontend.common


class TestCommon(unittest.TestCase):
    def test_wrap_api_response_success(self):
        """Test wrapping a successful API response."""
        data = {"key": "value"}
        code = http.HTTPStatus.OK
        endpoint = "test/endpoint"

        response = frontend.common.wrap_api_response(data, code, endpoint)

        self.assertTrue(response["success"])
        self.assertEqual(response["content"], data)
        self.assertEqual(response["endpoint"], endpoint)
        self.assertIsNone(response["message"])
        self.assertIn("server-version", response)

    def test_wrap_api_response_failure(self):
        """Test wrapping a failed API response."""
        code = http.HTTPStatus.BAD_REQUEST
        endpoint = "test/fail"
        message = "Something went wrong"

        response = frontend.common.wrap_api_response(None, code, endpoint, message=message, success=False)

        self.assertFalse(response["success"])
        self.assertEqual(response["message"], message)
        self.assertIsNone(response["content"])

    def test_build_api_route(self):
        """Test building an API route configuration."""
        def dummy_handler(handler, path, **kwargs):
            return {}, http.HTTPStatus.OK, None

        endpoint = "test/route"
        regex, handler, auth = frontend.common.build_api_route(endpoint, dummy_handler, token_required=True)

        self.assertEqual(regex, r'^/api/test/route$')
        self.assertIsNotNone(handler)
        self.assertIsNotNone(auth)

        # Test without auth.
        regex_no_auth, handler_no_auth, auth_no_auth = frontend.common.build_api_route(endpoint, dummy_handler, token_required=False)
        self.assertIsNone(auth_no_auth)
