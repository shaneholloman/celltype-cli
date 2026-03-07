"""Tests for the CellType Cloud client — mocked gateway API."""

from unittest.mock import patch, MagicMock
import pytest


class TestCloudClient:
    """Test cloud client job submission and polling."""

    def test_get_balance(self):
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"balance": 8.50}
        mock_resp.raise_for_status = MagicMock()

        with patch("ct.cloud.client.httpx.Client") as MockClient:
            mock_http = MagicMock()
            mock_http.get.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_http)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            balance = client.get_balance("test-token")
            assert balance == 8.50

    def test_submit_insufficient_credits(self):
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")

        # Mock balance check returning low balance
        with patch.object(client, "get_balance", return_value=0.03):
            result = client.submit_and_wait(
                tool_name="structure.esmfold",
                gpu_profile="structure",
                estimated_cost=0.10,
                token="test-token",
                sequence="MKWVTF",
            )
            assert result["skipped"] is True
            assert result["reason"] == "insufficient_credits"


class TestCloudClientIntegration:
    """Integration test — full flow with mocked gateway."""

    def test_submit_poll_complete(self):
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")

        # Mock the full flow: balance check, submit, poll
        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": "job-123", "estimated_cost": 0.05}
        submit_resp.raise_for_status = MagicMock()

        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {
            "status": "completed",
            "result": {"summary": "Structure predicted", "pdb_content": "ATOM..."},
            "actual_cost": 0.04,
            "balance": 9.96,
        }
        status_resp.raise_for_status = MagicMock()

        with patch.object(client, "get_balance", return_value=10.0):
            with patch("ct.cloud.client.httpx.Client") as MockClient:
                mock_http = MagicMock()
                mock_http.post.return_value = submit_resp
                mock_http.get.return_value = status_resp
                MockClient.return_value.__enter__ = MagicMock(return_value=mock_http)
                MockClient.return_value.__exit__ = MagicMock(return_value=False)

                with patch("ct.cloud.client.time.sleep"):
                    result = client.submit_and_wait(
                        tool_name="structure.esmfold",
                        gpu_profile="structure",
                        estimated_cost=0.05,
                        token="test-token",
                        sequence="MKWVTF",
                    )
                    assert result["summary"] == "Structure predicted"

    def test_submit_auto_proceeds(self):
        """Verify no approval prompt — jobs auto-proceed."""
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": "job-456", "estimated_cost": 0.05}
        submit_resp.raise_for_status = MagicMock()

        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {
            "status": "completed",
            "result": {"summary": "Done"},
            "actual_cost": 0.03,
        }
        status_resp.raise_for_status = MagicMock()

        with patch.object(client, "get_balance", return_value=10.0):
            with patch("ct.cloud.client.httpx.Client") as MockClient:
                mock_http = MagicMock()
                mock_http.post.return_value = submit_resp
                mock_http.get.return_value = status_resp
                MockClient.return_value.__enter__ = MagicMock(return_value=mock_http)
                MockClient.return_value.__exit__ = MagicMock(return_value=False)

                with patch("ct.cloud.client.time.sleep"):
                    # Should NOT call input() at all
                    with patch("builtins.input", side_effect=AssertionError("input() should not be called")):
                        result = client.submit_and_wait(
                            tool_name="structure.esmfold",
                            gpu_profile="structure",
                            estimated_cost=0.05,
                            token="test-token",
                            sequence="MKWVTF",
                        )
                        assert result["summary"] == "Done"
