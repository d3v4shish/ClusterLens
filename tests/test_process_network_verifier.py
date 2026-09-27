from pathlib import Path

import pytest

from scripts import verify_process_network


def test_external_attempt_parser_allows_unix_netlink_and_loopback() -> None:
    trace = "\n".join(
        (
            '1 connect(3, {sa_family=AF_UNIX, sun_path="/tmp/socket"}, 110) = 0',
            "1 sendto(4, NULL, 0, 0, {sa_family=AF_NETLINK, nl_pid=0}, 12) = 0",
            '1 connect(5, {sa_family=AF_INET, sin_port=htons(1), sin_addr=inet_addr("127.0.0.1")}, 16) = 0',
            '1 connect(6, {sa_family=AF_INET6, sin6_port=htons(1), sin6_addr=inet_pton(AF_INET6, "::1")}, 28) = 0',
        )
    )

    assert verify_process_network._external_attempts(trace) == []


def test_external_attempt_parser_rejects_ipv4_and_ipv6_destinations() -> None:
    trace = "\n".join(
        (
            '22 connect(3, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("203.0.113.7")}, 16) = -1',
            '22 sendto(4, "x", 1, 0, {sa_family=AF_INET6, sin6_port=htons(53), sin6_addr=inet_pton(AF_INET6, "2001:db8::2")}, 28) = 1',
        )
    )

    attempts = verify_process_network._external_attempts(trace)

    assert [attempt["address"] for attempt in attempts] == ["203.0.113.7", "2001:db8::2"]
    assert all(attempt["reason"] == "non-loopback IP destination" for attempt in attempts)


def test_external_attempt_parser_ignores_non_outbound_network_calls() -> None:
    trace = (
        '22 recvfrom(3, "x", 1, 0, {sa_family=AF_INET, sin_port=htons(443), '
        'sin_addr=inet_addr("203.0.113.7")}, [16]) = 1'
    )

    assert verify_process_network._external_attempts(trace) == []


def test_external_attempt_parser_fails_closed_on_unknown_ip_format() -> None:
    trace = "22 connect(3, {sa_family=AF_INET, sin_port=htons(443), sin_addr=UNKNOWN}, 16) = -1"

    attempts = verify_process_network._external_attempts(trace)

    assert len(attempts) == 1
    assert attempts[0]["address"] is None
    assert attempts[0]["reason"] == "unparseable IP destination"


def test_network_audit_report_must_be_outside_repository() -> None:
    inside = verify_process_network.REPO_ROOT / "ignored-network-audit"
    with pytest.raises(ValueError, match="outside the repository"):
        verify_process_network._require_external_report_dir(inside)


def test_missing_tracer_is_not_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify_process_network.shutil, "which", lambda _name: None)
    report_dir = tmp_path / "report"

    exit_code = verify_process_network.main(
        ["--report-dir", str(report_dir), "--", "bash", "scripts/test.sh"]
    )

    assert exit_code == verify_process_network.EXIT_NOT_RUN
    payload = (report_dir / "process_network.json").read_text(encoding="utf-8")
    assert '"status": "NOT_RUN"' in payload
