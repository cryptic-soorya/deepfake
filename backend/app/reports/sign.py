"""Cryptographic signing of generated forensic reports so they can be verified as untampered."""


def sign_report(report_bytes: bytes) -> bytes:
    raise NotImplementedError
