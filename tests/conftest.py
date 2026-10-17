"""Reference-dependent tests must never read or alter deployed source indexes."""
import pytest


@pytest.fixture(autouse=True)
def isolated_reference_indexes(tmp_path,monkeypatch):
    monkeypatch.setenv('BEACON_DB',str(tmp_path/'training-references.sqlite'))
    monkeypatch.setenv('BEACON_NCCL_DB',str(tmp_path/'nccl-references.sqlite'))
    monkeypatch.setenv('CONFLUENCE_EVIDENCE_DB',str(tmp_path/'confluence-evidence.sqlite'))
