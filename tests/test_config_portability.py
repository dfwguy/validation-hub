from pathlib import Path
import pytest
from backend import config
from backend.confluence_client import ConfluenceReader


def test_portable_paths_inherit_one_runtime_root(tmp_path, monkeypatch):
    monkeypatch.setenv('VALIDATION_DATA_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.delenv('VALIDATION_UPLOADS', raising=False)
    monkeypatch.delenv('CONFLUENCE_EVIDENCE_DIR', raising=False)
    assert config.uploads_root() == tmp_path / 'runtime/uploads'
    assert config.evidence_root() == tmp_path / 'runtime/atlassian-evidence'
    assert config.data_path('VALIDATION_DB', 'catalog/validation.sqlite') == tmp_path / 'runtime/catalog/validation.sqlite'
    monkeypatch.setenv('VALIDATION_UPLOADS', str(tmp_path / 'separate-upload-disk'))
    assert config.uploads_root() == tmp_path / 'separate-upload-disk'
    monkeypatch.setenv('VALIDATION_UPLOADS', '')
    assert config.uploads_root() == tmp_path / 'runtime/uploads'


def test_confluence_requires_operator_selected_tenant(tmp_path, monkeypatch):
    monkeypatch.delenv('ATLASSIAN_CLOUD_ID', raising=False)
    token = tmp_path / 'token'
    token.write_text('synthetic-unit-test-token')
    token.chmod(0o600)
    with pytest.raises(ValueError, match='ATLASSIAN_CLOUD_ID'):
        ConfluenceReader('operator@example.test', token)
    reader = ConfluenceReader('operator@example.test', token, cloud_id='00000000-0000-0000-0000-000000000000')
    assert reader.base.endswith('/00000000-0000-0000-0000-000000000000/wiki')


def test_integration_settings_reject_embedded_credentials_and_cleartext(monkeypatch):
    for value in ['http://example.test', 'https://user:secret@example.test']:
        monkeypatch.setenv('UNIT_TEST_ENDPOINT', value)
        with pytest.raises(ValueError, match='HTTPS'):
            config.https_setting('UNIT_TEST_ENDPOINT', 'https://example.test')
    monkeypatch.setenv('UNIT_TEST_ENDPOINT', 'https://example.test/service/')
    assert config.https_setting('UNIT_TEST_ENDPOINT', '') == 'https://example.test/service'
