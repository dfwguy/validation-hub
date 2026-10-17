"""Portable runtime paths and integration origins; never stores credentials."""
import os
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_root():
    return Path(os.environ.get('VALIDATION_DATA_ROOT') or PROJECT_ROOT / 'var' / 'validation-hub').expanduser().resolve()


def data_path(setting, relative):
    value = os.environ.get(setting)
    return Path(value).expanduser() if value else data_root() / relative


def uploads_root():
    return data_path('VALIDATION_UPLOADS', 'uploads')


def evidence_root():
    return data_path('CONFLUENCE_EVIDENCE_DIR', 'atlassian-evidence')


def https_setting(name, default):
    value = (os.environ.get(name) or default).rstrip('/')
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(name + ' must be an HTTPS origin or URL without credentials, query or fragment')
    return value


BEACON_BASE_URL = https_setting('BEACON_BASE_URL', 'https://llmb.nvidia.com/beacon-v2/runs')
BEACON_API_URL = https_setting('BEACON_API_URL', 'https://api.llmb.nvidia.com/v2/training/runs')
GPUCOMMS_URL = https_setting('GPUCOMMS_URL', 'https://gpucomms.nvidia.com') + '/'
NCCL_DASHBOARD_URL = https_setting('NCCL_DASHBOARD_URL', 'https://nccl-perf-dashboard-354b5e.gitlab-master-pages.nvidia.com') + '/'
ATLASSIAN_SITE = https_setting('ATLASSIAN_SITE', 'https://nvidia.atlassian.net/wiki')
BEACON_HOSTS = frozenset(urlsplit(url).hostname for url in (BEACON_BASE_URL, BEACON_API_URL))
REFERENCE_HOSTS = BEACON_HOSTS | frozenset(urlsplit(url).hostname for url in
    (GPUCOMMS_URL, NCCL_DASHBOARD_URL, ATLASSIAN_SITE, 'https://exemplar.nvidia.com'))
