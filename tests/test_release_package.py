"""Source delivery verifies inventory and member hashes before extraction."""
import hashlib
import io
import json
import tarfile

import pytest

from tools.verify_release import verify


def archive(tmp_path, payload=b'synthetic framework', *, extra=None, declared=None):
    entries=[dict(path='README.md',size=len(payload),mode='100644',sha256=hashlib.sha256(payload if declared is None else declared).hexdigest())]
    manifest=dict(schema_version=1,commit='a'*40,files=entries)
    target=tmp_path/'release.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for name,data in [('README.md',payload),('FILE_MANIFEST.json',json.dumps(manifest).encode())]+([extra] if extra else []):
            member=tarfile.TarInfo('nvis-ova-validation-hub/'+name);member.size=len(data);member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    return target,hashlib.sha256(target.read_bytes()).hexdigest()


def test_verified_source_inventory(tmp_path):
    path,digest=archive(tmp_path)
    result=verify(path,digest)
    assert result['status']=='VERIFIED' and result['files']==1 and result['commit']=='a'*40


def test_rejects_archive_tamper(tmp_path):
    path,digest=archive(tmp_path)
    path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='SHA-256'):verify(path,digest)


def test_rejects_member_tamper_even_with_matching_archive_digest(tmp_path):
    path,digest=archive(tmp_path,declared=b'different expected source')
    with pytest.raises(ValueError,match='mismatch'):verify(path,digest)


@pytest.mark.parametrize('name',['../escape.txt','unexpected.txt'])
def test_rejects_unsafe_or_unmanifested_member(tmp_path,name):
    path,digest=archive(tmp_path,extra=(name,b'not allowed'))
    with pytest.raises(ValueError):verify(path,digest)
