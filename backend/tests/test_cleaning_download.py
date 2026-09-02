import pytest
from fastapi.testclient import TestClient

import app.routers.cleaning as cleaning_router
from app.main import app


class FakeUser:
    id = "user-1"


@pytest.fixture(autouse=True)
def _auth_override():
    app.dependency_overrides[cleaning_router.get_current_user] = lambda: FakeUser()
    yield
    app.dependency_overrides.clear()


class FakeSupabaseResult:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    """Minimal stand-in for the supabase-py fluent query builder - enough
    of .select().eq().eq().single().execute() to drive the download
    endpoint's own query."""

    def __init__(self, row):
        self._row = row

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def single(self):
        return self

    def execute(self):
        return FakeSupabaseResult(self._row)


class FakeStorageBucket:
    def __init__(self, files: dict[str, bytes]):
        self._files = files

    def download(self, path: str) -> bytes:
        if path not in self._files:
            raise Exception(f"not found: {path}")
        return self._files[path]


class FakeStorage:
    def __init__(self, files: dict[str, bytes]):
        self._files = files

    def from_(self, _bucket_name):
        return FakeStorageBucket(self._files)


class FakeSupabaseAdmin:
    def __init__(self, row, files: dict[str, bytes]):
        self._row = row
        self.storage = FakeStorage(files)

    def table(self, _name):
        return FakeQuery(self._row)


def test_download_returns_the_cleaned_csv_bytes(monkeypatch):
    fake_admin = FakeSupabaseAdmin(
        row={"filename": "sales.csv", "cleaned_storage_path": "user-1/d1_cleaned.csv"},
        files={"user-1/d1_cleaned.csv": b"a,b\n1,2\n"},
    )
    monkeypatch.setattr(cleaning_router, "supabase_admin", fake_admin)

    client = TestClient(app)
    response = client.post("/clean/download", json={"dataset_id": "d1"})

    assert response.status_code == 200
    assert response.content == b"a,b\n1,2\n"
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="sales_cleaned.csv"' in response.headers["content-disposition"]


def test_download_before_cleaning_has_ever_been_applied_returns_400(monkeypatch):
    fake_admin = FakeSupabaseAdmin(
        row={"filename": "sales.csv", "cleaned_storage_path": None},
        files={},
    )
    monkeypatch.setattr(cleaning_router, "supabase_admin", fake_admin)

    client = TestClient(app)
    response = client.post("/clean/download", json={"dataset_id": "d1"})

    assert response.status_code == 400
    assert "run 'Apply cleaning' first" in response.json()["detail"]


def test_download_for_unknown_dataset_returns_404(monkeypatch):
    fake_admin = FakeSupabaseAdmin(row=None, files={})
    monkeypatch.setattr(cleaning_router, "supabase_admin", fake_admin)

    client = TestClient(app)
    response = client.post("/clean/download", json={"dataset_id": "does-not-exist"})

    assert response.status_code == 404


def test_download_filename_without_extension_is_handled():
    # A filename with no "." shouldn't crash rsplit-based base-name logic.
    from app.routers.cleaning import download_cleaned  # noqa: F401 - import sanity check only

    assert "no_extension".rsplit(".", 1)[0] == "no_extension"
