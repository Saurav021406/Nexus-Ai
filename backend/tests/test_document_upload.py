import io
import types

import pytest
from docx import Document
from fastapi import HTTPException
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import app.routers.upload as upload_router
import app.services.datasets as datasets_module


def _make_test_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 750, text)
    c.save()
    return buf.getvalue()


def _make_test_docx(text: str) -> bytes:
    doc = Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class _FakeStorage:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    def from_(self, bucket):
        return self

    def upload(self, path, data, opts):
        self.files[path] = data
        return {}

    def download(self, path):
        return self.files[path]


class _FakeTable:
    rows: dict[str, dict] = {}

    def __init__(self):
        self._filter = None

    def insert(self, row):
        self._insert_row = row
        return self

    def select(self, cols):
        return self

    def eq(self, col, val):
        self._filter = (col, val)
        return self

    def single(self):
        return self

    def execute(self):
        if hasattr(self, "_insert_row"):
            _FakeTable.rows[self._insert_row["id"]] = self._insert_row
            return types.SimpleNamespace(data=[self._insert_row])
        for row in _FakeTable.rows.values():
            if row.get(self._filter[0]) == self._filter[1] or row.get("user_id") == self._filter[1]:
                return types.SimpleNamespace(data=row)
        return types.SimpleNamespace(data=None)


class _FakeSupabase:
    def __init__(self):
        self.storage = _FakeStorage()

    def table(self, name):
        return _FakeTable()


class _FakeUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


class _FakeUser:
    id = "u1"


@pytest.fixture
def fake_supabase(monkeypatch):
    fake = _FakeSupabase()
    _FakeTable.rows = {}
    monkeypatch.setattr(upload_router, "supabase_admin", fake)
    return fake


async def test_pdf_upload_extracts_text_and_marks_kind_document(fake_supabase):
    pdf_bytes = _make_test_pdf("REFUND POLICY: 30 day window.")
    result = await upload_router.upload_dataset(_FakeUploadFile("policy.pdf", pdf_bytes), _FakeUser())

    assert result["kind"] == "document"
    assert result["document_type"] == "pdf"
    assert "REFUND POLICY" in result["extracted_text"]
    assert result["page_count"] == 1


async def test_docx_upload_extracts_text_and_marks_kind_document(fake_supabase):
    docx_bytes = _make_test_docx("Employee Handbook: vacation is 20 days.")
    result = await upload_router.upload_dataset(_FakeUploadFile("handbook.docx", docx_bytes), _FakeUser())

    assert result["kind"] == "document"
    assert result["document_type"] == "docx"
    assert "Employee Handbook" in result["extracted_text"]


async def test_csv_upload_still_works_unchanged(fake_supabase):
    csv_bytes = b"region,revenue\nNorth,1000\nSouth,1500\n"
    result = await upload_router.upload_dataset(_FakeUploadFile("sales.csv", csv_bytes), _FakeUser())

    assert result["kind"] == "tabular"
    assert result["row_count"] == 2
    assert result["column_count"] == 2


async def test_corrupt_pdf_upload_returns_clean_400_not_a_crash(fake_supabase):
    with pytest.raises(HTTPException) as exc_info:
        await upload_router.upload_dataset(_FakeUploadFile("fake.pdf", b"not a real pdf"), _FakeUser())
    assert exc_info.value.status_code == 400


async def test_unsupported_file_extension_is_rejected(fake_supabase):
    with pytest.raises(HTTPException) as exc_info:
        await upload_router.upload_dataset(_FakeUploadFile("notes.txt", b"hello"), _FakeUser())
    assert exc_info.value.status_code == 400


def test_document_dataset_is_rejected_by_the_tabular_pipeline(monkeypatch):
    """The guard in services/datasets.py must stop a document dataset from
    ever reaching pandas - EDA/Forecast/Multi-Agent all go through
    get_dataset_dataframe(), so this one guard protects all of them."""
    monkeypatch.setattr(
        datasets_module, "get_dataset_record",
        lambda dataset_id, user_id: {
            "id": dataset_id, "filename": "contract.pdf",
            "storage_path": "u1/abc_contract.pdf", "user_id": user_id,
        },
    )
    with pytest.raises(HTTPException) as exc_info:
        datasets_module.get_dataset_dataframe("d1", "u1")
    assert exc_info.value.status_code == 400
    assert "document dataset" in exc_info.value.detail.lower()
