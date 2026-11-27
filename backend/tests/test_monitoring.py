from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.routers.monitoring as monitoring_router
from app.main import app
from app.services import monitoring


class FakeUser:
    id = "user-1"


@pytest.fixture(autouse=True)
def _auth_override():
    app.dependency_overrides[monitoring_router.get_current_user] = lambda: FakeUser()
    yield
    app.dependency_overrides.clear()


def test_preview_is_scoped_to_the_authenticated_user(monkeypatch):
    observed = {}

    def fake_read_metric(table_name, metric_column, user_id):
        observed.update(table_name=table_name, metric_column=metric_column, user_id=user_id)
        return 1250.5

    monkeypatch.setattr(monitoring_router.monitoring, "read_metric_value", fake_read_metric)
    response = TestClient(app).post(
        "/monitoring/preview",
        json={"source_table": "user_kpis", "metric_column": "monthly_revenue"},
    )

    assert response.status_code == 200
    assert response.json() == {"current_value": 1250.5}
    assert observed == {
        "table_name": "user_kpis",
        "metric_column": "monthly_revenue",
        "user_id": FakeUser.id,
    }


def test_create_monitor_saves_rule_and_evaluates_initial_value(monkeypatch):
    recorded = {}

    def fake_read_metric(table_name, metric_column, user_id):
        return 120

    def fake_create_rule(user_id, rule):
        recorded.update(rule)
        return {"id": "rule-1", **rule, "user_id": user_id, "is_triggered": False}

    def fake_evaluate_rule(rule, value):
        recorded["initial_value"] = value
        return {**rule, "last_value": value, "is_triggered": value >= rule["threshold"]}

    monkeypatch.setattr(monitoring_router.monitoring, "read_metric_value", fake_read_metric)
    monkeypatch.setattr(monitoring_router.monitoring, "create_rule", fake_create_rule)
    monkeypatch.setattr(monitoring_router.monitoring, "evaluate_rule", fake_evaluate_rule)

    response = TestClient(app).post(
        "/monitoring/rules",
        json={
            "name": "High revenue",
            "source_table": "user_kpis",
            "metric_column": "revenue",
            "comparison": "above",
            "threshold": 100,
        },
    )

    assert response.status_code == 200
    assert response.json()["rule"]["last_value"] == 120
    assert response.json()["rule"]["is_triggered"] is True
    assert recorded["initial_value"] == 120


def test_monitor_source_requires_safe_postgres_identifiers():
    with pytest.raises(ValueError, match="simple PostgreSQL"):
        monitoring.read_metric_value("user_kpis;drop_table", "revenue", FakeUser.id)


def test_monitor_source_explains_missing_table_in_api_schema(monkeypatch):
    class MissingTableError(Exception):
        code = "PGRST205"

    class MissingTableQuery:
        def select(self, *_args):
            return self

        def eq(self, *_args):
            return self

        def limit(self, *_args):
            return self

        def execute(self):
            raise MissingTableError("table is not in the schema cache")

    class MissingTableClient:
        def table(self, _table):
            return MissingTableQuery()

    monkeypatch.setattr(monitoring, "supabase_admin", MissingTableClient())
    with pytest.raises(ValueError, match="API schema cache"):
        monitoring.read_metric_value("monitoring_demo_kpi", "monthly_revenue", FakeUser.id)


def test_email_alert_requires_smtp_configuration(monkeypatch):
    for field in ("smtp_host", "smtp_username", "smtp_password", "smtp_from_email"):
        monkeypatch.setattr(monitoring_router.settings, field, "")

    response = TestClient(app).post(
        "/monitoring/rules",
        json={
            "name": "High revenue",
            "source_table": "user_kpis",
            "metric_column": "revenue",
            "comparison": "above",
            "threshold": 100,
            "email_enabled": True,
        },
    )

    assert response.status_code == 503
    assert "Email alerts are not configured" in response.json()["detail"]


def test_email_alert_uses_authenticated_account_address_and_starttls(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent.update(host=host, port=port, timeout=timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def ehlo(self):
            sent["ehlo"] = sent.get("ehlo", 0) + 1

        def starttls(self, context):
            sent["tls_context"] = context

        def login(self, username, password):
            sent["login"] = (username, password)

        def send_message(self, message):
            sent["message"] = message

    monkeypatch.setattr(
        monitoring,
        "supabase_admin",
        SimpleNamespace(
            auth=SimpleNamespace(
                admin=SimpleNamespace(
                    get_user_by_id=lambda user_id: SimpleNamespace(
                        user=SimpleNamespace(email="user@example.com")
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(monitoring.settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(monitoring.settings, "smtp_port", 587)
    monkeypatch.setattr(monitoring.settings, "smtp_username", "smtp-user")
    monkeypatch.setattr(monitoring.settings, "smtp_password", "smtp-password")
    monkeypatch.setattr(monitoring.settings, "smtp_from_email", "alerts@example.com")
    monkeypatch.setattr(monitoring.smtplib, "SMTP", FakeSMTP)

    monitoring._send_alert_email(FakeUser.id, "High revenue", "Threshold crossed.")

    assert sent["host"] == "smtp.example.com"
    assert sent["port"] == 587
    assert sent["timeout"] == 10
    assert sent["ehlo"] == 2
    assert sent["message"]["To"] == "user@example.com"
    assert sent["message"]["Subject"] == "Nexus AI alert: High revenue"
    assert sent["login"] == ("smtp-user", "smtp-password")


def test_monitoring_fires_once_until_the_metric_returns_to_normal(monkeypatch):
    class FakeSupabase:
        def __init__(self):
            self.inserted_alerts = []
            self.rule_updates = []

        def table(self, name):
            return FakeQuery(self, name)

    class FakeQuery:
        def __init__(self, client, table):
            self.client = client
            self.table_name = table
            self.payload = None

        def insert(self, payload):
            self.payload = payload
            return self

        def update(self, payload):
            self.payload = payload
            return self

        def eq(self, *_args):
            return self

        def execute(self):
            if self.table_name == monitoring.ALERTS_TABLE:
                self.client.inserted_alerts.append(self.payload)
            else:
                self.client.rule_updates.append(self.payload)
            return SimpleNamespace(data=[self.payload])

    client = FakeSupabase()
    monkeypatch.setattr(monitoring, "supabase_admin", client)
    rule = {
        "id": "rule-1",
        "user_id": FakeUser.id,
        "name": "High revenue",
        "source_table": "user_kpis",
        "metric_column": "revenue",
        "comparison": "above",
        "threshold": 100,
        "is_triggered": False,
    }

    triggered = monitoring.evaluate_rule(rule, 120)
    monitoring.evaluate_rule(triggered, 130)
    recovered = monitoring.evaluate_rule(triggered, 90)
    monitoring.evaluate_rule(recovered, 110)

    assert len(client.inserted_alerts) == 2
    assert [update["is_triggered"] for update in client.rule_updates] == [True, True, False, True]


def test_email_opt_in_sends_on_trigger_and_records_delivery(monkeypatch):
    class FakeSupabase:
        def __init__(self):
            self.alert_rows = []
            self.alert_updates = []
            self.auth = SimpleNamespace(
                admin=SimpleNamespace(
                    get_user_by_id=lambda user_id: SimpleNamespace(
                        user=SimpleNamespace(email="user@example.com")
                    )
                )
            )

        def table(self, name):
            return FakeQuery(self, name)

    class FakeQuery:
        def __init__(self, client, table_name):
            self.client = client
            self.table_name = table_name
            self.operation = None
            self.payload = None

        def insert(self, payload):
            self.operation = "insert"
            self.payload = payload
            return self

        def update(self, payload):
            self.operation = "update"
            self.payload = payload
            return self

        def eq(self, *_args):
            return self

        def execute(self):
            if self.table_name == monitoring.ALERTS_TABLE and self.operation == "insert":
                row = {"id": "alert-1", **self.payload}
                self.client.alert_rows.append(row)
                return SimpleNamespace(data=[row])
            if self.table_name == monitoring.ALERTS_TABLE:
                self.client.alert_updates.append(self.payload)
            return SimpleNamespace(data=[self.payload])

    client = FakeSupabase()
    email_calls = []
    monkeypatch.setattr(monitoring, "supabase_admin", client)
    monkeypatch.setattr(
        monitoring,
        "_send_alert_email",
        lambda user_id, rule_name, text: email_calls.append((user_id, rule_name, text)),
    )
    rule = {
        "id": "rule-1",
        "user_id": FakeUser.id,
        "name": "High revenue",
        "source_table": "user_kpis",
        "metric_column": "revenue",
        "comparison": "above",
        "threshold": 100,
        "is_triggered": False,
        "email_enabled": True,
    }

    monitoring.evaluate_rule(rule, 120)

    assert email_calls and email_calls[0][:2] == (FakeUser.id, "High revenue")
    assert client.alert_rows[0]["email_status"] == "pending"
    assert client.alert_updates == [{"email_status": "sent"}]
