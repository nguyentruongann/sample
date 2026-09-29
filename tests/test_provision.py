"""The failure reported by Docker must fail fast or provision with admin rights."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from demand_forecasting.provision import parse_database_target, provision_external_databases


APP = "postgresql://demand_user:app-pass@host.docker.internal:5432/demand_db"
META = "postgresql+psycopg2://airflow:meta-pass@host.docker.internal:5432/airflow"
ADMIN = "postgresql://postgres:admin-pass@host.docker.internal:5432/postgres"


class OperationalError(Exception):
    pass


class FakeSQL(str):
    def format(self, *args):
        return FakeSQL(str(self).format(*args))


def fake_sql_module():
    return SimpleNamespace(
        SQL=FakeSQL,
        Identifier=lambda value: '"' + value.replace('"', '""') + '"',
        Literal=lambda value: "'" + value.replace("'", "''") + "'",
    )


def test_urls_decode_credentials_and_remove_airflow_driver_scheme():
    target = parse_database_target(
        "postgresql+psycopg2://airflow:part%40secret@host.docker.internal:5432/airflow",
        "AIRFLOW_METADATA_URL",
    )
    assert target.url.startswith("postgresql://")
    assert target.database == "airflow"
    assert target.password == "part@secret"


def test_missing_database_fails_immediately_with_actionable_message(monkeypatch):
    psycopg = ModuleType("psycopg")
    psycopg.OperationalError = OperationalError

    def connect(*_args, **_kwargs):
        raise OperationalError(
            'database "demand_db" does not exist; IPv6: Network is unreachable'
        )

    psycopg.connect = connect
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    with pytest.raises(RuntimeError, match="POSTGRES_ADMIN_URL") as error:
        provision_external_databases(APP, META)
    assert "app-pass" not in str(error.value)


def test_provision_is_idempotent_and_keeps_existing_roles(monkeypatch):
    roles = {"postgres", "demand_user"}
    databases = {"postgres"}
    created = []

    class Connection:
        def __init__(self, is_admin):
            self.is_admin = is_admin
            self.answer = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return self

        def execute(self, query, params=None):
            query = str(query)
            if query.startswith("SELECT 1 FROM pg_roles"):
                self.answer = (1,) if params[0] in roles else None
            elif query.startswith("SELECT 1 FROM pg_database"):
                self.answer = (1,) if params[0] in databases else None
            elif query.startswith("CREATE ROLE"):
                role = query.split('"')[1]
                roles.add(role)
                created.append(("role", role))
            elif query.startswith("CREATE DATABASE"):
                database = query.split('"')[1]
                databases.add(database)
                created.append(("database", database))
            else:
                raise AssertionError(query)

        def fetchone(self):
            return self.answer

    psycopg = ModuleType("psycopg")
    psycopg.OperationalError = OperationalError
    psycopg.Error = OperationalError
    psycopg.sql = fake_sql_module()
    psycopg.connect = lambda url, **_kwargs: Connection("/postgres" in url)
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)

    provision_external_databases(APP, META, ADMIN)
    provision_external_databases(APP, META, ADMIN)
    assert created == [
        ("database", "demand_db"),
        ("role", "airflow"),
        ("database", "airflow"),
    ]
    assert "demand_user" in roles  # its existing password was never changed


def test_refuses_admin_connection_to_another_server():
    with pytest.raises(ValueError, match="same PostgreSQL host/port"):
        provision_external_databases(APP, META, ADMIN.replace("host.docker.internal", "other-host"))
