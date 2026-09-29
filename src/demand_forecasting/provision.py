"""Idempotent provisioning for PostgreSQL installed outside Docker.

Only a dedicated bootstrap process receives POSTGRES_ADMIN_URL. Application and
Airflow containers continue using their separate, less privileged accounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlsplit, urlunsplit


@dataclass(frozen=True)
class DatabaseTarget:
    url: str
    host: str
    port: int
    database: str
    user: str
    password: str | None


def parse_database_target(value: str, label: str) -> DatabaseTarget:
    """Convert SQLAlchemy's PostgreSQL driver scheme to a psycopg URL."""

    parsed = urlsplit(value.strip())
    if parsed.scheme not in ("postgresql", "postgres", "postgresql+psycopg2", "postgresql+psycopg"):
        raise ValueError(f"{label} must be a PostgreSQL connection URL")
    if not parsed.hostname or not parsed.username or not parsed.path.lstrip("/"):
        raise ValueError(f"{label} must include host, username and database")
    try:
        port = parsed.port or 5432
    except ValueError as error:
        raise ValueError(f"{label} has an invalid port") from error
    return DatabaseTarget(
        url=urlunsplit(("postgresql", parsed.netloc, parsed.path, parsed.query, parsed.fragment)),
        host=parsed.hostname.lower(),
        port=port,
        database=unquote(parsed.path.lstrip("/")),
        user=unquote(parsed.username),
        password=unquote(parsed.password) if parsed.password is not None else None,
    )


def _connect_target(target: DatabaseTarget, label: str) -> None:
    import psycopg

    try:
        with psycopg.connect(target.url, connect_timeout=5):
            return
    except psycopg.OperationalError as error:
        message = str(error).lower()
        if "does not exist" in message:
            raise RuntimeError(
                f"{label}: database or role is missing at {target.host}:{target.port}. "
                "Set POSTGRES_ADMIN_URL to an existing PostgreSQL admin account, "
                "then rerun init-runtime."
            ) from None
        if "password authentication failed" in message or "no password supplied" in message:
            raise RuntimeError(
                f"{label}: authentication failed at {target.host}:{target.port}; "
                "check the account/password in .env. Existing passwords are not changed automatically."
            ) from None
        raise RuntimeError(
            f"{label}: cannot connect to {target.host}:{target.port}/{target.database}. "
            "Check PostgreSQL is running, its TCP settings and the network/firewall."
        ) from None


def provision_external_databases(
    database_url: str,
    airflow_metadata_url: str | None,
    admin_url: str | None = None,
) -> None:
    """Create missing roles/databases if admin credentials are explicitly given.

    Never drop data, change an existing role's password, or grant admin rights
    to the application roles. Running the operation again is harmless.
    """

    targets = [parse_database_target(database_url, "DATABASE_URL")]
    if airflow_metadata_url:
        targets.append(parse_database_target(airflow_metadata_url, "AIRFLOW_METADATA_URL"))

    if not admin_url:
        for index, target in enumerate(targets):
            _connect_target(target, "DATABASE_URL" if index == 0 else "AIRFLOW_METADATA_URL")
        return

    admin = parse_database_target(admin_url, "POSTGRES_ADMIN_URL")
    if any((t.host, t.port) != (admin.host, admin.port) for t in targets):
        raise ValueError("POSTGRES_ADMIN_URL must point to the same PostgreSQL host/port as both database URLs")

    import psycopg
    from psycopg import sql

    try:
        with psycopg.connect(admin.url, connect_timeout=5, autocommit=True) as connection:
            with connection.cursor() as cursor:
                for target in targets:
                    cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (target.user,))
                    if cursor.fetchone() is None:
                        if not target.password:
                            raise ValueError(f"Cannot create role for {target.database} without a URL password")
                        cursor.execute(
                            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                                sql.Identifier(target.user), sql.Literal(target.password)
                            )
                        )
                        print(f"Created PostgreSQL role {target.user}")

                    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target.database,))
                    if cursor.fetchone() is None:
                        cursor.execute(
                            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                                sql.Identifier(target.database), sql.Identifier(target.user)
                            )
                        )
                        print(f"Created PostgreSQL database {target.database}")
    except psycopg.Error as error:
        raise RuntimeError(
            "PostgreSQL bootstrap failed: POSTGRES_ADMIN_URL must connect to an "
            "existing database using a role allowed to create roles and databases "
            f"(SQLSTATE {error.sqlstate or 'unknown'})."
        ) from None

    # This also detects a stale password on a role that already existed.
    for index, target in enumerate(targets):
        _connect_target(target, "DATABASE_URL" if index == 0 else "AIRFLOW_METADATA_URL")
