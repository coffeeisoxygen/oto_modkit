"""Database connection module for OCR Voucher project using Streamlit best practices."""

from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd
import streamlit as st
from sqlalchemy import text

from mylog import LogContext, logger, timer

# Constants for error messages
ERROR_NO_CONNECTION = "No database connection available"
ERROR_DB_CONFIG_NOT_FOUND = (
    "Database configuration not found. Please check your secrets.toml file."
)
ERROR_MISSING_CONFIG = "Missing database configuration"


@logger.catch(reraise=True)
@st.cache_resource
def get_connection():
    """Get Streamlit SQL connection using modern st.connection API with best practices."""
    try:
        # Validate secrets are available
        if (
            "connections" not in st.secrets
            or "sqlserver" not in st.secrets["connections"]
        ):
            logger.error(ERROR_DB_CONFIG_NOT_FOUND)
            return None

        # Using st.connection with named connection from secrets.toml
        logger.info("Establishing database connection...")
        conn = st.connection("sqlserver", type="sql")

        # Test connection immediately after creation
        try:
            conn.query("SELECT 1", ttl=0)
        except Exception as test_error:
            logger.error(f"Database connection test failed: {test_error}")
            return None
        else:
            logger.success("Database connection established successfully")
            return conn

    except KeyError as e:
        logger.error(f"Missing configuration key: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to establish database connection: {e}")
        return None


@logger.catch(reraise=True)
@contextmanager
def get_session() -> Generator:
    """Get database session with proper cleanup using Streamlit connection."""
    conn = get_connection()
    if conn is None:
        logger.error(ERROR_NO_CONNECTION)
        yield None
        return

    # Use the session property from SQLConnection
    try:
        with conn.session as session:
            logger.debug("Database session created")
            yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        yield None


@logger.catch(reraise=True)
def test_connection() -> bool:
    """Test database connection using Streamlit connection."""
    try:
        with LogContext("DATABASE_CONNECTION_TEST", level="INFO"):
            conn = get_connection()
            if conn is None:
                return False

            # Simple test query
            conn.query("SELECT 1", ttl=0)
            logger.success("Database connection test passed")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


@timer("DATABASE_QUERY")
@logger.catch(reraise=True)
def execute_query(sql: str, params: dict | None = None, ttl: int = 600) -> pd.DataFrame:
    """Execute a read-only query and return DataFrame."""
    try:
        conn = get_connection()
        if conn is None:
            logger.error(ERROR_NO_CONNECTION)
            return pd.DataFrame()

        logger.debug(f"Executing query with params: {params}")

        if params:
            result = conn.query(sql, params=params, ttl=ttl)
        else:
            result = conn.query(sql, ttl=ttl)

    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return pd.DataFrame()
    else:
        logger.info(f"Query executed successfully, returned {len(result)} rows")
        return result


@logger.catch(reraise=True)
def execute_write(sql: str, params: dict | None = None):
    """Execute a write query (INSERT, UPDATE, DELETE) using session."""
    try:
        with (
            LogContext("DATABASE_WRITE_OPERATION", level="INFO"),
            get_session() as session,
        ):
            if session is None:
                logger.error(ERROR_NO_CONNECTION)
                return False

            logger.debug(f"Executing write query with params: {params}")

            if params:
                result = session.execute(text(sql), params)
            else:
                result = session.execute(text(sql))

            session.commit()
            logger.success(
                f"Write operation completed, affected rows: {result.rowcount}"
            )
            return True

    except Exception as e:
        logger.error(f"Write operation failed: {e}")
        return False
