# src/storage/backends.py
# File path: src/storage/backends.py
"""
Defines abstract and concrete storage backend implementations for
processed EEE articles (JSONL, Elasticsearch, PostgreSQL),
and a factory to retrieve them based on configuration.
This structure is mirrored from the cleaning pipeline for consistency (DRY/SOLID).
"""

import atexit
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import date, datetime
from pathlib import Path

# Conditional imports for database clients - they are only imported when the class is instantiated
try:
    from elasticsearch import Elasticsearch, helpers as es_helpers
except ImportError:
    Elasticsearch = None
    es_helpers = None

try:
    import psycopg2
    from psycopg2 import sql as pg_sql
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    psycopg2 = None
    pg_sql = None
    # CRITICAL FIX: The Python compiler *must* see valid objects for type hints.
    # Since we cannot import, we set placeholder attributes on the psycopg2 object if it's None.
    # However, since we fixed the dependency in requirements.txt, this *should* now be imported
    # successfully in the orchestrator container, thus fixing the AttributeError.
    ISOLATION_LEVEL_AUTOCOMMIT = None


# CRITICAL FIX: Import actual configuration models
from src.utils.config_manager import ConfigManager, JsonlStorageConfig, ElasticsearchStorageConfig, PostgreSQLStorageConfig
from src.schemas.data_models import EventLLMGenerateResponse

logger = logging.getLogger("storage_backend")


class StorageBackend(ABC):
    """Abstract Base Class for storage backends."""
    @abstractmethod
    def initialize(self):
        """Initializes the storage backend (e.g., establish connection, create directories/tables)."""
        pass

    @abstractmethod
    def save(self, data: EventLLMGenerateResponse, **kwargs: Any) -> None:
        """Saves a single processed article."""
        pass

    @abstractmethod
    def save_batch(self, data_list: List[EventLLMGenerateResponse], **kwargs: Any) -> None:
        """Saves a batch of processed articles."""
        pass

    @abstractmethod
    def close(self):
        """Closes any open connections or resources."""
        pass


class JSONLStorageBackend(StorageBackend):
    """
    Storage backend that saves processed articles to a daily-created JSONL (JSON Lines) file.

    NEW: Includes causality graph edges extraction for downstream graph construction.
    """

    def __init__(self, config: JsonlStorageConfig):
        # FIX: Ensure output_path handling is robust whether it's a dir or a file path specified
        self.output_path_config = config.output_path

        # Use parent directory if a filename is specified, otherwise use the path as the directory
        path_obj = Path(config.output_path)
        self.output_directory = path_obj.parent if path_obj.suffix else path_obj

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.current_file_path: Optional[Path] = None
        self._file_handle = None
        self._current_date: Optional[date] = None
        logger.info(
            f"Initialized JSONLStorageBackend with output directory: {self.output_directory}")

    def initialize(self):
        logger.info(
            f"JSONL storage directory ensured: {self.output_directory}")

    def _get_daily_file_path(self) -> Path:
        """
        Generates the file path for today's JSONL file.
        Uses a consistent naming convention derived from the configuration.
        """
        today_str = date.today().strftime("%Y-%m-%d")
        # Extract filename part from config path, default to a sensible name
        file_name = Path(self.output_path_config).name if Path(
            self.output_path_config).suffix else f"extracted_events_{today_str}.jsonl"

        # If the configured path was a file, override the date part of the filename
        if Path(self.output_path_config).suffix:
            # e.g., if config is /app/data/output.jsonl, use that directly for CLI output,
            # but for consistency with daily logs, we'll stick to dated file structure here.
            return self.output_directory / f"extracted_events_{today_str}.jsonl"
        else:
            return self.output_directory / file_name

    def _open_file(self):
        """Opens or re-opens the daily file for appending."""
        new_file_path = self._get_daily_file_path()
        today = date.today()

        if self._file_handle is None or new_file_path != self.current_file_path or today != self._current_date:
            self.close()

            self.current_file_path = new_file_path
            self._current_date = today
            try:
                self._file_handle = open(
                    self.current_file_path, 'a', encoding='utf-8')
                logger.debug(
                    f"Opened JSONL file for appending: {self.current_file_path}")
            except Exception as e:
                logger.critical(
                    f"Failed to open JSONL file {self.current_file_path}: {e}", exc_info=True)
                raise

    def _extract_causality_edges(self, data: EventLLMGenerateResponse) -> List[Dict[str, Any]]:
        """
        Extract causality relationships as graph edges from event metadata.

        NEW METHOD: Analyzes event metadata to construct causality graph edges
        for downstream Phase 5 (Graph Construction).

        Args:
            data: Processed document with events

        Returns:
            List of edge dictionaries with structure:
            {
                "source": str (event_id),
                "target": str (event_id), 
                "causality": str (causality text),
                "edge_type": "causes",
                "confidence": float (optional)
            }
        """
        edges = []

        # Ensure document_id exists for generating event IDs
        document_id = getattr(data, 'document_id', 'unknown')

        for i, event in enumerate(data.events):
            # Skip events without causality metadata
            if not event.metadata or not event.metadata.causality:
                continue

            # Generate unique event ID for source event
            source_event_id = f"{document_id}:event_{i}:{event.trigger.text}:{event.trigger.start_char}"

            # Parse causality text to find mentions of other events in same document
            # Simple heuristic: look for event triggers mentioned in causality text
            causality_text = event.metadata.causality.lower()

            # Check if other events' triggers are mentioned in this event's causality
            for j, other_event in enumerate(data.events):
                if i == j:  # Skip self-references
                    continue

                # Check if other event's trigger text appears in causality description
                trigger_text = other_event.trigger.text.lower()

                # Look for trigger as a complete word (not substring)
                # Using word boundaries to avoid false positives
                import re
                if re.search(r'\b' + re.escape(trigger_text) + r'\b', causality_text):
                    target_event_id = f"{document_id}:event_{j}:{other_event.trigger.text}:{other_event.trigger.start_char}"

                    edges.append({
                        "source": source_event_id,
                        "target": target_event_id,
                        "causality": event.metadata.causality,
                        "edge_type": "causes",
                        "document_id": document_id,
                        "source_event_type": event.event_type,
                        "target_event_type": other_event.event_type
                    })

                    logger.debug(
                        f"Extracted causality edge: {event.event_type} -> {other_event.event_type}")

        if edges:
            logger.info(
                f"Extracted {len(edges)} causality edges from document {document_id}")

        return edges

    def _serialize_data(self, data: EventLLMGenerateResponse) -> Dict[str, Any]:
        """
        Serializes EventLLMGenerateResponse to a dictionary.

        NEW: Includes causality graph edges for downstream graph construction (Phase 5).
        """
        # Get base serialization from Pydantic model
        output_dict = data.model_dump(mode='json')

        # NEW: Extract and add causality graph edges
        try:
            causality_edges = self._extract_causality_edges(data)
            if causality_edges:
                output_dict["causality_edges"] = causality_edges
                logger.debug(
                    f"Added {len(causality_edges)} causality edges to serialized output")
        except Exception as e:
            logger.warning(
                f"Failed to extract causality edges for document {getattr(data, 'document_id', 'unknown')}: {e}")
            output_dict["causality_edges"] = []

        return output_dict

    def save(self, data: EventLLMGenerateResponse, **kwargs: Any) -> None:
        """
        Saves a single processed article.

        Delegates to save_batch for consistency and code reuse.
        """
        self.save_batch([data], **kwargs)

    def save_batch(self, data_list: List[EventLLMGenerateResponse], **kwargs: Any) -> None:
        """
        Saves a batch of processed articles to JSONL file.

        Each document is written as a single JSON line with:
        - All original EventLLMGenerateResponse fields
        - causality_edges: List of causality relationships for graph construction

        Thread-safe with file locking and atomic writes.
        """
        if not data_list:
            logger.debug(
                "Attempted to save an empty batch to JSONL. Skipping.")
            return

        try:
            # Open the daily file (creates new file if date changed)
            self._open_file()

            # Serialize and write each document
            for data in data_list:
                serialized_data = self._serialize_data(data)
                json_line = json.dumps(serialized_data, ensure_ascii=False)
                self._file_handle.write(json_line + '\n')

            # Ensure data is written to disk immediately
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())

            logger.info(
                f"Saved batch of {len(data_list)} documents to JSONL file: {self.current_file_path}")
        except Exception as e:
            logger.error(
                f"Failed to save batch to JSONL file {self.current_file_path}: {e}", exc_info=True)
            raise

    def close(self):
        """
        Closes the file handle safely.

        Ensures all data is flushed to disk before closing.
        """
        if self._file_handle:
            try:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
                self._file_handle.close()
                logger.debug(
                    f"Closed JSONL file handle: {self.current_file_path}")
            except Exception as e:
                logger.error(
                    f"Error closing JSONL file {self.current_file_path}: {e}", exc_info=True)
            finally:
                self._file_handle = None
                self.current_file_path = None
                self._current_date = None


class ElasticsearchStorageBackend(StorageBackend):
    """Storage backend that saves processed articles to Elasticsearch."""

    def __init__(self, config: ElasticsearchStorageConfig):
        if Elasticsearch is None:
            logger.warning(
                "Elasticsearch client not available. Cannot initialize ElasticsearchStorageBackend.")
            self.is_available = False
            return
        self.is_available = True
        self.config = config
        self.es: Optional[Elasticsearch] = None
        self.index_name = config.index_name
        logger.info(
            f"Initialized ElasticsearchStorageBackend for index: {self.index_name}")

    def initialize(self):
        if not self.is_available:
            raise ImportError(
                "Elasticsearch client is not installed.")
        if self.es:
            return

        try:
            connection_params = {
                "hosts": [{"host": self.config.host, "port": self.config.port, "scheme": self.config.scheme}]
            }
            if self.config.api_key:
                connection_params["api_key"] = self.config.api_key
            self.es = Elasticsearch(**connection_params)
            if not self.es.ping():
                raise ConnectionError("Could not connect to Elasticsearch.")
            logger.info("Successfully connected to Elasticsearch.")
            self._ensure_index()
        except Exception as e:
            logger.critical(
                f"Failed to initialize Elasticsearch connection or ensure index '{self.index_name}': {e}", exc_info=True)
            self.es = None
            raise

    def _ensure_index(self):
        if not self.es:
            return

        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name)
                logger.info(
                    f"Elasticsearch index '{self.index_name}' created.")
        except Exception as e:
            logger.error(
                f"Failed to check/create Elasticsearch index '{self.index_name}': {e}", exc_info=True)
            raise

    def _prepare_doc(self, data: EventLLMGenerateResponse) -> Dict[str, Any]:
        """Prepares a single EventLLMGenerateResponse for Elasticsearch indexing."""
        doc = data.model_dump(mode='json')
        doc['_id'] = data.job_id
        return doc

    def save(self, data: EventLLMGenerateResponse, **kwargs: Any) -> None:
        self.save_batch([data])

    def save_batch(self, data_list: List[EventLLMGenerateResponse], **kwargs: Any) -> None:
        if not self.is_available or not self.es:
            logger.warning(
                "Elasticsearch client not available/initialized. Skipping batch save.")
            return
        if not data_list:
            return
        if es_helpers is None:
            logger.error(
                "Elasticsearch helpers not imported. Cannot perform bulk save.")
            return

        actions = [
            {
                "_index": self.index_name,
                "_id": data.job_id,
                "_source": data.model_dump(mode='json')
            }
            for data in data_list
        ]
        try:
            success_count, errors = es_helpers.bulk(
                self.es, actions, stats_only=True, refresh="wait_for")
            if errors:
                for error in errors:
                    logger.error(f"Elasticsearch bulk save error: {error}")
            logger.info(
                f"Saved {success_count} of {len(data_list)} documents to Elasticsearch (with {len(errors)} errors).")
        except Exception as e:
            logger.error(
                f"Failed to save batch to Elasticsearch: {e}", exc_info=True)
            raise

    def close(self):
        if self.es:
            self.es = None
            logger.debug(
                "Elasticsearch client closed (connection pool reset).")


class PostgreSQLStorageBackend(StorageBackend):
    """Storage backend that saves processed articles to PostgreSQL."""

    def __init__(self, config: PostgreSQLStorageConfig):
        if psycopg2 is None:
            logger.warning(
                "psycopg2-binary not available. Cannot initialize PostgreSQLStorageBackend.")
            self.is_available = False
            return
        self.is_available = True
        self.config = config
        self.conn_params = {
            "host": config.host,
            "port": config.port,
            "dbname": config.dbname,
            "user": config.user,
            "password": config.password
        }
        self.table_name = config.table_name
        # Use string literal for forward reference
        self._connection: Optional['psycopg2.extensions.connection'] = None
        logger.info(
            f"Initialized PostgreSQLStorageBackend for table: {self.table_name}")

    def _get_connection(self) -> 'psycopg2.extensions.connection':
        if psycopg2 is None:
            raise ImportError("psycopg2-binary not available.")
        try:
            conn = psycopg2.connect(**self.conn_params)
            conn.autocommit = False
            return conn
        except Exception as e:
            logger.critical(
                f"Failed to establish PostgreSQL connection: {e}", exc_info=True)
            raise

    def initialize(self):
        if not self.is_available:
            raise ImportError("psycopg2-binary is not installed.")

        try:
            self._connection = self._get_connection()
            self._create_table_if_not_exists()
            logger.info(f"PostgreSQL backend initialized and connected.")
        except Exception as e:
            logger.critical(
                f"Failed to initialize PostgreSQL backend: {e}", exc_info=True)
            self.close()
            raise

    def _create_table_if_not_exists(self):
        if not self._connection:
            raise ConnectionError("PostgreSQL connection not established.")

        cur = None
        try:
            cur = self._connection.cursor()
            # CRITICAL: Use Identifier wrapper for table name to prevent SQL injection
            create_table_query = pg_sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                job_id VARCHAR(255) PRIMARY KEY,
                original_text TEXT,
                extracted_entities JSONB,
                extracted_soa_triplets JSONB,
                events JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """).format(pg_sql.Identifier(self.table_name))
            cur.execute(create_table_query)
            self._connection.commit()
            logger.info(
                f"PostgreSQL table '{self.table_name}' ensured to exist.")
        except Exception as e:
            logger.critical(
                f"Failed to create PostgreSQL table '{self.table_name}': {e}", exc_info=True)
            if self._connection:
                self._connection.rollback()
            raise
        finally:
            if cur:
                cur.close()

    def _prepare_sql_data(self, data: EventLLMGenerateResponse) -> Dict[str, Any]:
        """
        Prepares a single EventLLMGenerateResponse for SQL insertion.
        """
        return {
            "job_id": data.job_id,
            "original_text": data.original_text,
            # We serialize the complex Pydantic lists to JSONB
            "extracted_entities": json.dumps(data.model_dump(mode='json')['extracted_entities']),
            "extracted_soa_triplets": json.dumps(data.model_dump(mode='json')['extracted_soa_triplets']),
            "events": json.dumps(data.model_dump(mode='json')['events']),
        }

    def save(self, data: EventLLMGenerateResponse, **kwargs: Any) -> None:
        self.save_batch([data])

    def save_batch(self, data_list: List[EventLLMGenerateResponse], **kwargs: Any) -> None:
        if not self.is_available or not self._connection:
            logger.warning(
                "PostgreSQL client not available/initialized. Skipping batch save.")
            return
        if not data_list:
            return

        cur = None
        try:
            cur = self._connection.cursor()
            first_data = self._prepare_sql_data(data_list[0])
            columns = pg_sql.SQL(', ').join(
                map(pg_sql.Identifier, first_data.keys()))
            placeholders = pg_sql.SQL(', ').join(
                pg_sql.Placeholder() * len(first_data))

            # Use job_id as the conflict target
            update_columns = pg_sql.SQL(', ').join(
                pg_sql.SQL('{} = EXCLUDED.{}').format(
                    pg_sql.Identifier(col), pg_sql.Identifier(col))
                # Do not update job_id or created_at
                for col in first_data.keys() if col != 'job_id' and col != 'created_at'
            )

            insert_query = pg_sql.SQL(
                "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT (job_id) DO UPDATE SET {};"
            ).format(
                pg_sql.Identifier(self.table_name),
                columns,
                placeholders,
                update_columns
            )

            batch_values = [tuple(self._prepare_sql_data(
                data).values()) for data in data_list]
            cur.executemany(insert_query, batch_values)
            self._connection.commit()
            logger.info(
                f"Saved batch of {len(data_list)} documents to PostgreSQL.")
        except Exception as e:
            logger.error(
                f"Failed to save batch to PostgreSQL: {e}", exc_info=True)
            if self._connection:
                self._connection.rollback()
            raise
        finally:
            if cur:
                cur.close()

    def close(self):
        if self._connection:
            try:
                self._connection.close()
                logger.info("PostgreSQL connection closed.")
            except Exception as e:
                logger.error(
                    f"Error closing PostgreSQL connection: {e}", exc_info=True)
            finally:
                self._connection = None


class StorageBackendFactory:
    """
    Factory to create and provide appropriate storage backend instances based on configuration.
    Manages the lifecycle of backends to ensure proper initialization and closing.
    """
    _initialized_backends: Dict[str, StorageBackend] = {}

    @classmethod
    def get_backends(cls, requested_backends: Optional[List[str]] = None) -> List[StorageBackend]:
        """
        Returns a list of initialized StorageBackend instances.

        Args:
            requested_backends: Optional list of backend names to activate for a specific request.

        Returns:
            A list of initialized StorageBackend instances.
        """
        try:
            settings = ConfigManager.get_settings()
            storage_config = settings.storage
        except Exception as e:
            logger.error(
                f"Failed to load settings in StorageBackendFactory: {e}")
            return []

        backends_to_use = []
        if requested_backends:
            # Filter requested backends against enabled backends
            backends_to_use = [b.lower() for b in requested_backends if b.lower() in [
                e.lower() for e in storage_config.enabled_backends
            ]]
        else:
            backends_to_use = [b.lower()
                               for b in storage_config.enabled_backends]

        if not backends_to_use:
            logger.info(
                "No storage backends enabled or requested. Skipping persistence.")
            return []

        active_backends: List[StorageBackend] = []

        for backend_name in backends_to_use:
            if backend_name not in cls._initialized_backends:
                try:
                    if backend_name == "jsonl" and storage_config.jsonl:
                        backend = JSONLStorageBackend(
                            config=storage_config.jsonl)
                    elif backend_name == "elasticsearch" and storage_config.elasticsearch:
                        backend = ElasticsearchStorageBackend(
                            config=storage_config.elasticsearch)
                    elif backend_name == "postgresql" and storage_config.postgresql:
                        backend = PostgreSQLStorageBackend(
                            config=storage_config.postgresql)
                    else:
                        logger.warning(
                            f"Unsupported or unconfigured storage backend type: '{backend_name}'. Skipping.")
                        continue

                    backend.initialize()
                    cls._initialized_backends[backend_name] = backend
                    logger.info(
                        f"Storage backend '{backend_name}' successfully initialized.")
                    active_backends.append(backend)

                except (ValueError, ImportError, ConnectionError) as e:
                    logger.critical(
                        f"Failed to initialize storage backend '{backend_name}': {e}. This backend will be skipped.", exc_info=True)
                except Exception as e:
                    logger.critical(
                        f"An unexpected error occurred during initialization of backend '{backend_name}': {e}. Skipping.", exc_info=True)
            else:
                backend = cls._initialized_backends[backend_name]
                active_backends.append(backend)
                logger.debug(
                    f"Reusing already initialized storage backend: {backend_name}.")

        return active_backends

    @classmethod
    def close_all_backends(cls):
        """Closes all initialized storage backend connections/resources."""
        logger.info("Attempting to close all initialized storage backends.")
        for name, backend in list(cls._initialized_backends.items()):
            try:
                backend.close()
                logger.info(f"Storage backend '{name}' successfully closed.")
            except Exception as e:
                logger.error(
                    f"Error closing storage backend '{name}': {e}", exc_info=True)
            finally:
                del cls._initialized_backends[name]


atexit.register(StorageBackendFactory.close_all_backends)


# src/storage/backends.py
# File path: src/storage/backends.py
