# src/utils/config_manager.py
# File path: src/utils/config_manager.py

import os
from functools import lru_cache
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field, HttpUrl


class GeneralSettings(BaseModel):
    log_level: str = "INFO"
    gpu_enabled: bool = True


class ServiceSettings(BaseModel):
    port: int


class NERServiceSettings(ServiceSettings):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_cache_dir: str


class DPServiceSettings(ServiceSettings):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_cache_dir: str


class EventLLMServiceSettings(ServiceSettings):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_path: str
    model_cache_dir: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    request_timeout_seconds: int = 300
    generation_max_retries: int = Field(
        3, description="Number of retries for LLM generation if output is invalid.")
    generation_retry_delay_seconds: int = Field(
        2, description="Delay in seconds between generation retries.")



class OrchestratorServiceSettings(ServiceSettings):
    ner_service_url: HttpUrl
    dp_service_url: HttpUrl
    event_llm_service_url: HttpUrl
    batch_processing_chunk_size: int = 100
    batch_processing_job_results_ttl: int = 3600  # seconds
    request_timeout_seconds: int = 120


class CelerySettings(BaseModel):
    broker_url: str
    result_backend: str
    task_acks_late: bool = True
    worker_prefetch_multiplier: int = 1
    dask_local_cluster_n_workers: Optional[int] = Field(
        None, description="Number of Dask workers for local cluster, None to auto-detect")
    dask_local_cluster_threads_per_worker: int = 1
    dask_local_cluster_memory_limit: Optional[str] = None


class AppSettings(BaseModel):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    ner_service: NERServiceSettings
    dp_service: DPServiceSettings
    event_llm_service: EventLLMServiceSettings
    orchestrator_service: OrchestratorServiceSettings
    celery: CelerySettings


class ConfigManager:
    _instance: Optional[AppSettings] = None
    _config_path: str = "./config/settings.yaml"

    @classmethod
    @lru_cache(maxsize=1)
    def get_settings(cls) -> AppSettings:
        if cls._instance is None:
            try:
                with open(cls._config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                cls._instance = AppSettings.parse_obj(config_data)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Configuration file not found at {cls._config_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration: {e}")
            except Exception as e:
                raise ValueError(f"Error loading application settings: {e}")
        return cls._instance

    @classmethod
    def set_config_path(cls, path: str):
        cls._config_path = path
        cls.get_settings.cache_clear()
        cls._instance = None


if __name__ == "__main__":
    try:
        dummy_config_content = """
        general:
          log_level: DEBUG
          gpu_enabled: True
        ner_service:
          port: 8001
          model_name: "test/ner"
          model_cache_dir: "/tmp/hf"
        dp_service:
          port: 8002
          model_name: "test_spacy"
          model_cache_dir: "/tmp/spacy"
        event_llm_service:
          port: 8003
          model_name: "test/llm"
          model_path: "/tmp/llm/model"
          model_cache_dir: "/tmp/llm"
          generation_max_retries: 3
          generation_retry_delay_seconds: 2
        orchestrator_service:
          port: 8000
          ner_service_url: "http://localhost:8001"
          dp_service_url: "http://localhost:8002"
          event_llm_service_url: "http://localhost:8003"
        celery:
          broker_url: "redis://localhost:6379/0"
          result_backend: "redis://localhost:6379/0"
          dask_local_cluster_n_workers: 4
          dask_local_cluster_memory_limit: "1GB"
        """
        with open("./config/settings.yaml", "w") as f:
            f.write(dummy_config_content)
        settings = ConfigManager.get_settings()
        print(f"Log Level: {settings.general.log_level}")
        print(f"NER Model: {settings.ner_service.model_name}")
        print(f"Celery Broker: {settings.celery.broker_url}")
        print(f"Dask Workers: {settings.celery.dask_local_cluster_n_workers}")
        settings_again = ConfigManager.get_settings()
        print("Settings retrieved again (should be cached):",
              settings_again.general.log_level)
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        if os.path.exists("./config/settings.yaml"):
            os.remove("./config/settings.yaml")

# src/utils/config_manager.py
# File path: src/utils/config_manager.py