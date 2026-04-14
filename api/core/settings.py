from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="ecg-classifier-api", validation_alias="APP_NAME")
    api_log_level: str = Field(default="INFO", validation_alias="API_LOG_LEVEL")

    jwt_secret_key: str = Field(validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(validation_alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=60, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    admin_username: str = Field(validation_alias="ADMIN_USERNAME")
    admin_password: str = Field(validation_alias="ADMIN_PASSWORD")

    celery_broker_url: str = Field(validation_alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(validation_alias="CELERY_RESULT_BACKEND")

    database_url: str = Field(validation_alias="DATABASE_URL")

    flower_basic_auth: str = Field(validation_alias="FLOWER_BASIC_AUTH")
    flower_port: int = Field(default=5555, validation_alias="FLOWER_PORT")

    mlflow_tracking_uri: str | None = Field(
        default=None, validation_alias="MLFLOW_TRACKING_URI"
    )
    mlflow_s3_endpoint_url: str | None = Field(
        default=None, validation_alias="MLFLOW_S3_ENDPOINT_URL"
    )
    aws_access_key_id: str | None = Field(
        default=None, validation_alias="AWS_ACCESS_KEY_ID"
    )
    aws_secret_access_key: str | None = Field(
        default=None, validation_alias="AWS_SECRET_ACCESS_KEY"
    )

    registry_dir: str = Field(default="artifacts/registry", validation_alias="REGISTRY_DIR")
    temp_upload_dir: str = Field(
        default="artifacts/uploads", validation_alias="TEMP_UPLOAD_DIR"
    )

    dataset_name: str = Field(default="ecg_img_v1", validation_alias="DATASET_NAME")
    shared_data_dir: str = Field(default="/shared-data", validation_alias="SHARED_DATA_DIR")
    shared_dataset_dir: str = Field(
        default="/shared-data/ecg_img",
        validation_alias="SHARED_DATASET_DIR",
    )

settings = Settings()