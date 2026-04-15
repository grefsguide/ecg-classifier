from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    telegram_bot_token: str = Field(validation_alias="TG_BOT_TOKEN")
    api_base_url: str = Field(validation_alias="BOT_API_BASE_URL")

    bot_temp_dir: str = Field(default="/tmp/telegram-bot", validation_alias="BOT_TEMP_DIR")
    poll_interval_seconds: float = Field(default=1.5, validation_alias="BOT_POLL_INTERVAL_SECONDS")
    poll_timeout_seconds: int = Field(default=180, validation_alias="BOT_POLL_TIMEOUT_SECONDS")


settings = BotSettings()