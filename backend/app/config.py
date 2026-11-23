from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    supabase_url: str
    supabase_publishable_key: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str
    cors_origins: str = "http://localhost:5173"
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    nvidia_api_key: str = ""
    groq_api_key: str = ""
    windmill_base_url: str = ""
    windmill_token: str = ""
    windmill_workspace: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def supabase_public_key(self) -> str:
        return self.supabase_publishable_key or self.supabase_anon_key

    @property
    def smtp_configured(self) -> bool:
        return all(
            (
                self.smtp_host,
                self.smtp_username,
                self.smtp_password,
                self.smtp_from_email,
            )
        )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
