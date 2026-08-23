from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    cors_origins: str = "http://localhost:5173"
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    nvidia_api_key: str = ""
    groq_api_key: str = ""
    windmill_base_url: str = ""
    windmill_token: str = ""
    windmill_workspace: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
