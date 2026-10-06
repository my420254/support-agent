from enum import StrEnum, auto


class Provider(StrEnum):
    OPENAI_COMPATIBLE = auto()
    DEEPSEEK = auto()
    FAKE = auto()


class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_V4_FLASH = "deepseek-v4-flash"
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"


class OpenAICompatibleName(StrEnum):
    """Any OpenAI-compatible endpoint (e.g. a self-hosted or third-party gateway)."""

    OPENAI_COMPATIBLE = "openai-compatible"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


type AllModelEnum = DeepseekModelName | OpenAICompatibleName | FakeModelName
