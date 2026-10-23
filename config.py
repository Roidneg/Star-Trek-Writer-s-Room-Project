"""Logical model names shared by the CLI and specialist agents.

Provider credentials belong in environment variables, never in this file.
The provider adapter maps these logical names to Cloudflare identifiers.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
LORE_MODEL = os.getenv("LORE_MODEL", DEFAULT_MODEL)
CHARACTER_MODEL = os.getenv("CHARACTER_MODEL", DEFAULT_MODEL)
STORY_ARCHITECT_MODEL = os.getenv("STORY_ARCHITECT_MODEL", DEFAULT_MODEL)
SCRIPTWRITER_MODEL = os.getenv("SCRIPTWRITER_MODEL", DEFAULT_MODEL)
CONTINUITY_MODEL = os.getenv("CONTINUITY_MODEL", DEFAULT_MODEL)
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", DEFAULT_MODEL)
