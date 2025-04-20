import yaml
import os
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution."""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            # Check if config file exists, if not create a default one
            if not os.path.exists(self.config_path):
                logger.info(f"Config file not found at {self.config_path}. Creating default configuration.")
                self._create_default_config()

            with open(self.config_path, 'r') as file:
                config_text = file.read()

            # Replace environment variables
            for env_var in os.environ:
                placeholder = f"${{{env_var}}}"
                if placeholder in config_text:
                    config_text = config_text.replace(placeholder, os.environ[env_var])

            return yaml.safe_load(config_text)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = self._get_default_config()

        with open(self.config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)

    def _get_default_config(self) -> dict:
        """Return default configuration."""
        return {
            'telegram': {
                'api_key': os.getenv('TELEGRAM_API_KEY', ''),
                'chat_id': '-4606618937'
            },
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'model': 'gpt-4o-mini',
                'max_tokens': 1024,
                'temperature': 0.1
            },
            'database': {
                'type': 'sqlite',
                'name': 'financial_monitor.db',
                'backup_enabled': True,
                'backup_path': './backups'
            },
            'news_collector': {
                'interval_seconds': 30,
                'business_hours': {
                    'start': 8,
                    'end': 15
                },
                'sources': ['benzinga', 'cnbc', 'yahoo_finance', 'google_business', 'cnn', 'fox'],
                'skip_domains': ['investors.com', 'wsj.com', 'bloomberg.com', 'investing.com']
            },
            'stock_collector': {
                'window_size': 30,
                'interval_seconds': 60,
                'default_watchlist': ['AAPL', 'GOOGL', 'MSFT']
            },
            'summarizer': {
                'max_sentences': 150,
                'batch_size': 8,
                'max_retries': 3
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'file': 'logs/application.log'
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value