# =============================================================================
# Claude API Client
# =============================================================================
# Wrapper around the Anthropic API for interacting with Claude.
#
# Features:
#   - Automatic model selection based on task complexity
#   - Response caching to reduce costs
#   - Rate limiting to avoid hitting API limits
#   - Retry logic for transient failures
#   - Cost tracking
#
# This client is used by all AI features in the system.
# =============================================================================

import os
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Import the Anthropic SDK
try:
    import anthropic
except ImportError:
    anthropic = None
    logging.warning(
        "anthropic package not installed. "
        "Install with: pip install anthropic"
    )

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configurations with costs per 1M tokens (as of 2024)
MODELS = {
    # Fast, cheap model for simple tasks
    "haiku": {
        "id": "claude-3-haiku-20240307",
        "input_cost": 0.25,   # $ per 1M input tokens
        "output_cost": 1.25,  # $ per 1M output tokens
        "max_tokens": 4096,
        "use_for": ["quality_check", "news_summary", "simple_questions"]
    },
    # Balanced model for most tasks
    "sonnet": {
        "id": "claude-sonnet-4-20250514",
        "input_cost": 3.00,
        "output_cost": 15.00,
        "max_tokens": 4096,
        "use_for": ["anomaly_detection", "report_generation", "complex_analysis"]
    },
    # Most capable model for complex reasoning
    "opus": {
        "id": "claude-3-opus-20240229",
        "input_cost": 15.00,
        "output_cost": 75.00,
        "max_tokens": 4096,
        "use_for": ["strategic_insights", "complex_queries"]
    }
}

# Default rate limits
DEFAULT_RATE_LIMIT = 10  # requests per minute


# =============================================================================
# CLAUDE CLIENT CLASS
# =============================================================================

class ClaudeClient:
    """
    Client for interacting with Claude AI.

    This class provides a simplified interface to the Anthropic API with
    added features like caching, rate limiting, and cost tracking.

    Example usage:
        client = ClaudeClient()

        # Simple message
        response = client.send_message(
            "What are the key indicators of tourism health?",
            model="haiku"
        )
        print(response)

        # With system prompt
        response = client.send_message(
            "Analyze this tourism data: ...",
            system="You are an economic analyst specializing in tourism.",
            model="sonnet"
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT
    ):
        """
        Initialize the Claude client.

        Args:
            api_key: Anthropic API key. If not provided, reads from
                    ANTHROPIC_API_KEY environment variable.
            cache_enabled: Whether to cache responses
            rate_limit_per_minute: Maximum requests per minute
        """
        if anthropic is None:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        # Get API key
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Caching
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry_hours = 168  # 1 week

        # Rate limiting
        self.rate_limit = rate_limit_per_minute
        self._request_times: List[datetime] = []

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        logger.info("Claude client initialized")

    # =========================================================================
    # MAIN API METHODS
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.APIError if anthropic else Exception,))
    )
    def send_message(
        self,
        message: str,
        system: Optional[str] = None,
        model: str = "sonnet",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> str:
        """
        Send a message to Claude and get a response.

        Args:
            message: The user message to send
            system: Optional system prompt to set context
            model: Model to use ("haiku", "sonnet", or "opus")
            max_tokens: Maximum tokens in response
            temperature: Creativity (0-1, lower = more deterministic)
            use_cache: Whether to use cached response if available

        Returns:
            Claude's response text

        Example:
            response = client.send_message(
                "Summarize this news article: ...",
                system="You are a journalist summarizing economic news.",
                model="haiku"
            )
        """
        # Check cache first
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(message, system, model)
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.debug("Using cached response")
                return cached

        # Apply rate limiting
        self._wait_for_rate_limit()

        # Get model configuration
        model_config = MODELS.get(model, MODELS["sonnet"])
        model_id = model_config["id"]

        try:
            # Make the API call
            logger.info(f"Sending message to Claude ({model})")

            response = self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful AI assistant.",
                messages=[
                    {"role": "user", "content": message}
                ]
            )

            # Extract response text
            response_text = response.content[0].text

            # Track usage
            self._track_usage(response, model_config)

            # Cache the response
            if use_cache and self.cache_enabled:
                self._add_to_cache(cache_key, response_text)

            return response_text

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def send_structured_message(
        self,
        message: str,
        system: Optional[str] = None,
        model: str = "sonnet",
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Send a message expecting structured (JSON) output.

        Args:
            message: The user message
            system: System prompt
            model: Model to use
            output_format: Expected format (currently only "json")

        Returns:
            Parsed JSON response as dictionary

        Example:
            result = client.send_structured_message(
                "Analyze this data and return key metrics as JSON: ...",
                system="Return your analysis as valid JSON."
            )
            print(result["metrics"])
        """
        import json

        # Add instruction for JSON output
        enhanced_system = (system or "") + "\n\nIMPORTANT: Respond with valid JSON only."

        response = self.send_message(
            message,
            system=enhanced_system,
            model=model,
            temperature=0.3  # Lower temperature for more consistent structure
        )

        # Try to parse JSON from response
        try:
            # Find JSON in response (might be wrapped in markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            return json.loads(json_str.strip())

        except json.JSONDecodeError:
            logger.warning("Could not parse JSON response, returning raw text")
            return {"raw_response": response}

    # =========================================================================
    # MODEL SELECTION
    # =========================================================================

    def select_model_for_task(self, task_type: str) -> str:
        """
        Automatically select the best model for a task.

        This implements the "balanced" cost strategy - using cheaper models
        for simple tasks and more capable models for complex ones.

        Args:
            task_type: Type of task (e.g., "quality_check", "anomaly_detection")

        Returns:
            Model name to use ("haiku", "sonnet", or "opus")

        Example:
            model = client.select_model_for_task("news_summary")
            # Returns "haiku" for simple summarization
        """
        for model_name, config in MODELS.items():
            if task_type in config.get("use_for", []):
                logger.debug(f"Selected {model_name} for task: {task_type}")
                return model_name

        # Default to sonnet for unknown tasks
        return "sonnet"

    # =========================================================================
    # CACHING
    # =========================================================================

    def _get_cache_key(
        self,
        message: str,
        system: Optional[str],
        model: str
    ) -> str:
        """Generate a unique cache key for a request."""
        content = f"{model}:{system or ''}:{message}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get a response from cache if valid."""
        if key not in self._cache:
            return None

        cached = self._cache[key]
        expiry = cached.get("expiry")

        if expiry and datetime.now() > expiry:
            # Cache expired
            del self._cache[key]
            return None

        return cached.get("response")

    def _add_to_cache(self, key: str, response: str) -> None:
        """Add a response to cache."""
        self._cache[key] = {
            "response": response,
            "expiry": datetime.now() + timedelta(hours=self._cache_expiry_hours),
            "created": datetime.now()
        }

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self._cache = {}
        logger.info("Cache cleared")

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    def _wait_for_rate_limit(self) -> None:
        """Wait if we've exceeded the rate limit."""
        now = datetime.now()

        # Remove old request times (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self._request_times = [t for t in self._request_times if t > cutoff]

        # Check if we're at the limit
        if len(self._request_times) >= self.rate_limit:
            # Calculate how long to wait
            oldest = min(self._request_times)
            wait_time = 60 - (now - oldest).total_seconds()

            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)

        # Record this request
        self._request_times.append(now)

    # =========================================================================
    # COST TRACKING
    # =========================================================================

    def _track_usage(
        self,
        response: Any,
        model_config: Dict[str, Any]
    ) -> None:
        """Track token usage and costs."""
        usage = response.usage

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * model_config["input_cost"]
        output_cost = (output_tokens / 1_000_000) * model_config["output_cost"]
        request_cost = input_cost + output_cost

        self.total_cost += request_cost

        logger.debug(
            f"Request: {input_tokens} in, {output_tokens} out, "
            f"${request_cost:.4f}"
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "cache_size": len(self._cache)
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> bool:
        """
        Check if the Claude API is accessible.

        Returns:
            True if API is working, False otherwise
        """
        try:
            response = self.send_message(
                "Respond with 'OK' if you can read this.",
                model="haiku",
                max_tokens=10,
                use_cache=False
            )
            return "OK" in response.upper()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Claude Client...")

    # Check if API key is available
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping live tests.")
        print("   Set the environment variable to test the client.")
    else:
        try:
            client = ClaudeClient()

            print("\n1. Health Check:")
            is_healthy = client.health_check()
            print(f"   API Status: {'✅ OK' if is_healthy else '❌ Failed'}")

            print("\n2. Simple Message (Haiku):")
            response = client.send_message(
                "What are 3 key economic indicators for tourism?",
                model="haiku",
                max_tokens=200
            )
            print(f"   Response: {response[:200]}...")

            print("\n3. Usage Stats:")
            stats = client.get_usage_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ Error: {e}")
