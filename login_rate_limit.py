
import math
import time
from threading import Lock


class LoginRateLimiter:
    """
    Simple in-memory login rate limiter.

    Limits login attempts per client IP address.

    Default:
        5 attempts per 15 minutes.

    Returns:
        0 = attempt allowed
        Positive integer = retry after N seconds
    """

    def __init__(
        self,
        max_attempts=5,
        window_seconds=900
    ):
        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be positive."
            )

        if window_seconds < 1:
            raise ValueError(
                "window_seconds must be positive."
            )

        self.max_attempts = max_attempts
        self.window_seconds = window_seconds

        self._attempts = {}
        self._lock = Lock()

    def consume(self, client_ip, now=None):
        """
        Record a login attempt if allowed.

        Returns 0 when allowed.
        Returns retry-after seconds when blocked.

        The check and recording happen under one lock
        to avoid concurrent attempts bypassing the limit.
        """

        if now is None:
            now = time.monotonic()

        client_ip = str(client_ip or "unknown")

        with self._lock:

            previous = self._attempts.get(
                client_ip, []
            )

            # Retain only attempts inside the window.
            recent = [
                timestamp
                for timestamp in previous
                if 0 <= now - timestamp < self.window_seconds
            ]

            # Reject requests after the allowed limit.
            if len(recent) >= self.max_attempts:

                oldest = recent[0]

                retry_after = math.ceil(
                    self.window_seconds - (now - oldest)
                )

                self._attempts[client_ip] = recent

                return max(1, retry_after)

            # Reserve a login attempt.
            recent.append(now)

            self._attempts[client_ip] = recent

            return 0

    def reset(self, client_ip):
        """
        Clear attempt history after successful login.
        """

        client_ip = str(client_ip or "unknown")

        with self._lock:
            self._attempts.pop(client_ip, None)


# Shared limiter used later by the login endpoint.
login_limiter = LoginRateLimiter(
    max_attempts=5,
    window_seconds=15 * 60
)
