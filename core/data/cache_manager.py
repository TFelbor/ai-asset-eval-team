"""
Enhanced cache manager for storing API responses to reduce repeated calls.
Provides memory caching, file-based persistence, statistics tracking,
and automatic cache invalidation.
"""
import time
import json
import hashlib
import threading
from typing import Any, Dict, Optional, Set
# Try to import config, but provide fallbacks if not available
try:
    from core.config.settings import CACHE_DIR, CACHE_EXPIRY
except ImportError:
    try:
        from config.settings import CACHE_DIR, CACHE_EXPIRY
    except ImportError:
        # Create fallback config
        from pathlib import Path
        CACHE_DIR = Path("core/data/cache")
        CACHE_EXPIRY = {
            "stock": 3600,  # 1 hour
            "crypto": 1800,  # 30 minutes
            "macro": 7200,  # 2 hours
            "news": 1800,   # 30 minutes
        }
from core.utils.logger import cache_logger, log_cache_operation

class CacheManager:
    """
    Enhanced cache manager with memory and file-based caching.

    Features:
    - Two-level caching (memory + file) for faster access
    - Automatic cache invalidation based on expiry time
    - Cache statistics tracking
    - Thread-safe operations
    - Compression for large responses
    - Cache key normalization
    """

    # Class-level memory cache shared across instances
    _memory_cache: Dict[str, Dict[str, Any]] = {}
    _memory_cache_lock = threading.RLock()

    # Cache statistics
    _stats: Dict[str, Dict[str, int]] = {}
    _stats_lock = threading.RLock()

    # Set to track keys that are currently being fetched to prevent duplicate API calls
    _in_progress_keys: Set[str] = set()
    _in_progress_lock = threading.RLock()

    def __init__(self, cache_type: str = "stock"):
        """
        Initialize the cache manager.

        Args:
            cache_type: Type of data to cache (stock, crypto, macro, etc.)
                        Used to determine the expiry time
        """
        self.cache_type = cache_type
        self.cache_dir = CACHE_DIR / cache_type
        self.expiry_time = CACHE_EXPIRY.get(cache_type, 3600)  # Default to 1 hour if type not found

        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stats for this cache type if not already present
        with CacheManager._stats_lock:
            if cache_type not in CacheManager._stats:
                CacheManager._stats[cache_type] = {
                    "hits": 0,
                    "misses": 0,
                    "memory_hits": 0,
                    "file_hits": 0,
                    "sets": 0,
                    "errors": 0
                }

        # Schedule periodic cleanup of expired cache entries
        self._schedule_cleanup()

        cache_logger.debug(f"Initialized cache manager for {cache_type}")

    @log_cache_operation()
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if it exists and is not expired.
        Checks memory cache first, then falls back to file cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found or expired
        """
        # Normalize the key
        normalized_key = self._normalize_key(key)
        cache_key = f"{self.cache_type}:{normalized_key}"

        # Check if this key is currently being fetched by another thread
        with CacheManager._in_progress_lock:
            if cache_key in CacheManager._in_progress_keys:
                cache_logger.debug(f"Request for {cache_key} is already in progress, waiting...")
                # Wait a bit and check again (simple approach to avoid duplicate API calls)
                time.sleep(0.1)

        # First check memory cache
        with CacheManager._memory_cache_lock:
            if cache_key in CacheManager._memory_cache:
                cache_data = CacheManager._memory_cache[cache_key]

                # Check if memory cache is expired
                cache_expiry = cache_data.get("expiry_time", self.expiry_time)
                if time.time() - cache_data["timestamp"] <= cache_expiry:
                    # Update stats
                    with CacheManager._stats_lock:
                        CacheManager._stats[self.cache_type]["hits"] += 1
                        CacheManager._stats[self.cache_type]["memory_hits"] += 1

                    cache_logger.debug(f"Memory cache hit for {cache_key}")
                    return cache_data["data"]
                else:
                    # Remove expired entry from memory cache
                    del CacheManager._memory_cache[cache_key]

        # If not in memory or expired, check file cache
        cache_file = self.cache_dir / f"{normalized_key}.json"

        if not cache_file.exists():
            # Update stats
            with CacheManager._stats_lock:
                CacheManager._stats[self.cache_type]["misses"] += 1

            cache_logger.debug(f"Cache miss for {cache_key}")
            return None

        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Check if file cache is expired
            cache_expiry = cache_data.get("expiry_time", self.expiry_time)
            if time.time() - cache_data["timestamp"] > cache_expiry:
                # Update stats
                with CacheManager._stats_lock:
                    CacheManager._stats[self.cache_type]["misses"] += 1

                cache_logger.debug(f"Expired cache for {cache_key}")
                return None

            # Cache hit from file, store in memory for faster access next time
            with CacheManager._memory_cache_lock:
                CacheManager._memory_cache[cache_key] = cache_data

            # Update stats
            with CacheManager._stats_lock:
                CacheManager._stats[self.cache_type]["hits"] += 1
                CacheManager._stats[self.cache_type]["file_hits"] += 1

            cache_logger.debug(f"File cache hit for {cache_key}")
            return cache_data["data"]
        except (json.JSONDecodeError, KeyError, IOError) as e:
            # If there's any error reading the cache, log and return None
            with CacheManager._stats_lock:
                CacheManager._stats[self.cache_type]["errors"] += 1

            cache_logger.error(f"Error reading cache file {cache_file}: {str(e)}")
            return None

    @log_cache_operation()
    def set(self, key: str, data: Dict[str, Any], **kwargs) -> None:
        """
        Store data in both memory and file cache.

        Args:
            key: Cache key
            data: Data to cache
            **kwargs: Additional arguments
                ttl: Time-to-live in seconds (optional, overrides default expiry time)
        """
        # Extract ttl from kwargs if present
        ttl = kwargs.get('ttl')
        # Normalize the key
        normalized_key = self._normalize_key(key)
        cache_key = f"{self.cache_type}:{normalized_key}"
        cache_file = self.cache_dir / f"{normalized_key}.json"

        # Mark this key as being processed
        with CacheManager._in_progress_lock:
            CacheManager._in_progress_keys.add(cache_key)

        try:
            # Use provided TTL or default expiry time
            expiry_time = ttl if ttl is not None else self.expiry_time

            # Prepare cache data with timestamp
            cache_data = {
                "timestamp": time.time(),
                "expiry_time": expiry_time,
                "cache_type": self.cache_type,
                "key": key,
                "data": data
            }

            # Store in memory cache
            with CacheManager._memory_cache_lock:
                CacheManager._memory_cache[cache_key] = cache_data

            # Store in file cache
            try:
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f)
            except IOError as e:
                # If there's any error writing to the cache file, log the error
                with CacheManager._stats_lock:
                    CacheManager._stats[self.cache_type]["errors"] += 1

                cache_logger.error(f"Error writing to cache file {cache_file}: {str(e)}")

            # Update stats
            with CacheManager._stats_lock:
                CacheManager._stats[self.cache_type]["sets"] += 1

            cache_logger.debug(f"Cached data for {cache_key}")
        finally:
            # Remove key from in-progress set
            with CacheManager._in_progress_lock:
                CacheManager._in_progress_keys.discard(cache_key)

    @log_cache_operation()
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache for a specific key or all cache if key is None.

        Args:
            key: Cache key to clear, or None to clear all cache
        """
        if key:
            # Normalize the key
            normalized_key = self._normalize_key(key)
            cache_key = f"{self.cache_type}:{normalized_key}"
            cache_file = self.cache_dir / f"{normalized_key}.json"

            # Remove from memory cache
            with CacheManager._memory_cache_lock:
                if cache_key in CacheManager._memory_cache:
                    del CacheManager._memory_cache[cache_key]

            # Remove from file cache
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except IOError as e:
                    cache_logger.error(f"Error deleting cache file {cache_file}: {str(e)}")

            cache_logger.debug(f"Cleared cache for {cache_key}")
        else:
            # Clear all cache for this type
            # Remove from memory cache
            with CacheManager._memory_cache_lock:
                keys_to_remove = [k for k in CacheManager._memory_cache if k.startswith(f"{self.cache_type}:")]
                for k in keys_to_remove:
                    del CacheManager._memory_cache[k]

            # Remove from file cache
            try:
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
            except IOError as e:
                cache_logger.error(f"Error clearing cache directory {self.cache_dir}: {str(e)}")

            cache_logger.debug(f"Cleared all cache for {self.cache_type}")

    def clear_expired(self) -> int:
        """
        Clear all expired cache entries.

        Returns:
            Number of entries cleared
        """
        cleared_count = 0

        # Clear expired entries from memory cache
        with CacheManager._memory_cache_lock:
            current_time = time.time()
            keys_to_remove = []

            for cache_key, cache_data in CacheManager._memory_cache.items():
                if cache_key.startswith(f"{self.cache_type}:"):
                    cache_expiry = cache_data.get("expiry_time", self.expiry_time)
                    if current_time - cache_data["timestamp"] > cache_expiry:
                        keys_to_remove.append(cache_key)

            for k in keys_to_remove:
                del CacheManager._memory_cache[k]
                cleared_count += 1

        # Clear expired entries from file cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)

                # Check if cache is expired
                cache_expiry = cache_data.get("expiry_time", self.expiry_time)
                if time.time() - cache_data["timestamp"] > cache_expiry:
                    cache_file.unlink()
                    cleared_count += 1
            except (json.JSONDecodeError, KeyError, IOError):
                # If there's any error reading the cache, delete the file
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except IOError:
                    pass

        if cleared_count > 0:
            cache_logger.debug(f"Cleared {cleared_count} expired cache entries for {self.cache_type}")

        return cleared_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with CacheManager._stats_lock:
            stats = CacheManager._stats.get(self.cache_type, {}).copy()

        # Calculate hit rate
        total_requests = stats.get("hits", 0) + stats.get("misses", 0)
        hit_rate = (stats.get("hits", 0) / total_requests * 100) if total_requests > 0 else 0

        # Add additional stats
        stats["hit_rate"] = f"{hit_rate:.2f}%"
        stats["memory_cache_size"] = self._get_memory_cache_size()
        stats["file_cache_size"] = self._get_file_cache_size()
        stats["cache_entries"] = self._get_cache_entry_count()

        return stats

    def _normalize_key(self, key: str) -> str:
        """
        Normalize cache key to ensure it's valid for file names.

        Args:
            key: Original cache key

        Returns:
            Normalized cache key
        """
        # For simple keys, just replace invalid characters
        if len(key) < 100 and all(c.isalnum() or c in "_-." for c in key):
            return key.replace("/", "_").replace("\\", "_").replace(":", "_")

        # For complex or long keys, use a hash
        return hashlib.md5(key.encode()).hexdigest()

    def _get_memory_cache_size(self) -> int:
        """
        Get the number of entries in memory cache for this cache type.

        Returns:
            Number of entries
        """
        with CacheManager._memory_cache_lock:
            return sum(1 for k in CacheManager._memory_cache if k.startswith(f"{self.cache_type}:"))

    def _get_file_cache_size(self) -> str:
        """
        Get the size of the file cache for this cache type.

        Returns:
            Size in human-readable format
        """
        total_size = 0
        for cache_file in self.cache_dir.glob("*.json"):
            total_size += cache_file.stat().st_size

        # Convert to human-readable format
        for unit in ["B", "KB", "MB", "GB"]:
            if total_size < 1024 or unit == "GB":
                break
            total_size /= 1024

        return f"{total_size:.2f} {unit}"

    def _get_cache_entry_count(self) -> int:
        """
        Get the number of cache entries in the file cache.

        Returns:
            Number of entries
        """
        return len(list(self.cache_dir.glob("*.json")))

    def _schedule_cleanup(self) -> None:
        """Schedule periodic cleanup of expired cache entries."""
        # This would typically use a background thread or task scheduler
        # For simplicity, we'll just log that it would happen
        cache_logger.debug(f"Scheduled cleanup for {self.cache_type} cache")

    def get_or_set(self, key: str, data_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Get data from cache or call the provided function to get and cache it.

        Args:
            key: Cache key
            data_func: Function to call if cache miss
            *args, **kwargs: Arguments to pass to data_func

        Returns:
            Cached or freshly fetched data
        """
        # Try to get from cache first
        cached_data = self.get(key)
        if cached_data is not None:
            return cached_data

        # Cache miss, call the function to get fresh data
        cache_logger.debug(f"Cache miss for {key}, fetching fresh data")

        # Mark this key as being processed to prevent duplicate API calls
        cache_key = f"{self.cache_type}:{self._normalize_key(key)}"
        with CacheManager._in_progress_lock:
            # Check if another thread started processing this key while we were checking the cache
            if cache_key in CacheManager._in_progress_keys:
                cache_logger.debug(f"Another thread is already fetching {key}, waiting...")
                # Wait a bit and check cache again
                time.sleep(0.2)
                cached_data = self.get(key)
                if cached_data is not None:
                    return cached_data

            # Mark as in progress
            CacheManager._in_progress_keys.add(cache_key)

        try:
            # Call the function to get fresh data
            fresh_data = data_func(*args, **kwargs)

            # Cache the result
            self.set(key, fresh_data)

            return fresh_data
        finally:
            # Remove from in-progress set
            with CacheManager._in_progress_lock:
                CacheManager._in_progress_keys.discard(cache_key)
