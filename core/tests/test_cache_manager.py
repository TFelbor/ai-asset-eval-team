"""
Unit tests for the cache manager.
"""
import sys
import os
import unittest
import tempfile
import shutil
import time
from pathlib import Path

# Add the parent directory to the path so we can import the cache_manager module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cache_manager import CacheManager

class TestCacheManager(unittest.TestCase):
    """Test the CacheManager class."""
    
    def setUp(self):
        """Set up a temporary cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Patch the CACHE_DIR in the cache_manager module
        import config
        self.original_cache_dir = config.CACHE_DIR
        config.CACHE_DIR = self.cache_dir
        
    def tearDown(self):
        """Clean up the temporary cache directory."""
        # Restore the original CACHE_DIR
        import config
        config.CACHE_DIR = self.original_cache_dir
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_get_nonexistent_key(self):
        """Test getting a nonexistent key."""
        cache = CacheManager(cache_type="test")
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)
        
    def test_set_and_get(self):
        """Test setting and getting a key."""
        cache = CacheManager(cache_type="test")
        data = {"test": "data"}
        cache.set("test_key", data)
        
        # Check that the file was created
        cache_file = self.cache_dir / "test_key.json"
        self.assertTrue(cache_file.exists())
        
        # Get the data
        result = cache.get("test_key")
        self.assertEqual(result, data)
        
    def test_expiry(self):
        """Test that the cache expires."""
        # Create a cache with a very short expiry time
        cache = CacheManager(cache_type="test")
        
        # Override the expiry time
        cache.expiry_time = 1  # 1 second
        
        # Set a key
        data = {"test": "data"}
        cache.set("test_key", data)
        
        # Get the data immediately
        result = cache.get("test_key")
        self.assertEqual(result, data)
        
        # Wait for the cache to expire
        time.sleep(2)
        
        # Get the data again
        result = cache.get("test_key")
        self.assertIsNone(result)
        
    def test_clear(self):
        """Test clearing the cache."""
        cache = CacheManager(cache_type="test")
        
        # Set some keys
        cache.set("test_key1", {"test": "data1"})
        cache.set("test_key2", {"test": "data2"})
        
        # Check that the files were created
        cache_file1 = self.cache_dir / "test_key1.json"
        cache_file2 = self.cache_dir / "test_key2.json"
        self.assertTrue(cache_file1.exists())
        self.assertTrue(cache_file2.exists())
        
        # Clear a specific key
        cache.clear("test_key1")
        
        # Check that the file was removed
        self.assertFalse(cache_file1.exists())
        self.assertTrue(cache_file2.exists())
        
        # Clear all keys
        cache.clear()
        
        # Check that all files were removed
        self.assertFalse(cache_file1.exists())
        self.assertFalse(cache_file2.exists())
        
    def test_clear_expired(self):
        """Test clearing expired cache entries."""
        # Create a cache with a very short expiry time
        cache = CacheManager(cache_type="test")
        
        # Override the expiry time
        cache.expiry_time = 1  # 1 second
        
        # Set some keys
        cache.set("test_key1", {"test": "data1"})
        time.sleep(2)  # Wait for the first key to expire
        cache.set("test_key2", {"test": "data2"})
        
        # Check that the files were created
        cache_file1 = self.cache_dir / "test_key1.json"
        cache_file2 = self.cache_dir / "test_key2.json"
        self.assertTrue(cache_file1.exists())
        self.assertTrue(cache_file2.exists())
        
        # Clear expired entries
        cache.clear_expired()
        
        # Check that only the expired file was removed
        self.assertFalse(cache_file1.exists())
        self.assertTrue(cache_file2.exists())
        
if __name__ == "__main__":
    unittest.main()
