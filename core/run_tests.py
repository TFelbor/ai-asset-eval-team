#!/usr/bin/env python3
"""
Run all the tests for the financial analysis dashboard.
"""
import unittest
import sys
import os

# Add the parent directory to the path so we can import the test modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import the test modules
from tests.test_api import TestAPI
from tests.test_agents import TestFundamentalAnalyst, TestCryptoAnalyst, TestMacroSentimentAnalyst
from tests.test_cache_manager import TestCacheManager

if __name__ == "__main__":
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add the test cases
    test_suite.addTest(unittest.makeSuite(TestAPI))
    test_suite.addTest(unittest.makeSuite(TestFundamentalAnalyst))
    test_suite.addTest(unittest.makeSuite(TestCryptoAnalyst))
    test_suite.addTest(unittest.makeSuite(TestMacroSentimentAnalyst))
    test_suite.addTest(unittest.makeSuite(TestCacheManager))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with a non-zero code if there were failures
    sys.exit(not result.wasSuccessful())
