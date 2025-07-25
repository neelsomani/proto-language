"""
Test configuration and fixtures for the proto-language test suite.
"""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4


@pytest.fixture(autouse=True)
def mock_celery():
    """Mock the task queue and a cache dependencies for all tests."""
    
    # Mock the task queue task
    mock_task = Mock()
    mock_task.delay.return_value = Mock(id=str(uuid4()))
    
    # Mock the task queue app
    mock_celery_app = Mock()
    mock_celery_app.control.inspect.return_value.active.return_value = {"worker1": []}
    mock_celery_app.AsyncResult.return_value = Mock(status="PENDING")
    
    # Patch imports in the API module
    with patch("api.main.celery_app", mock_celery_app), \
         patch("api.main.run_program_task", mock_task), \
         patch("api.celery_config.celery_app", mock_celery_app):
        yield {
            "celery_app": mock_celery_app,
            "task": mock_task
        }


@pytest.fixture(autouse=True) 
def mock_redis():
    """Mock a cache connections."""
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    
    with patch("cache.a cache", return_value=mock_redis):
        yield mock_redis