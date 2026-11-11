"""Tests for R4 dead-ball mask generation."""
import pytest
import numpy as np
from unittest.mock import patch

from mtah.deadball import dead_mask

def test_dead_mask_marks_whole_intervals():
    """Every frame of each inclusive interval is dead; trimming windows is the caller's job."""
    T = 200
    intervals = [(50, 60), (100, 150)]
    with patch("mtah.deadball.SECOND_HALF_D_BLOCK", None):
        mask = dead_mask(T, intervals)

    assert mask.shape == (T,) and mask.dtype == bool
    assert np.all(mask[50:61]) and np.all(mask[100:151])
    assert not np.any(mask[0:50]) and not np.any(mask[61:100]) and not np.any(mask[151:])

def test_dead_mask_with_config_block():
    """Test that the hardcoded second half block is applied correctly."""
    T = 500
    intervals = []
    
    # Mock the config block to a smaller manageable range for testing
    with patch("mtah.deadball.SECOND_HALF_D_BLOCK", (300, 400)):
        mask = dead_mask(T, intervals)
        
    # The config block is inclusive, so 300 to 400 should all be True
    assert np.all(mask[300:401]), "Frames inside the config block should be True"
    
    # Everything else should be False
    assert not np.any(mask[0:300])
    assert not np.any(mask[401:])

def test_dead_mask_out_of_bounds_interval():
    """Test that intervals exceeding the total frames T raise a ValueError."""
    T = 100
    # 110 is greater than T, which should trigger the safety check
    intervals = [(90, 110)]
    
    with patch("mtah.deadball.SECOND_HALF_D_BLOCK", None):
        with pytest.raises(ValueError, match="is not inside"):
            dead_mask(T, intervals)