"""Tests for R4 dead-ball mask generation."""
import pytest
import numpy as np
from unittest.mock import patch

from mtah.deadball import dead_mask

def test_dead_mask_set_piece_midpoint():
    """Test that set piece intervals mark only the first half as dead (D)."""
    T = 200
    # Interval 1: 50 to 60 (midpoint 55). Dead: 50-54. Live: 55-60.
    # Interval 2: 100 to 150 (midpoint 125). Dead: 100-124. Live: 125-150.
    intervals = [(50, 60), (100, 150)]
    
    # Patch the config block to None so it doesn't interfere with this test
    with patch("mtah.deadball.SECOND_HALF_D_BLOCK", None):
        mask = dead_mask(T, intervals)
        
    assert mask.shape == (T,)
    assert mask.dtype == bool
    
    # Verify first interval
    assert np.all(mask[50:55]), "First half of interval 1 should be True"
    assert not np.any(mask[55:61]), "Second half of interval 1 should be False"
    
    # Verify second interval
    assert np.all(mask[100:125]), "First half of interval 2 should be True"
    assert not np.any(mask[125:151]), "Second half of interval 2 should be False"
    
    # Verify outside frames are strictly False
    assert not np.any(mask[0:50])
    assert not np.any(mask[61:100])
    assert not np.any(mask[151:])

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