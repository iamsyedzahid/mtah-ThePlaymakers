"""Tests for R4 possession segmentation (Viterbi Decoder)."""
import pytest
import numpy as np

from mtah.interfaces import Touch, HOME, AWAY, H, A, D
from mtah.segment import segment

def test_segment_invalid_shapes():
    """Test that mismatched array lengths raise a ValueError."""
    e = np.zeros(10)
    lost = np.zeros(9, dtype=bool)
    dead = np.zeros(10, dtype=bool)
    
    with pytest.raises(ValueError, match="must all be \\[T\\]"):
        segment(e, lost, dead, [])

def test_segment_touch_out_of_bounds():
    """Test that a touch recorded outside the frame limits raises an error."""
    T = 10
    e = np.zeros(T)
    lost = np.zeros(T, dtype=bool)
    dead = np.zeros(T, dtype=bool)
    touches = [Touch(frame=15, team=HOME, kind="reception")]
    
    with pytest.raises(ValueError, match="outside"):
        segment(e, lost, dead, touches)

def test_segment_pure_evidence_no_touches():
    """Test that with no touches, strong evidence dictates the steady state."""
    T = 20
    e = np.full(T, 0.8)  # Strongly favors Home (+ve)
    lost = np.zeros(T, dtype=bool)
    dead = np.zeros(T, dtype=bool)
    
    labels = segment(e, lost, dead, [])
    assert np.all(labels == H), "Possession should be entirely Home"

def test_segment_dead_mask_enforcement():
    """Test that dead_mask forcibly overrides the evidence to output 'D'."""
    T = 10
    e = np.ones(T)  # Favors Home
    lost = np.zeros(T, dtype=bool)
    dead = np.zeros(T, dtype=bool)
    
    # Enforce Dead ball from frame 4 to 6
    dead[4:7] = True 
    
    labels = segment(e, lost, dead, [])
    assert np.all(labels[0:4] == H)
    assert np.all(labels[4:7] == D), "dead_mask frames must be labeled 'D'"
    assert np.all(labels[7:] == H)

def test_segment_flip_on_reception():
    """A reception by the other team flips possession at its own frame."""
    T = 200  # runs must be longer than MIN_RUN_FRAMES, or they are absorbed as flicker
    e = np.concatenate([np.full(100, 0.8), np.full(100, -0.8)])
    lost = np.zeros(T, dtype=bool)
    dead = np.zeros(T, dtype=bool)
    touches = [Touch(frame=100, team=AWAY, kind="reception")]

    labels = segment(e, lost, dead, touches)

    assert np.all(labels[0:100] == H)
    assert np.all(labels[100:] == A), "Possession should flip to Away at the reception"

def test_segment_no_flip_on_deflection():
    """A deflection never flips possession, whatever the evidence does."""
    T = 200
    e = np.concatenate([np.full(150, 0.8), np.full(50, -0.8)])
    lost = np.zeros(T, dtype=bool)
    dead = np.zeros(T, dtype=bool)
    touches = [Touch(frame=150, team=AWAY, kind="deflection")]

    labels = segment(e, lost, dead, touches)

    assert np.all(labels == H), "Deflections should not allow possession flips"

def test_segment_lost_mask_carries_possession():
    """Test that lost_mask forces the algorithm to carry the previous state."""
    T = 20
    e = np.full(T, 0.8)
    lost = np.zeros(T, dtype=bool)
    dead = np.zeros(T, dtype=bool)
    
    # Ball is lost for 5 frames in the middle
    lost[10:15] = True
    
    labels = segment(e, lost, dead, [])
    assert np.all(labels == H), "Possession should seamlessly bridge the lost gap"