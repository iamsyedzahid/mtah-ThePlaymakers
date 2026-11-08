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
    """Test that a 'reception' touch allows possession to flip based on evidence."""
    T = 20
    # First 10 frames favor Home (+0.8), next 10 favor Away (-0.8)
    e = np.concatenate([np.full(10, 0.8), np.full(10, -0.8)])
    lost = np.zeros(T, dtype=bool)
    dead = np.zeros(T, dtype=bool)
    
    # A valid reception touch occurs right as evidence flips
    touches = [Touch(frame=10, team=AWAY, kind="reception")]
    
    labels = segment(e, lost, dead, touches)
    
    assert np.all(labels[0:10] == H)
    assert np.all(labels[10:] == A), "Possession should flip to Away after reception"

def test_segment_no_flip_on_deflection():
    """Test that a 'deflection' touch blocks possession from flipping."""
    T = 20
    # Evidence strongly shifts to Away...
    e = np.concatenate([np.full(10, 0.8), np.full(10, -0.8)])
    lost = np.zeros(T, dtype=bool)
    dead = np.zeros(T, dtype=bool)
    
    # ...but it's only a deflection!
    touches = [Touch(frame=10, team=AWAY, kind="deflection")]
    
    labels = segment(e, lost, dead, touches)
    
    # Viterbi switching penalty should prevent the flip
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