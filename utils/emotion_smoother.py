"""
Emotion Smoother - Semester 2, Phase 2

Implements temporal smoothing to reduce flickering in emotion predictions.
Uses a moving average over the last N frames.

Why Smoothing?
- Emotion models give noisy frame-by-frame predictions
- A face might flicker between "Happy" and "Neutral" rapidly
- Averaging over multiple frames provides stable, readable output

This is similar to how video games smooth character movement
or how audio software applies low-pass filters.

Time: 30-45 minutes
TODOs: 3
"""

import numpy as np


class EmotionSmoother:
    """
    Smooths emotion predictions over time using a moving average.
    
    This implements a circular buffer (ring buffer) that stores the last N
    probability vectors. When you query for the smoothed result, it averages
    all stored vectors.
    
    Example:
        smoother = EmotionSmoother(window_size=5)
        
        # Each frame, add new prediction
        smoother.update(probabilities)
        
        # Get smoothed result
        smoothed_probs = smoother.get_smoothed()
        emotion = EMOTIONS[np.argmax(smoothed_probs)]
    """
    
    def __init__(self, window_size=5, num_classes=7):
        """
        Initialize the emotion smoother.
        
        Args:
            window_size: Number of frames to average (default: 5)
            num_classes: Number of emotion classes (default: 7)
        """
        
        # TODO 18: Initialize circular buffer
        # ====================================
        # A circular buffer is a fixed-size array where new data overwrites
        # the oldest data. It's memory-efficient for sliding window operations.
        #
        # Steps:
        # 1. Store parameters:
        #    - self.window_size = window_size
        #    - self.num_classes = num_classes
        #
        # 2. Create the buffer:
        #    - Create a numpy array of zeros with shape (window_size, num_classes)
        #    - This will hold the last `window_size` probability vectors
        #    - Store as self.buffer
        #
        # 3. Initialize position tracker:
        #    - self.position = 0  (where to write next)
        #    - self.count = 0     (how many entries we've added)
        #
        # Why a circular buffer?
        # - Fixed memory usage regardless of how long the program runs
        # - O(1) insertions - just write at current position
        # - No array resizing or shifting needed
        #
        # Visual example with window_size=3:
        #   Start:    [empty, empty, empty]  position=0, count=0
        #   Add A:    [A,     empty, empty]  position=1, count=1
        #   Add B:    [A,     B,     empty]  position=2, count=2
        #   Add C:    [A,     B,     C    ]  position=0, count=3 (wrapped!)
        #   Add D:    [D,     B,     C    ]  position=1, count=3 (overwrote A)
        
        raise NotImplementedError("TODO 18: Initialize circular buffer")
    
    def update(self, probabilities):
        """
        Add new probability vector to the buffer.
        
        Args:
            probabilities: Array of emotion probabilities, shape (num_classes,)
        """
        
        # TODO 19: Add new prediction to buffer
        # ======================================
        # Steps:
        # 1. Convert input to numpy array (in case it isn't already):
        #    - probabilities = np.array(probabilities)
        #
        # 2. Validate shape:
        #    - Check that len(probabilities) == self.num_classes
        #    - If not, print warning and return without updating
        #
        # 3. Write to buffer at current position:
        #    - self.buffer[self.position] = probabilities
        #
        # 4. Update position (wrap around using modulo):
        #    - self.position = (self.position + 1) % self.window_size
        #    - This makes position cycle: 0, 1, 2, 0, 1, 2, ...
        #
        # 5. Update count (cap at window_size):
        #    - self.count = min(self.count + 1, self.window_size)
        #    - count tells us how many valid entries exist
        #
        # Why modulo for position?
        # - When position reaches window_size, we want it to wrap to 0
        # - Example: window_size=3, position goes 0->1->2->0->1->2->...
        # - This is what makes it a "circular" buffer
        
        raise NotImplementedError("TODO 19: Add new prediction to buffer")
    
    def get_smoothed(self):
        """
        Get the smoothed probability vector.
        
        Returns:
            Averaged probability vector, shape (num_classes,)
            Returns uniform distribution if buffer is empty
        """
        
        # TODO 20: Return averaged probabilities
        # =======================================
        # Steps:
        # 1. Handle empty buffer case:
        #    - If self.count == 0, return uniform distribution
        #    - Uniform: np.ones(self.num_classes) / self.num_classes
        #    - This means "equal probability for all emotions"
        #
        # 2. Get valid portion of buffer:
        #    - Only average the entries we've actually filled
        #    - valid_entries = self.buffer[:self.count]
        #    - (If count < window_size, we haven't filled the buffer yet)
        #
        # 3. Compute mean across entries:
        #    - Use np.mean(valid_entries, axis=0)
        #    - axis=0 means average across rows (frames)
        #    - Result shape: (num_classes,)
        #
        # 4. Return the averaged probabilities
        #
        # Note: The averaged probabilities might not sum to exactly 1.0
        # due to floating point arithmetic, but they'll be very close.
        # This is fine for finding the max (argmax) emotion.
        #
        # Why average instead of voting?
        # - Averaging preserves confidence information
        # - "90% happy + 90% happy + 50% happy" → different from
        #   "90% happy + 90% happy + 90% sad" (tie in voting, clear win in avg)
        
        raise NotImplementedError("TODO 20: Return averaged probabilities")
    
    def get_emotion(self, emotions_list):
        """
        Convenience method: get the current smoothed emotion label.
        
        Args:
            emotions_list: List of emotion names (e.g., EMOTIONS from emotion_model.py)
        
        Returns:
            (emotion_label, confidence): Tuple of predicted emotion and probability
        """
        smoothed = self.get_smoothed()
        idx = np.argmax(smoothed)
        return emotions_list[idx], smoothed[idx]
    
    def reset(self):
        """Reset the buffer (e.g., when tracking a new face)."""
        self.buffer.fill(0)
        self.position = 0
        self.count = 0


# =============================================================================
# Test Code - Run this file directly to test your implementation
# =============================================================================
if __name__ == '__main__':
    print("="*70)
    print("Testing Emotion Smoother")
    print("="*70)
    
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    
    try:
        # Test 1: Basic functionality
        print("\n📊 Test 1: Basic smoothing")
        smoother = EmotionSmoother(window_size=3, num_classes=7)
        
        # Simulate predictions over 5 frames
        predictions = [
            [0.1, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0],  # Happy
            [0.1, 0.0, 0.0, 0.7, 0.2, 0.0, 0.0],  # Happy
            [0.2, 0.0, 0.0, 0.3, 0.5, 0.0, 0.0],  # Neutral
            [0.1, 0.0, 0.0, 0.6, 0.3, 0.0, 0.0],  # Happy
            [0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],  # Happy
        ]
        
        for i, probs in enumerate(predictions):
            smoother.update(probs)
            emotion, conf = smoother.get_emotion(EMOTIONS)
            print(f"   Frame {i+1}: {emotion} ({conf:.1%})")
        
        print("   ✅ Smoothing reduces flickering between emotions")
        
        # Test 2: Empty buffer
        print("\n📊 Test 2: Empty buffer handling")
        smoother2 = EmotionSmoother(window_size=5, num_classes=7)
        empty_result = smoother2.get_smoothed()
        print(f"   Empty buffer returns: {empty_result}")
        print(f"   Sum: {np.sum(empty_result):.4f} (should be 1.0)")
        print(f"   All equal: {np.allclose(empty_result, empty_result[0])} (should be True)")
        print("   ✅ Empty buffer returns uniform distribution")
        
        # Test 3: Circular behavior
        print("\n📊 Test 3: Circular buffer behavior")
        smoother3 = EmotionSmoother(window_size=2, num_classes=7)
        
        # Add 3 entries to a size-2 buffer (should overwrite first)
        smoother3.update([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Anger
        smoother3.update([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Happy
        smoother3.update([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # Neutral (overwrites Anger)
        
        smoothed = smoother3.get_smoothed()
        emotion, _ = smoother3.get_emotion(EMOTIONS)
        # Should be average of Happy and Neutral, not Anger
        print(f"   After 3 updates to size-2 buffer:")
        print(f"   Happiness prob: {smoothed[3]:.2f}, Neutral prob: {smoothed[4]:.2f}")
        print(f"   Anger prob: {smoothed[0]:.2f} (should be 0 - was overwritten)")
        print("   ✅ Circular buffer correctly overwrites oldest entries")
        
        print("\n" + "="*70)
        print("✅ Semester 2, Phase 2 Complete!")
        print("="*70)
        print("\nNext: Update core/face_recognizer.py (Phase 3)")
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("\n💡 Implement the TODOs in this file")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
