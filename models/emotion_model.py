"""
Emotion Recognition Model - Semester 2, Phase 1

Wraps a MobileNet ONNX model trained on AffectNet for emotion classification.
This runs IN PARALLEL with the face recognition system from Semester 1.

Model: MobileNet (7-class AffectNet)
Input: 224x224 RGB normalized image
Output: 7 emotion probabilities

Time: 1-2 hours
TODOs: 4
"""

import numpy as np
import cv2
import onnxruntime as ort


# ============================================================================
# CONSTANTS - DO NOT MODIFY
# ============================================================================
# These are hardcoded to match the specific model we're using.
# If you change the model, you must update these values.

IMG_SIZE = 224  # MobileNet uses standard 224x224 input

NUM_CLASSES = 7  # STRICT - we only support 7-class models (no Contempt)

# Emotion labels in the EXACT order the model outputs them
# This is the 7-class AffectNet order (Contempt removed)
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# ImageNet normalization constants
# These are the mean and std of the ImageNet dataset used to train MobileNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmotionModel:
    """
    Loads a MobileNet ONNX model and classifies facial emotions.
    
    This is similar to FaceEmbeddingModel from Semester 1, but instead of
    producing embeddings for verification, it produces class probabilities
    for classification.
    
    Key Difference from Semester 1:
    - Semester 1: Embeddings + Cosine Similarity (open-set verification)
    - Semester 2: Logits + Softmax (closed-set classification)
    """
    
    def __init__(self, model_path='assets/mobilenet_7.onnx'):
        """
        Initialize the emotion recognition model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        
        print(f"Loading emotion model from {model_path}...")
        
        # TODO 14: Load ONNX model and validate output shape
        # ===================================================
        # Steps:
        # 1. Create an ONNX Runtime InferenceSession:
        #    - Use ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        #    - Store as self.session
        #
        # 2. Get model metadata:
        #    - Get input name: self.session.get_inputs()[0].name
        #    - Store as self.input_name
        #
        # 3. Validate output shape (FAIL FAST):
        #    - Get output shape: self.session.get_outputs()[0].shape
        #    - Check if the last dimension equals NUM_CLASSES (7)
        #    - If not, raise ValueError with helpful message:
        #      f"Wrong model! Expected 7-class model, got {output_shape[-1]} classes. "
        #      f"Download the correct model: mobilenet_7.onnx"
        #
        # 4. Print success message with model info
        #
        # Hints:
        # - providers=['CPUExecutionProvider'] ensures CPU execution
        # - Output shape might be [None, 7] or [1, 7] - check the last dimension
        # - This validation prevents silent failures from wrong model downloads
        #
        # Why validate? Different emotion models have different class counts:
        # - 8-class includes Contempt
        # - 7-class excludes Contempt (our target)
        # Using the wrong model would cause index mismatches and wrong labels!
        
        raise NotImplementedError("TODO 14: Load ONNX model and validate output shape")
    
    def preprocess(self, face_img):
        """
        Preprocess face image for MobileNet inference.
        
        Args:
            face_img: Face image in BGR format (OpenCV default)
        
        Returns:
            Preprocessed image tensor ready for ONNX inference
            Shape: [1, 3, 224, 224], dtype: float32
        """
        
        # TODO 15: Implement preprocessing with ImageNet normalization
        # =============================================================
        # Steps:
        # 1. Resize image to IMG_SIZE x IMG_SIZE (224x224):
        #    - Use cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        #
        # 2. Convert BGR to RGB:
        #    - OpenCV loads images in BGR, but MobileNet expects RGB
        #    - Use cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # 3. Scale pixel values to [0, 1]:
        #    - Divide by 255.0
        #    - Make sure to use float division!
        #
        # 4. Apply ImageNet normalization:
        #    - Subtract MEAN: (img - MEAN)
        #    - Divide by STD: (img - MEAN) / STD
        #    - This centers the data around 0 with unit variance
        #
        # 5. Transpose from HWC to CHW format:
        #    - OpenCV uses (Height, Width, Channels)
        #    - ONNX expects (Channels, Height, Width)
        #    - Use img.transpose(2, 0, 1)
        #
        # 6. Add batch dimension:
        #    - ONNX expects shape [batch, channels, height, width]
        #    - Use img[np.newaxis, ...] to add batch dim
        #
        # 7. Ensure dtype is float32:
        #    - Use .astype(np.float32)
        #
        # Return the preprocessed tensor.
        #
        # Educational Note - Why ImageNet Normalization?
        # ==============================================
        # In Semester 1, InsightFace handled normalization internally.
        # Now YOU must do it explicitly. Here's why:
        #
        # MobileNet was trained on ImageNet, where images were normalized
        # using ImageNet's mean and std. To get good predictions, we must
        # normalize our input the same way the training data was normalized.
        #
        # Without normalization: pixel values range 0-255
        # With normalization: values centered around 0, roughly in [-2, 2]
        #
        # This is like converting temperatures from Fahrenheit to Celsius
        # before using a Celsius-trained weather model!
        
        raise NotImplementedError("TODO 15: Implement preprocessing")
    
    def softmax(self, logits):
        """
        Apply softmax to convert logits to probabilities.
        
        Args:
            logits: Raw model outputs (unnormalized scores)
        
        Returns:
            probabilities: Normalized probabilities that sum to 1
        """
        
        # TODO 16: Implement Softmax function manually
        # =============================================
        # The softmax function converts raw scores (logits) into probabilities.
        #
        # Mathematical Definition:
        #   softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
        #
        # Steps:
        # 1. Subtract the max value for numerical stability:
        #    - exp(large numbers) can overflow to infinity
        #    - Subtracting max doesn't change the result but prevents overflow
        #    - shifted = logits - np.max(logits)
        #
        # 2. Compute exponentials:
        #    - exp_values = np.exp(shifted)
        #
        # 3. Normalize by sum:
        #    - probabilities = exp_values / np.sum(exp_values)
        #
        # 4. Return the probabilities
        #
        # Educational Note - Why Manual Softmax?
        # ======================================
        # You COULD use scipy.special.softmax, but implementing it yourself
        # helps you understand what classification models actually output.
        #
        # Contrast with Semester 1:
        # - Cosine similarity: measures DISTANCE between embeddings
        # - Softmax: converts scores to PROBABILITIES over classes
        #
        # Cosine tells you "how similar", Softmax tells you "how likely"
        #
        # Example:
        #   logits = [2.0, 1.0, 0.1]  # Raw scores for 3 classes
        #   softmax(logits) = [0.659, 0.242, 0.099]  # Probabilities
        #   Sum = 1.0 (always!)
        
        raise NotImplementedError("TODO 16: Implement softmax")
    
    def predict(self, face_img):
        """
        Predict emotion from face image.
        
        Args:
            face_img: Face image in BGR format (OpenCV default)
        
        Returns:
            (emotion_label, confidence): Tuple of predicted emotion and probability
            Returns ("Unknown", 0.0) if prediction fails
        """
        
        # TODO 17: Run inference and return prediction
        # =============================================
        # Steps:
        # 1. Preprocess the image:
        #    - Call self.preprocess(face_img)
        #    - Store as input_tensor
        #
        # 2. Run ONNX inference:
        #    - Call self.session.run(None, {self.input_name: input_tensor})
        #    - This returns a list; get the first element [0]
        #    - Get the first batch item [0] to get shape [7]
        #    - These are the raw logits (unnormalized scores)
        #
        # 3. Apply softmax to get probabilities:
        #    - Call self.softmax(logits)
        #
        # 4. Find the predicted class:
        #    - Use np.argmax(probabilities) to get index of highest prob
        #    - Get emotion label: EMOTIONS[predicted_index]
        #    - Get confidence: probabilities[predicted_index]
        #
        # 5. Return (emotion_label, confidence)
        #
        # Error Handling:
        # - Wrap in try/except
        # - On any error, print warning and return ("Unknown", 0.0)
        #
        # Hints:
        # - session.run() returns list of outputs; we only have one output
        # - np.argmax returns the index of the maximum value
        # - EMOTIONS is the list mapping index to emotion name
        
        raise NotImplementedError("TODO 17: Run inference and return prediction")


# =============================================================================
# Test Code - Run this file directly to test your implementation
# =============================================================================
if __name__ == '__main__':
    import os
    
    print("="*70)
    print("Testing Emotion Recognition Model")
    print("="*70)
    
    model_path = 'assets/mobilenet_7.onnx'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found: {model_path}")
        print("\n📥 Download the model:")
        print('   curl -L -o assets/mobilenet_7.onnx \\')
        print('     "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/mobilenet_7.onnx?raw=true"')
        exit(1)
    
    try:
        # Test model loading
        model = EmotionModel(model_path=model_path)
        print("✅ Model loaded successfully\n")
        
        # Test with random image (won't give meaningful results, just tests the pipeline)
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emotion, confidence = model.predict(test_img)
        
        print(f"✅ Prediction works!")
        print(f"   Test prediction: {emotion} ({confidence:.1%})")
        print(f"   (Random image - result not meaningful)\n")
        
        # Test softmax properties
        logits = np.array([2.0, 1.0, 0.5, 0.1, 0.0, -0.5, -1.0])
        probs = model.softmax(logits)
        print(f"✅ Softmax test:")
        print(f"   Sum of probabilities: {np.sum(probs):.6f} (should be 1.0)")
        print(f"   All positive: {np.all(probs > 0)} (should be True)\n")
        
        print("="*70)
        print("✅ Semester 2, Phase 1 Complete!")
        print("="*70)
        print("\nNext: utils/emotion_smoother.py (Phase 2)")
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("\n💡 Implement the TODOs in this file")
        print("   Follow the step-by-step instructions in each TODO")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
