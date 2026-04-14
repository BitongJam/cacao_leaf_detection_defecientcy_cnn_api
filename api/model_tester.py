# save as test_extract.py
import tensorflow as tf
import numpy as np

# Load your model
MODEL = tf.keras.models.load_model("./models/1.keras")

def get_actual_model(model):
    """Extract actual model from sequential wrapper"""
    if isinstance(model.layers[0], tf.keras.Sequential):
        print("✅ Found Sequential wrapper! Extracting inner model...")
        return model.layers[0]
    else:
        print("ℹ️ No wrapper found, using model as is")
        return model

def test_extraction():
    print("=" * 50)
    print("TESTING MODEL EXTRACTION")
    print("=" * 50)
    
    # 1. Show original model structure
    print("\n1. ORIGINAL MODEL:")
    print(f"   Type: {type(MODEL)}")
    print(f"   Number of layers: {len(MODEL.layers)}")
    print(f"   First layer type: {type(MODEL.layers[0])}")
    print(f"   First layer name: {MODEL.layers[0].name}")
    
    # 2. Extract actual model
    print("\n2. EXTRACTING ACTUAL MODEL:")
    actual_model = get_actual_model(MODEL)
    print(f"   Type: {type(actual_model)}")
    print(f"   Number of layers: {len(actual_model.layers)}")
    
    # 3. Show all layers in actual model
    print("\n3. LAYERS IN ACTUAL MODEL:")
    for i, layer in enumerate(actual_model.layers):
        is_conv = isinstance(layer, tf.keras.layers.Conv2D)
        conv_mark = "✅ CONV" if is_conv else "   "
        print(f"   {i:2d}: {layer.name:20s} - {type(layer).__name__:15s} {conv_mark}")
    
    # 4. Find last conv layer
    print("\n4. FINDING LAST CONVOLUTIONAL LAYER:")
    last_conv = None
    for layer in reversed(actual_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    
    if last_conv:
        print(f"   ✅ Found: {last_conv}")
        
        # 5. Test getting layer output
        print("\n5. TESTING LAYER ACCESS:")
        try:
            conv_layer = actual_model.get_layer(last_conv)
            print(f"   ✅ Can access layer: {conv_layer.name}")
            print(f"   ✅ Layer type: {type(conv_layer)}")
            print(f"   ✅ Output shape: {conv_layer.output.shape}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   ❌ No Conv2D layer found!")
    
    # 6. Test prediction with extracted model
    print("\n6. TESTING PREDICTION:")
    try:
        dummy_input = np.random.random((1, 256, 256, 3))
        predictions = actual_model.predict(dummy_input, verbose=0)
        print(f"   ✅ Prediction successful!")
        print(f"   ✅ Output shape: {predictions.shape}")
        print(f"   ✅ Sample output: {predictions[0]}")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    test_extraction()