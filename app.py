from flask import Flask, request, jsonify
import pickle

# 🔹 Create Flask app
app = Flask(__name__)

# 🔹 Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 🔹 Location encoding
def encode_location(loc):
    if loc == "home":
        return 0
    elif loc == "public":
        return 1
    elif loc == "market":
        return 2
    return 0

# 🔹 Predict API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        capacity = float(data.get("capacity", 0))
        avgWaste = float(data.get("avgWaste", 0))
        hours = float(data.get("hours", 0))
        location = data.get("location", "home")

        loc_encoded = encode_location(location)

        # 🔥 Model predicts WASTE (not %)
        predicted_waste = model.predict([[capacity, avgWaste, hours, loc_encoded]])[0]

        # 🔥 Convert to percentage
        if capacity > 0:
            predicted_fill = (predicted_waste / capacity) * 100
        else:
            predicted_fill = 0

        # 🔥 Clamp value between 0–100
        predicted_fill = max(0, min(predicted_fill, 100))

        # 🔹 Priority logic
        if predicted_fill > 80:
            priority = "OVERFLOW RISK"
        elif predicted_fill > 60:
            priority = "HIGH"
        elif predicted_fill > 30:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        return jsonify({
            "predicted_fill": round(predicted_fill, 2),
            "priority": priority
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# 🔹 Run server
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)