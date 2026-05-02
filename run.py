# inspiration for making NN was sir kashif 
# next update i'll add real data and input validation
# icons taken from font awesome (please don't sue me)

from data import generate_coffee_data, prepare_data
from neural_net import CoffeeNeuralNet
import numpy as np

print("=" * 55)
print("   ☕ Coffee Shop Revenue Predictor")
print("=" * 55)

# generate data (365 days of a fictional café)
print("\n[1] Generating 365 days of coffee shop data...")
df = generate_coffee_data(n_days=365)
print(f"    Sample of the data:")
print(df.head(5).to_string(index=False))
print()

# prepare it for the network
print("[2] Normalizing features and splitting into train/test...")
X_train, X_test, y_train, y_test, scale = prepare_data(df)
print(f"    Training samples: {len(X_train)}")
print(f"    Test samples:     {len(X_test)}")
print()

# build and train the network
print("[3] Training the neural network...")

model = CoffeeNeuralNet()
losses = model.train(X_train, y_train, learning_rate=0.01, epochs=1000)

print(f"\n    Loss went from {losses[0]:.4f}  to  {losses[-1]:.4f}")
print("    (Lower = better, wah jee)")
print()

# test it on data it has never seen
print("[4] Testing on unseen data...")

predictions_norm = model.predict(X_test).flatten()

# undo the normalization to get actual PKR values
y_min, y_max = scale["y_min"], scale["y_max"]
predictions_pkr = predictions_norm * (y_max - y_min) + y_min
actuals_pkr     = y_test         * (y_max - y_min) + y_min

mae = np.mean(np.abs(predictions_pkr - actuals_pkr))
print(f"    Average prediction error: PKR {mae:,.0f} per day")
print()

# show some actual vs predicted
print("[5] Sample predictions (Actual vs Predicted):\n")
print(f"    {'Day':<6} {'Actual Revenue':>16} {'Predicted':>12} {'Difference':>12}")
print("    " + "-" * 50)

for i in range(10):
    actual = actuals_pkr[i]
    pred   = predictions_pkr[i]
    diff   = pred - actual
    sign   = "+" if diff >= 0 else ""
    print(f"    {i+1:<6} PKR {actual:>10,.0f}   PKR {pred:>9,.0f}   {sign}{diff:>+9,.0f}")

print()

# let the user make a custom prediction
print("=" * 55)
print("   🔮 Make Your Own Prediction")
print("=" * 55)
print()
print("   Enter details about a day and the network will")
print("   predict how much revenue the café will make.\n")

try:
    dow     = int(input("   Day of week? (0=Mon, 1=Tue, ..., 6=Sun): "))
    temp    = float(input("   Temperature in °C? (e.g. 28): "))
    rain    = int(input("   Is it raining? (1=yes, 0=no): "))
    event   = int(input("   Nearby event today? (1=yes, 0=no): "))
    buzz    = int(input("   Instagram mentions of the café today? (e.g. 20): "))

    # normalize the user input using the same min/max from training
    raw = np.array([[dow, temp, rain, event, buzz]], dtype=float)

    # hardcoded min/max matching what prepare_data does
    raw_norm = raw.copy()
    raw_norm[0, 0] = dow / 6.0          # day_of_week: 0-6
    raw_norm[0, 1] = (temp - 5) / 37    # temperature: 5-42
    raw_norm[0, 2] = rain               # already 0 or 1
    raw_norm[0, 3] = event              # already 0 or 1
    raw_norm[0, 4] = buzz / 49.0        # social_buzz: 0-49

    pred_norm = model.predict(raw_norm).flatten()[0]
    pred_pkr  = pred_norm * (y_max - y_min) + y_min

    print(f"\n   Predicted Revenue: PKR {pred_pkr:,.0f}")

    if pred_pkr > 30000:
        print("   Strong day - maybe stock up on supplies!")
    elif pred_pkr > 20000:
        print("   Decent day. Normal operations.")
    else:
        print("   Slow day expected. Maybe run a promo?")

except Exception:
    print("   (Skipped custom prediction)")

print()
print("   Done! The network is saved in neural_net.py")
print("   Check data.py to see how the training data was made.")
print()
