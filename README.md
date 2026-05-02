# ☕ Coffee Shop Revenue Predictor - Basic Neural Network
A neural network that predicts daily revenue for a coffee shop based on weather, day of the week, nearby events, and social media activity.

Built entirely in Python and NumPy.

---

## Files

| File | What it does |
|------|--------------|
| `run.py` | **Start here.** Generates data, trains the network, shows results |
| `neural_net.py` | The actual neural network (forward pass, backpropagation, training) |
| `data.py` | Generates 365 days of café data and normalizes it |

---

## How to run it

**Step 1 - Install the two libraries needed:**
```
pip install numpy pandas
```

**Step 2 - Run it:**
```
python run.py
```

That's it. It will train, test, and then ask you to enter a day's details so it can predict revenue.

---

## What the network actually does

The network has 3 layers:

```
Inputs (5)  →  Hidden Layer (8 neurons)  →  Output (1)
day, temp,       learns patterns               revenue
rain, event,     from the data                 in PKR
social buzz
```

You give it info about a day. It multiplies by weights, applies a function called ReLU, and spits out a revenue prediction. During training it does this 1000 times, checks how wrong it was each time, and nudges the weights to be less wrong next time. That's the whole thing.

---

## Features used to predict revenue

- **Day of week** - weekends earn more
- **Temperature** - cooler days bring more café visits
- **Is it raining** - rain reduces foot traffic
- **Nearby event** - concerts, cricket matches, expos drive revenue
- **Social media buzz** - Instagram mentions on that day

---

## Results

After training on 292 days, the network predicts revenue on 73 unseen days with an average error of roughly PKR 2,500–4,000 per day.

---

## What you can change and experiment with

In `neural_net.py`:
- Change `8` in the hidden layer to a bigger number (like `16`) - more neurons = more pattern-finding ability
- Change `learning_rate=0.01` in `run.py` to `0.001` (slower but smoother) or `0.1` (faster but messier)
- Change `epochs=1000` to `2000` - more training = lower loss (usually)

In `data.py`:
- Change the revenue formula to see how the network adapts
- Add a new feature like `is_holiday` and add it to the inputs in `neural_net.py` too

---

This project is a simplified version of real forecasting tools used at companies like Starbucks, Foodpanda, and any retailer with historical sales data.
