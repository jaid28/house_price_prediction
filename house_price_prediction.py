sizes = [1000, 1500, 2000, 2500, 3000]
prices = [300, 400, 500, 600, 700]

m = 0.1
b = 100
learning_rate = 0.0000001
epochs = 10

for _ in range(epochs):
    m_gradient = 0
    b_gradient = 0
    n = len(sizes)
    
    # Compute gradients
    for i in range(n):
        x = sizes[i]
        y = prices[i]
        prediction = m * x + b
        m_gradient += -2 * x * (y - prediction)
        b_gradient += -2 * (y - prediction)
    
    m_gradient /= n
    b_gradient /= n
    
    # Update parameters
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient
    
    # Optional: Print progress
    mse = sum((prices[i] - (m * sizes[i] + b))**2 for i in range(n)) / n
    print(f"Epoch {_+1}: m={m:.4f}, b={b:.4f}, MSE={mse:.2f}")

# Test a prediction
test_size = 2200
predicted_price = m * test_size + b
print(f"Predicted price for {test_size} sq ft: ${predicted_price:.2f}k")
