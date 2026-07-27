📐 STEP 1: The Forward Pass (Calculate Everything)
Input: x = 3
Weight1: w1 = 2
Weight2: w2 = 4

Forward Pass:
h1 = x × w1           (First multiplication)
h2 = ReLU(h1)         (Activation)
y_pred = h2 × w2      (Second multiplication)
Loss = (y_pred - 5)²  (Compare to target = 5)

Sub-step 1.1: Calculate h1 (First hidden layer)
h1 = x × w1
h1 = 3 × 2
h1 = 6

Sub-step 1.2: Calculate h2 (ReLU activation)
ReLU(6) = max(0, 6) = 6
h2 = 6

Sub-step 1.3: Calculate y_pred (Final prediction)
y_pred = h2 × w2
y_pred = 6 × 4
y_pred = 24

Sub-step 1.4: Calculate Loss                             - h1 = 6
Loss = (y_pred - target)²                                - h2 = 6
Loss = (24 - 5)²                                         - y_pred = 24
Loss = (19)²                                             - Loss = 361
Loss = 361


🔄 STEP 2: The Backward Pass (Gradient Flow)
Sub-step 2.1: Start with ∂L/∂y_pred
Loss = (y_pred - 5)²
∂L/∂y_pred = 2 × (y_pred - 5)
∂L/∂y_pred = 2 × (24 - 5)
∂L/∂y_pred = 2 × 19
∂L/∂y_pred = 38

Sub-step 2.2: Calculate ∂L/∂h2
y_pred = h2 × w2
∂y_pred/∂h2 = w2 (because derivative of h2×4 with respect to h2 is 4)
∂y_pred/∂h2 = 4

Now use chain rule:
∂L/∂h2 = ∂L/∂y_pred × ∂y_pred/∂h2
∂L/∂h2 = 38 × 4
∂L/∂h2 = 152

Sub-step 2.3: Calculate ∂L/∂w2 (THIS IS ONE OF OUR ANSWERS!)
∂y_pred/∂w2 = h2 (because derivative of 6×w2 with respect to w2 is 6)
∂y_pred/∂w2 = 6

∂L/∂w2 = ∂L/∂y_pred × ∂y_pred/∂w2
∂L/∂w2 = 38 × 6
∂L/∂w2 = 228

Sub-step 2.4: Calculate ∂L/∂h1 (Through ReLU)
h2 = ReLU(h1) = max(0, h1)
For h1 = 6 (which is > 0):
∂h2/∂h1 = 1

If h1 had been negative, ∂h2/∂h1 would be 0. But here it's 1.
∂L/∂h1 = ∂L/∂h2 × ∂h2/∂h1
∂L/∂h1 = 152 × 1
∂L/∂h1 = 152

Sub-step 2.5: Calculate ∂L/∂w1 (OUR SECOND ANSWER!)
We ask: "How does changing w1 affect Loss?"
h1 = x × w1
∂h1/∂w1 = x (because derivative of 3×w1 with respect to w1 is 3)
∂h1/∂w1 = 3

∂L/∂w1 = ∂L/∂h1 × ∂h1/∂w1
∂L/∂w1 = 152 × 3
∂L/∂w1 = 456