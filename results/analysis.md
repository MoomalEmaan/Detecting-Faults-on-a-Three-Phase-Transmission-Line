# Task 6 - Analysis

All numbers below come from our own runs (`results/tables/`). Validation numbers come from Tasks 3 and 4, and test numbers come from Task 5.

Quick reminder of the main results:

| Model | Validation accuracy | Test accuracy | Missed faults (test) | False alarms (test) |
|---|---|---|---|---|
| Logistic regression, Set R (6 raw features) | 73.1% | 72.6% | 657 | 0 |
| Logistic regression, Set Q (raw + squares) | 97.8% | 97.9% | 45 | 6 |
| Linear SVM (C = 0.1) | 72.6% | 73.6% | 634 | 0 |
| RBF SVM (C = 100, γ = 1) | 99.7% | 99.6% | 6 | 4 |

The test set has 2401 rows, of which 1096 are faults.

---

## 6.1 Why do logistic regression on Set R and the linear SVM score almost the same? What does that tell us about the data?

**Short answer:** Both models draw a straight line (a flat boundary), and no straight line can separate this data well. So they both hit the same limit of about 73%.

**In more detail:**

- Logistic regression predicts "fault" when θᵀx ≥ 0. The boundary θᵀx = 0 is a straight line in 2D, or a flat plane (a "hyperplane") in our 6 dimensions.
- A linear SVM predicts "fault" when wᵀx + b ≥ 0. That is also a flat hyperplane.
- The only difference is **how** each model picks its line:
  - logistic regression minimises the log-loss J(θ) (it wants good probabilities),
  - the SVM maximises the margin, i.e. the gap between the line and the nearest points.
- Even though they choose the line in different ways, they are both choosing from the **same family of shapes**: straight lines. If the best possible straight line only gets about 73%, then any linear model will end up near 73%. That is exactly what we see: 73.1% vs 72.6% on validation, and 72.6% vs 73.6% on test.

**What it tells us about the data:** the data is **not linearly separable**. We already saw why in Task 2. The no-fault points sit in a small blob around zero, and the fault points surround that blob on every side (big positive current *and* big negative current both mean fault). A straight line can only cut the plane into two halves, so it can never put "the middle" on one side and "all around the outside" on the other.

You can actually see this in the confusion matrices. Both linear models have **0 false alarms** but more than 600 missed faults. The line has been placed off to one side of the normal blob, so it only catches faults where the current is extreme in one direction. Every fault on the other side of zero is missed. Changing C or α does not fix this, because the problem is the *shape* of the boundary, not its exact position.

---

## 6.2 What sign do the I² weights and the V² weights have in Set Q? Why?

Our learned weights for the squared terms (Table 5):

| Weight | Value | Sign |
|---|---|---|
| Ia² | +18.89 | positive |
| Ib² | +4.75 | positive |
| Ic² | +13.96 | positive |
| Va² | −6.03 | negative |
| Vb² | −6.46 | negative |
| Vc² | −6.14 | negative |

**Short answer:** all current-squared weights are **positive** and all voltage-squared weights are **negative**. During a fault, current goes **up** and voltage goes **down**.

**Remember how the model decides:** it calculates z = θᵀx. A bigger z means a higher probability of fault (z ≥ 0 means "fault"). So:

- a **positive** weight means "the bigger this feature, the more likely it is a fault",
- a **negative** weight means "the bigger this feature, the less likely it is a fault".

**Why the I² weights are positive:**
When a phase touches ground or another phase, the impedance becomes very small, so a huge current flows (a short circuit). In our data the average size of the current goes from about 40 A to about 460 A for phase A (Ia: 40 → 461 A, Ib: 43 → 423 A, Ic: 43 → 311 A). So I² becomes very large during a fault. A large I² should push the model towards "fault", which means its weight must be **positive**.

The square matters because the current is AC. At any instant a fault current can be +800 A or −800 A. The raw value Ia cannot tell these two cases apart from "normal" with one straight line, but **Ia² is large in both cases**. Squaring turns "far from zero in either direction" into "big number", which is something a linear model can use. That is why Set Q jumps from 73% to 98%.

**Why the V² weights are negative:**
During a fault the voltage on the faulted phase **sags** (collapses towards zero), because the fault is pulling the line down to ground or to another phase. In our data the average V² falls from about 0.18 to about 0.04–0.06 during faults (Va²: 0.187 → 0.040). So a **small** V² means "fault" and a **large** (normal) V² means "no fault". Bigger V² should push the model *away* from fault, so the weight is **negative**.

Another way to see it: after standardising, a sagging V² is *below* the training average, so its standardised value is negative. A negative value times a negative weight gives a **positive** contribution to z, which pushes the prediction towards "fault". That is exactly what we want.

**Small extra point (in case it is asked):** Ib² has a smaller weight (4.75) than Ia² and Ic², even though Ib also rises a lot during faults. This does not mean Ib is unimportant. The features overlap: in many faults two or three phases rise together, so the model can "share" the credit between them. Individual weights in a model with correlated features should not be over-read. The **sign** is the reliable part.

---

## 6.3 Compare the number of support vectors of the best linear SVM and the best RBF SVM. What does a very large number mean?

| Model | Support vectors | Out of 7200 training rows |
|---|---|---|
| Best linear SVM (C = 0.1) | 5846 | 81% |
| Best RBF SVM (C = 100, γ = 1) | 278 | 3.9% |

**What a support vector is:** after training, an SVM only "remembers" some of the training points. These are the points that lie **on the margin, inside the margin, or on the wrong side of the boundary**. These are the support vectors. Every other point is safely on the correct side and has no effect on the boundary at all.

**What a very large number of support vectors means:**
The linear SVM needs 81% of all training points as support vectors. That means most of the points are either inside the margin or on the wrong side of the line. In other words, **the model cannot find a clean boundary**. It is "struggling" with almost every point. This is another sign that a straight line is the wrong shape for this data, and it agrees with the 73% accuracy.

The RBF SVM needs only 278 points (about 4%). Almost all points are clearly on the correct side, and only a few points near the boundary are needed to describe it. **A small number of support vectors means the classes are well separated by the model's boundary.**

Practical side effect: the number of support vectors also decides how much work prediction takes, because the SVM compares a new point with every support vector. So the RBF model is not only more accurate here, it is also about 20 times lighter than the linear one.

---

## 6.4 What happens to accuracy and to the number of support vectors as C increases? Why?

From Table 3:

| C | Linear: acc / SVs | RBF γ=0.1: acc / SVs | RBF γ=1: acc / SVs |
|---|---|---|---|
| 0.1 | 72.58% / 5846 | 95.00% / 1348 | 98.04% / 867 |
| 1 | 72.58% / 5838 | 97.50% / 723 | 99.17% / 501 |
| 10 | 72.58% / 5836 | 99.21% / 381 | 99.62% / 336 |
| 100 | 72.58% / 5837 | 99.54% / 174 | 99.67% / 278 |

**What we see:**
- **RBF:** as C goes up, **accuracy goes up** and the **number of support vectors goes down** (for γ = 0.1, 1348 → 174).
- **Linear:** C makes almost **no difference**. Accuracy stays at 72.58% and the support vectors stay around 5840.

**What C controls:**
C is the **penalty for breaking the margin**, i.e. how much the SVM "cares" about training points that are inside the margin or on the wrong side.

- **Small C** = "mistakes are cheap". The SVM prefers a wide, relaxed margin and allows many points to sit inside it. Lots of points end up inside the margin, so there are **lots of support vectors**. The boundary is smooth and simple, maybe too simple (underfitting).
- **Large C** = "mistakes are expensive". The SVM tries hard to put every training point on the correct side, so the margin becomes narrower and the boundary bends more to follow the data. Fewer points are left inside the margin, so there are **fewer support vectors**, and the training (and here also validation) accuracy goes up.

**Why the linear SVM doesn't change:** no matter how hard we punish mistakes, a straight line still cannot wrap around the normal blob. Raising C just moves the line a tiny bit. The limit comes from the shape, not from the penalty.

**Warning about very large C:** if C is too large, the model can start fitting noise in the training data (overfitting), and validation accuracy would start to drop. In our range (up to C = 100) we did not reach that point, because validation accuracy kept rising. This is exactly why C is chosen with the **validation** set and not the training set.

---

## 6.5 The split was random and neighbouring samples look almost identical. Is our test accuracy optimistic or pessimistic for a real relay?

**Short answer: optimistic.** The real relay would probably do worse than 99.6%.

**Why:**
- The rows are consecutive time samples from a simulation. Two neighbouring rows are almost the same. On average, one row differs from the next by only about **1% of a standard deviation** in each feature.
- We shuffled all rows randomly before splitting. So for most test rows, the row just before or just after it in time ended up in the **training** set. In our split, **84.5% of test rows have a direct neighbour (t−1 or t+1) in the training set**.
- That means the test set is not really "new" data. For most test points, the model has already seen an almost identical point during training. The test is more like checking whether it remembers what it has seen than whether it handles a situation it has never met.
- This helps the RBF SVM most of all, because the RBF kernel basically classifies a point by how close it is to training points it has seen. If a near-copy is in the training set, the answer is easy.

**What a real relay would face:**
- New faults that were never in the training data: different fault locations along the line, different fault resistance, different load levels, different moments in the AC cycle when the fault starts.
- Real-world measurement noise, sensor errors and harmonics. Our data is a clean MATLAB/Simulink simulation.

**How we could get an honest estimate:** split by **time blocks** or by **fault event** instead of by random rows. For example, train on the first faults and test on completely different, later faults, so no part of a test fault was ever seen in training. We expect the accuracy would drop, especially for the RBF SVM.

---

## 6.6 Which model would we deploy in the relay? (based on missed faults and false alarms)

**Test-set errors (2401 rows, 1096 actual faults):**

| Model | Missed faults | False alarms |
|---|---|---|
| Logistic regression, Set R | 657 (60% of faults) | 0 |
| Linear SVM | 634 (58% of faults) | 0 |
| Logistic regression, Set Q | 45 (4.1% of faults) | 6 |
| RBF SVM | **6 (0.5% of faults)** | **4** |

**The two kinds of error are not equally bad:**
- A **missed fault** means the breaker does not trip while a short circuit is happening. This can damage transformers and lines, cause fires, endanger people and make the whole grid unstable. **This is the dangerous error.**
- A **false alarm** means the breaker trips when there is no fault. Customers lose power for a while and it is annoying and costly, but nothing gets destroyed.

So we care **first about missed faults**, and only then about false alarms.

**Our choice: the RBF SVM (C = 100, γ = 1).**
- It misses only 6 of 1096 faults, the fewest by far.
- It also has the fewest false alarms (4). So we don't even have to trade one error for the other. It is the best on both.

**Why not the others:**
- The two linear models (Set R logistic and linear SVM) are ruled out immediately. Their "0 false alarms" looks nice, but it only happens because they rarely say "fault" at all, and they miss about 60% of real faults. A relay like that is dangerous.
- Logistic regression on Set Q is a good runner-up: 97.9% accuracy, simple, fast and easy to explain (we can read its weights, as in 6.2). But it misses 45 faults, which is 7.5 times more than the RBF SVM. If we needed a very simple model, we could lower its decision threshold below 0.5 (for example 0.3) to catch more faults at the cost of some extra false alarms. That threshold would have to be chosen on the validation set, not the test set.

**One honest caveat (links to 6.5):** these numbers are optimistic because of the random split. Before putting any model in a real relay, we would retest it on completely unseen fault events, and on noisy or real measurements, to make sure it still catches nearly every fault.