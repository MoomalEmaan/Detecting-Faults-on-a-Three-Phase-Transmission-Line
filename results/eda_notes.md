# Task 2 - Exploratory plots

All three plots use the raw (unstandardised) data from `data/clean_dataset.csv`.

## 2.1 Currents against row number (first 2000 rows)

![currents vs time](figures/2_1_currents_vs_time.png)

Before the fault (rows 0 to about 505) the three currents are normal sine waves of about ±60 A, shifted from each other like a healthy three-phase system should be. When the fault starts (grey region, rows ~505 to ~1630) Ia jumps to about ±880 A, roughly 14 times its normal size, with some fast ringing at the very start, while Ib and Ic stay small, so this fault mainly involves phase A. Once the fault is cleared the currents go back to their normal small sine waves.

## 2.2 Ia against Ib, coloured by fault

![scatter Ia Ib](figures/2_2_scatter_Ia_Ib.png)

The no-fault points (blue) sit in a small blob around the origin, because a healthy current never goes far from zero. The fault points (orange) form big rings and ellipses that go all the way around that blob, on every side: large positive Ia, large negative Ia, large positive Ib and large negative Ib all mean fault. Some orange points also land inside the blue blob, which are faults on the other phases where Ia and Ib themselves stay normal.

## 2.3 Histogram of Ia for fault and no-fault rows

![histogram Ia](figures/2_3_hist_Ia.png)

No-fault rows are packed into a narrow range of about -170 A to +105 A (mean |Ia| ≈ 40 A). Fault rows are spread over the whole range from about -880 A to +885 A (mean |Ia| ≈ 460 A), with bumps at both ends where the large sine wave spends most of its time near its peaks. The fault histogram is almost symmetric around zero, so the average Ia is close to zero for both classes. The difference between them is how far from zero the current is, not which side of zero it is on.

## Could a single straight line in the (Ia, Ib) plane separate the two classes?

No. In the scatter plot the no-fault points sit in the middle and the fault points surround them on all sides, so any straight line that cuts off the blue blob from one side leaves fault points on the other side together with the normal ones. The reason is what a fault does to the current: it makes the current much **bigger in size**, but the current is still an AC sine wave, so at any instant it can be large and positive or large and negative. A fault therefore pushes the point away from the origin in either direction, and only a closed boundary (like a circle or ellipse around the normal blob) can separate the classes. A straight line can only say "Ia is bigger than something" or "smaller than something", not "Ia is far from zero". 
