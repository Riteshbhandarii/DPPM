# Random Forest Feature Pruning Summary

Date: 2026-04-22

Baseline validation MAE: 18.2409
Original feature count: 66

Pruning results:
- Bottom 10% removed -> 59 features, validation MAE = 18.8389, delta = +0.5980
- Bottom 20% removed -> 53 features, validation MAE = 18.6868, delta = +0.4459
- Bottom 30% removed -> 46 features, validation MAE = 18.6639, delta = +0.4230

Conclusion:
Removing the lowest-ranked features did not improve validation performance.
The original 66-feature set was retained.
