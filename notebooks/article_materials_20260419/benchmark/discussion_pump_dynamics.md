## Discussion: Hybrid Pump Dynamics

The grouped evaluation distinguishes the four passive / gravitational sensors from the pump-controlled sensor `G80F13P_LevelPS`. This decomposition shows that the aggregate error is not caused by uniformly weak performance across the network. Instead, the best non-pump mean reaches `0.9463` NSE (`dual-branch LNN`), whereas the best pump-only score reaches only `0.4594` NSE (`dual-branch LNN`). The resulting gap of `0.4870` NSE indicates that the pump sensor remains the principal source of residual error.

The best aggregate model is `dual-branch LNN` with `NSE=0.8489` across all sensors. Relative to the stable reference, the best pump-focused variant improves `G80F13P_LevelPS` by `0.4459` NSE.

This pattern is consistent with a hybrid-dynamics interpretation of the pump station. The passive sensors are dominated by smoother storage and conveyance processes, which are well aligned with the continuous-time inductive bias of the liquid architecture. By contrast, the pump sensor is affected by thresholded on/off control and hysteretic switching, which introduce discrete state transitions on top of the underlying continuous hydraulic response. Explicit pump-state features therefore improve the model, but they do not remove the performance gap entirely.

 For `G80F13P_LevelPS`, the best recorded configuration reaches `NSE=0.4594`, `RMSE=0.1699`, `MAE=0.1251`, and `Volume_Error=2.0800`.

For manuscript reporting, the main benchmark should therefore separate `all_sensors`, `non_pump_mean`, and `pump_only`, rather than relying solely on a single aggregate score. This framing makes clear that the liquid model is highly competitive on the passive sensors, while the remaining error is concentrated in the pump-controlled regime. If one additional architectural experiment is pursued, the most defensible next step is a dual-branch pump encoder that injects the pump-control signal into a dedicated temporal sub-encoder, rather than another round of broad loss tuning or additional global feature engineering.
