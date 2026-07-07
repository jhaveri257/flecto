# Federated Learning Evaluation Summary

| Mode | Final Accuracy | Comm Cost (MB) | Compression Ratio | Bytes Saved (MB) | Comm Reduction (%) | Total Time (s) | Avg Recon Error |
|---|---|---|---|---|---|---|---|
| Full Update | 0.0800 | 250.3677 | 1.0000 | 0.0000 | 0.0000 | 5.3978 | 0.0000 |
| Top-K (10%) | 0.0800 | 25.0381 | 0.1000 | 225.3296 | 89.9995 | 4.9489 | 0.6792 |
| Top-K + Error Feedback | 0.0850 | 25.0381 | 0.1000 | 225.3296 | 89.9995 | 5.0368 | 0.8160 |
