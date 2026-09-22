# lpi-radar-snr-glance-and-focus
Official code for "Robust LPI Radar Signal Recognition via SNR-Guided Generative Restoration and Active Recurrent Glance-and-Focus" (IET Radar, Sonar &amp; Navigation, 2026, DOI: 10.1049/rsn2.70169)
        
        
# Robust LPI Radar Signal Recognition via SNR-Guided Generative Restoration and Active Recurrent Glance-and-Focus

[![Paper]([https://img.shields.io/badge/Paper-IET%20RSN%202026-blue](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.70169))](https://doi.org/10.1049/rsn2.70169
        
        )
[![License: MIT]([https://img.shields.io/badge/License-MIT-yellow.svg](https://github.com/Chengyu0918/lpi-radar-snr-glance-and-focus/blob/main/LICENSE))](LICENSE)

This repository is the official implementation of the paper:

> **Robust LPI Radar Signal Recognition via SNR-Guided Generative Restoration and Active Recurrent Glance-and-Focus**
> Yu Cheng, Jiantao Wang, Jie Huang, Yiming Li, Dexiu Hu
> *IET Radar, Sonar & Navigation*, 2026.
> DOI: [10.1049/rsn2.70169
        
        ](https://doi.org/10.1049/rsn2.70169
        
        )

## Overview
- **SNR-guided generative restoration**: restore low-SNR LPI radar spectrograms before recognition.
- **Active recurrent glance-and-focus network**: progressively focuses on informative time-frequency regions.
- Achieves state-of-the-art recognition accuracy under strong noise (e.g. −XX dB).

## Environment
- Python 3.10
- PyTorch 2.6
- CUDA 12.0

Install dependencies:
```bash
pip install -r requirements.txt
