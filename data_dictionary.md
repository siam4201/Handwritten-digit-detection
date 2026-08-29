# Data Dictionary: Optical Recognition of Handwritten Digits

## Overview
- **Dataset Name**: Optical Recognition of Handwritten Digits (Digits / NIST)
- **Source**: UCI Machine Learning Repository / NIST
- **Authors**: E. Alpaydin, C. Kaynak (Bogazici University)
- **License**: Public Domain / Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Sample Size**: 1,797 samples
- **Feature Dimension**: 64 numeric pixel features (representing an 8x8 normalized grayscale grid)
- **Target Dimension**: 1 multi-class integer label (0 to 9)

## Variable Specifications

| Field Name | Data Type | Range / Domain | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `pixel_0` to `pixel_63` | Integer / Float | 0 to 16 | Normalized Pixel Count | Grayscale pixel intensity count within an 8x8 normalized matrix (0 = white background, 16 = maximum black stroke). |
| `target` | Integer | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | Class Label | The true handwritten digit class represented in the image. |

## Class Distribution

| Class Label | Character Represented | Sample Count | Percentage |
| :---: | :---: | :---: | :---: |
| 0 | '0' | 178 | 9.91% |
| 1 | '1' | 182 | 10.13% |
| 2 | '2' | 177 | 9.85% |
| 3 | '3' | 183 | 10.18% |
| 4 | '4' | 181 | 10.07% |
| 5 | '5' | 182 | 10.13% |
| 6 | '6' | 181 | 10.07% |
| 7 | '7' | 179 | 9.96% |
| 8 | '8' | 174 | 9.68% |
| 9 | '9' | 180 | 10.02% |
| **Total** | | **1,797** | **100.0%** |

## Ethical, Privacy, and License Statement
- **Anonymity**: All samples consist strictly of pixel matrices generated from isolated handwritten characters collected from consenting volunteer forms. No personally identifiable information (PII), geographic tags, or personal metadata are attached.
- **Fair Use**: Open access academic benchmark distributed under standard open licensing terms.
