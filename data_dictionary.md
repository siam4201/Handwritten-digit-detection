# Data Dictionary: Optical Recognition of Handwritten Digits

## 1. Dataset Provenance & Overview
- **Dataset Name**: Optical Recognition of Handwritten Digits
- **Original Source**: UCI Machine Learning Repository (DOI: 10.24432/C50P49) / National Institute of Standards and Technology (NIST)
- **Creators / Authors**: E. Alpaydin and C. Kaynak, Department of Computer Engineering, Bogazici University, 80815 Istanbul, Turkey
- **License / Terms**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Sample Size**: 1,797 samples
- **Total Variables**: 65 (64 continuous pixel intensity features + 1 discrete target class)

---

## 2. Feature & Target Specifications

| Variable / Field Name | Data Type | Domain / Range | Units | Derived / Scaled Representation | Description & Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pixel_0` to `pixel_63` | Integer | 0 to 16 | Normalized Pixel Count | Scaled to $[0.0, 1.0]$ via runtime `MinMaxScaler` | Grayscale pixel intensity count within an 8x8 normalized matrix (0 = white background, 16 = maximum black stroke). Each feature represents the count of ON pixels in a 4x4 sub-block extracted from original 32x32 binary bitmaps. Feature indexing follows row-major order: `pixel_i` corresponds to matrix position $(r, c)$ where $r = \lfloor i / 8 \rfloor$ and $c = i \pmod 8$. |
| `target` | Integer | 0 to 9 | Discrete Class Label | N/A (Ground Truth) | The true handwritten numerical digit class represented in the image (0, 1, 2, 3, 4, 5, 6, 7, 8, 9). |

---

## 3. Spatial Feature Indexing Matrix (8x8 Normalized Grid)

Each image instance is structured as an 8x8 matrix where individual elements map to the 64 tabular columns:

| | Col 0 | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Row 0** | `pixel_0` | `pixel_1` | `pixel_2` | `pixel_3` | `pixel_4` | `pixel_5` | `pixel_6` | `pixel_7` |
| **Row 1** | `pixel_8` | `pixel_9` | `pixel_10` | `pixel_11` | `pixel_12` | `pixel_13` | `pixel_14` | `pixel_15` |
| **Row 2** | `pixel_16` | `pixel_17` | `pixel_18` | `pixel_19` | `pixel_20` | `pixel_21` | `pixel_22` | `pixel_23` |
| **Row 3** | `pixel_24` | `pixel_25` | `pixel_26` | `pixel_27` | `pixel_28` | `pixel_29` | `pixel_30` | `pixel_31` |
| **Row 4** | `pixel_32` | `pixel_33` | `pixel_34` | `pixel_35` | `pixel_36` | `pixel_37` | `pixel_38` | `pixel_39` |
| **Row 5** | `pixel_40` | `pixel_41` | `pixel_42` | `pixel_43` | `pixel_44` | `pixel_45` | `pixel_46` | `pixel_47` |
| **Row 6** | `pixel_48` | `pixel_49` | `pixel_50` | `pixel_51` | `pixel_52` | `pixel_53` | `pixel_54` | `pixel_55` |
| **Row 7** | `pixel_56` | `pixel_57` | `pixel_58` | `pixel_59` | `pixel_60` | `pixel_61` | `pixel_62` | `pixel_63` |

---

## 4. Target Class Distribution

| Class Label | Digit Character | Sample Count | Percentage | Train Count (80%) | Test Count (20%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | '0' | 178 | 9.91% | 142 | 36 |
| 1 | '1' | 182 | 10.13% | 146 | 36 |
| 2 | '2' | 177 | 9.85% | 142 | 35 |
| 3 | '3' | 183 | 10.18% | 146 | 37 |
| 4 | '4' | 181 | 10.07% | 145 | 36 |
| 5 | '5' | 182 | 10.13% | 145 | 37 |
| 6 | '6' | 181 | 10.07% | 145 | 36 |
| 7 | '7' | 179 | 9.96% | 143 | 36 |
| 8 | '8' | 174 | 9.68% | 139 | 35 |
| 9 | '9' | 180 | 10.02% | 144 | 36 |
| **Total** | | **1,797** | **100.0%** | **1,437** | **360** |

---

## 5. Raw vs. Processed Data Lifecycle
- **Raw Data State**: The original dataset extracted from the repository is saved to `data/raw_digits.csv` in its pristine, unscaled integer form [0 to 16]. It is never manually altered.
- **Processed Data State**: Preprocessing transformations (scaling pixel values to the range [0.0, 1.0] using `MinMaxScaler`) are applied strictly at runtime inside scikit-learn `Pipeline` objects fitted solely on training splits, guaranteeing complete leakage control.

---

## 6. Sampling Limitations & Demographic Representation
- **Writer Demographics**: The dataset was constructed from 44 writers at Bogazici University. It contains low-resolution, isolated handwritten digits and may not fully represent handwriting styles, writing instruments (fountain pens, markers, ballpoints, styluses), image quality, or demographic variation encountered in real-world OCR systems. Consequently, performance may decrease under distribution shift, particularly for handwriting that differs substantially from the training distribution.
- **Style Variations**: Regional writing conventions (such as European crossed '7', crossed '0', or open vs. closed '4') are not systematically sampled.

---

## 7. Ethics, Privacy, and Responsible Use Statement
- **Privacy**: The dataset consists exclusively of segmented numerical handwriting strokes collected from anonymized forms. It contains no personally identifiable information (PII), geographic identifiers, or demographic metadata.
- **Licensing & Attribution**: Distributed under Creative Commons Attribution 4.0 International (CC BY 4.0).
