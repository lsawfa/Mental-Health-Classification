## Overview
This document evaluates the performance of various machine learning (ML) and deep learning (DL) models across different datasets, using key metrics such as F1-Macro (F1-Mac), Precision-Macro (P-Mac), Recall-Macro (R-Mac), F1-Micro (F1-Mic), Precision-Micro (P-Mic), Recall-Micro (R-Mic), and their average.

| Method      | Type | F1-Mac | P-Mac | R-Mac | F1-Mic | P-Mic | R-Mic | Avg.   |
|-------------|------|--------|-------|-------|--------|-------|-------|--------|
| **Dataset 1**                                                                                   |
| SVM         | ML   | 0.46   | 0.71  | 0.42  | 0.62   | 0.75  | 0.52  | 0.58   |
| Light GBM   | ML   | 0.58   | 0.48  | 0.80  | 0.65   | 0.52  | 0.86  | 0.65   |
| XGBoost     | ML   | 0.57   | 0.62  | 0.54  | 0.65   | 0.69  | 0.62  | 0.62   |
| GAN-BERT    | DL   | 0.70   | 0.69  | 0.72  | 0.75   | 0.73  | 0.77  | 0.73   |
| BERT        | DL   | 0.74   | 0.72  | 0.77  | 0.79   | 0.76  | 0.83  | 0.77   |
| BART        | DL   | 0.76   | 0.70  | 0.81  | 0.80   | 0.74  | 0.86  | 0.78   |
| **Dataset 2**                                                                                   |
| SVM         | ML   | 0.04   | 0.04  | 0.04  | 0.07   | 0.13  | 0.04  | 0.06   |
| Light GBM   | ML   | 0.04   | 0.03  | 0.03  | 0.06   | 0.12  | 0.04  | 0.05   |
| XGBoost     | ML   | 0.06   | 0.06  | 0.05  | 0.07   | 0.10  | 0.05  | 0.07   |
| GAN-BERT    | DL   | 0.17   | 1.00  | 0.10  | 0.17   | 1.00  | 0.10  | 0.56   |
| BERT        | DL   | 0.80   | 0.81  | 0.80  | 0.79   | 0.79  | 0.79  | 0.79   |
| BART        | DL   | 0.32   | 0.34  | 0.46  | 0.39   | 0.35  | 0.45  | 0.39   |
| **Dataset 3**                                                                                   |
| SVM         | ML   | 0.12   | 0.25  | 0.11  | 0.16   | 0.26  | 0.12  | 0.17   |
| Light GBM   | ML   | 0.13   | 0.26  | 0.13  | 0.17   | 0.26  | 0.13  | 0.18   |
| XGBoost     | ML   | 0.12   | 0.25  | 0.11  | 0.16   | 0.26  | 0.12  | 0.17   |
| GAN-BERT    | DL   | 0.50   | 0.56  | 0.62  | 0.54   | 0.50  | 0.60  | 0.55   |
| BERT        | DL   | 0.90   | 0.91  | 0.90  | 0.90   | 0.90  | 0.90  | 0.90   |
| BART        | DL   | 0.85   | 0.87  | 0.86  | 0.88   | 0.85  | 0.87  | 0.86   |
