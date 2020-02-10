# Zindi_Flood_Prediction_Malawi
https://zindi.africa/competitions/2030-vision-flood-prediction-in-malawi/

#### Changelog:



| ID   | Version                                 | Comments                                                     | Zindi score | Issues/Todo's                                                |
| ---- | --------------------------------------- | ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| id_1 | ML-Baseline                             | Baseline                                                     | 0.15528     | Some negative scores (shouldn't be there)<br />Nothing done with coordinates (important?)<br />Feature creation with the evolution of rain<br />Standardization of rain volume for both periods? |
| id_2 | ML-Feature Creation Class               | Created an additional column by applying classification algorithm (flooded/not flooded) | 0.14838     | When doing the classification, there is a problem with imbalanced classes.<br />Adding the results of the classification as a feature might not be enough for the algorithm to recognize it as a strong feature. |
| id_3 | ML-Feature Creation Total precipitation | Created an additional column as total precipitations         | 0.14306     | When doing the classification, there is a problem with imbalanced classes.<br />Adding the results of the classification as a feature might not be enough for the algorithm to recognize it as a strong feature. |
| id_4 | DL-Baseline                             | Baseline                                                     | 0.28376     |                                                              |
| id_5 | DL-Tuned batch_size and epochs          | Baseline with batch_size=512, 100 epochs                     | 0.14689     | Modify layers.<br />Change inputs                            |
