# Zindi_Flood_Prediction_Malawi

Quick ML and DL models applied to the flood prediction challenge by Zindi.

https://zindi.africa/competitions/2030-vision-flood-prediction-in-malawi/

#### Changelog:



| ID   | Version                                 | Comments                                                     | Zindi score | Issues/Todo's                                                |
| ---- | --------------------------------------- | ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| id_1 | ML-Baseline                             | Baseline                                                     | 0.15528     | Some negative scores (shouldn't be there)<br />Nothing done with coordinates (important?)<br />Feature creation with the evolution of rain<br />Standardization of rain volume for both periods? |
| id_2 | ML-Feature Creation Class               | Created an additional column by applying classification algorithm (flooded/not flooded) | 0.14838     | When doing the classification, there is a problem with imbalanced classes.<br />Adding the results of the classification as a feature might not be enough for the algorithm to recognize it as a strong feature. |
| id_3 | ML-Feature Creation Total precipitation | Created an additional column as total precipitations         | 0.14306     | When doing the classification, there is a problem with imbalanced classes.<br />Adding the results of the classification as a feature might not be enough for the algorithm to recognize it as a strong feature. |
| id_4 | DL-Baseline                             | Baseline                                                     | 0.28376     |                                                              |
| id_5 | DL-Tuned batch_size and epochs          | Baseline with batch_size=512, 100 epochs                     | 0.14689     | Modify layers.<br />Change inputs                            |
| id_6 | ML-Boolean vars for LC_Type1_mode       | Created boolean vars. **Does not work on the ML model (increases error).** | 0.13975     |                                                              |
| id_7 | ML- ID_6 with Scaler                    | Used Standard Scaler in the pipeline                         | 0.14043     | Not an improvement of score.                                 |
| id_8 | ML-New STD feature added                | Summary of the work done:<br />- Feature creation:<br />--> STD variable on rain columns<br />--> Total precipitation column<br />--> Flooded or not flooded variable (created through classification algorithm for test set)<br />--> Categorical variable switched to dummies<br />- Pipeline:<br />--> Standard Scaler<br />--> XGBRegressor | 0.13889     |                                                              |
| id_9 | ML-Same as id_8 with RandomForest       | RandomForestRegressor with searchgrid                        | 0.16302     | Worse than XGBoost.                                          |
