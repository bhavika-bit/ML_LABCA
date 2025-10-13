# Results Summary

- Random Forest Accuracy: ~0.99 (example)
- XGBoost Accuracy: ~0.997 (example)
- Voting Classifier Accuracy: ~0.997 (example)
- Cross-validation mean accuracy: ~0.9938 (example)

## Example prediction (sample run)

## Visualizations
- See `notebooks/Colab_Notebook.ipynb` for scatter plots (N vs P), rainfall vs humidity, treemap for rainfall effect, and model accuracy bar chart.
- Power BI dashboard screenshot and notes: use continuous X-axis for forecast, and Crop (categorical) for counts.

## Limitations & Future Work
- Consider adding temperature as a training feature (requires merging LST with historical dataset on temporal & spatial keys).
- Improve data balancing for rare crops.
- Provide an API to integrate predictions into a web dashboard.
