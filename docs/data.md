# Data provenance and prediction timing

Source: [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Fanaee-T, H. (2013), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894). Dataset license: CC BY 4.0. The code has a separate MIT license.

The downloaded `hour.csv` contains 17,379 observed hourly records from 1 January 2011 through 31 December 2012. Its SHA256 is `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`. Run `python scripts/download_data.py` to obtain the original files. The downloadable project package includes the raw data for offline use; raw data are excluded from Git.

Target: total hourly rentals (`cnt`). Predictors: observed weather, hour and calendar indicators. These are **contemporaneous demand estimates given observed weather**, not operational forecasts made before weather is known. The simulator predicts a complete weekly batch before exposing its targets, then assumes those targets become available at the batch end. A production system would need explicit label-latency handling.

`casual` and `registered` sum exactly to the target and are excluded. `instant` and `yr` are excluded. There are no missing values in the selected columns; absent hourly records are not filled with zeros.

Chronology: January–September 2011 training; October 2011 model selection; November–December 2011 threshold calibration; 2012 monitoring. Only batches with at least 140 observations enter the weekly experiment. Boundary/sparse-week exclusions are counted in the notebook.
