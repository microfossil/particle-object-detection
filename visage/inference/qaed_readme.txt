Implementation of sequence-based metric for evaluating COTS detection model and tracker, located here:
https://bitbucket.csiro.au/projects/VISAGE/repos/visage-ml/browse/visage/inference/qaed_metric.py

It compares sequences to sequences and calculates a precision / recall / f2score. A sequence is considered a true positive if the average IoU overlap between it's bboxs and the groundtruth bboxs is greater than 0.3. 

Input to the script above is the "detections.csv" that the detector/tracker will produce when run over the test-set and the ground truth JSON files. e.g. Example:

python3 qaed_metric.py --detections ./GX010008/detections.csv ./GX030005/detections.csv --groundtruth ./Reef_22_088a_SE_sector_20211002_GX010008-test.json ./Reef_22_088a_SE_sector_20211002_GX030005-test.json

To produce: 
Score: 0.61 - Took 104 tracks (19 fp, prec=81.7%) to find 86/118 (72.9%) groundtruth sequences = F2: 0.74
Score: 0.60 - Took 107 tracks (21 fp, prec=80.4%) to find 87/118 (73.7%) groundtruth sequences = F2: 0.75
Score: 0.54 - Took 117 tracks (29 fp, prec=75.2%) to find 88/118 (74.6%) groundtruth sequences = F2: 0.75
Score: 0.51 - Took 127 tracks (38 fp, prec=70.1%) to find 89/118 (75.4%) groundtruth sequences = F2: 0.74
Score: 0.43 - Took 141 tracks (51 fp, prec=63.8%) to find 90/118 (76.3%) groundtruth sequences = F2: 0.73
Score: 0.43 - Took 142 tracks (51 fp, prec=64.1%) to find 91/118 (77.1%) groundtruth sequences = F2: 0.74
Score: 0.40 - Took 152 tracks (60 fp, prec=60.5%) to find 92/118 (78.0%) groundtruth sequences = F2: 0.74
Score: 0.37 - Took 163 tracks (70 fp, prec=57.1%) to find 93/118 (78.8%) groundtruth sequences = F2: 0.73
Score: 0.26 - Took 210 tracks (116 fp, prec=44.8%) to find 94/118 (79.7%) groundtruth sequences = F2: 0.69
Score: 0.21 - Took 223 tracks (128 fp, prec=42.6%) to find 95/118 (80.5%) groundtruth sequences = F2: 0.68
Maximum F2 Score: 0.750

