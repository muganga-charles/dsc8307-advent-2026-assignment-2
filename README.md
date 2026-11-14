# DSC8307 Data Mining, Modeling and Analytics: Advent 2026 Assignment 2
## MUGANGA CHARLES B35391
This repository contains my submission for the two projects in the second DSC8307 assignment. Each project has its own Jupyter notebook, and both run from top to bottom after installing the packages in `requirements.txt`.

## What is in here

`project1_wholesale_segmentation.ipynb` is Project 1. It segments 440 wholesale customers by their spending, checks whether the data has any cluster tendency, compares partitioning, hierarchical and density based clustering, and names the segments with suggestions for serving each one.

`project2_fraud_detection.ipynb` is Project 2. It builds detectors that rank card transactions for fraud investigation, compares a statistical baseline, a cluster based approach and a classification based approach against the withheld fraud labels, and explains why an unusual transaction is not necessarily fraudulent.

`requirements.txt` lists the exact package versions used.

The `data` folder holds the datasets and their README files. The credit card file is 150 MB, which is larger than GitHub allows through its website, so I have stored it in `data/creditcard` as four compressed parts of about 17 MB each. The notebook joins them back together and checks that the result has exactly the 284,807 rows and 492 frauds described in the original README. Pandas reads the compressed parts directly, so nothing needs to be unzipped by hand.

## Summary of findings

### Project 1, wholesale customers

Spending is extremely skewed, so I clustered on log transformed, standardised spending, keeping channel and region aside to check the results afterwards. The Hopkins statistic is about 0.80, above both uniform data (0.50) and a single structureless blob with the same correlations (0.73), so there is real but moderate cluster tendency.

K means gave the best quality scores and the most stable clusters. Ward clustering agreed with it on the main split, while DBSCAN and HDBSCAN found only one dense core plus a fringe of atypical customers, because there are no low density gaps between groups. The two cluster split is very robust and closely matches the Horeca and retail channels even though channel was never used. The three working segments are Fresh food kitchens (212 customers), Big basket all rounders (147) and Packaged goods shops (81), each with suggestions for how the distributor could serve them.

### Project 2, credit card fraud

The data was split in time order, with detectors built on the earliest 70 percent of transactions and judged on the most recent 30 percent. Unsupervised detectors (the Mahalanobis baseline, k means distance and a one class SVM) rank fraud above average overall but have no frauds at all in their top 100. A cluster rarity score, which flags members of very small clusters, did better because k means isolated a tiny cluster that turned out to be about 80 percent fraud, but that fraud pattern faded in the later period. Supervised gradient boosting found 82 frauds in its top 100 and reached an average precision of about 0.80, against at most about 0.11 for any method that does not learn from fraud labels.

None of the 100 most statistically unusual test transactions was fraud, and in the most unusual 1 percent, 94 percent of transactions were legitimate. This is the evidence behind the final section on why an unusual transaction is not necessarily fraudulent.

## Data sources

Wholesale customers data: Abreu, N. (2011). Análise do perfil do cliente Recheio e desenvolvimento de um sistema promocional. Master's dissertation, ISCTE IUL, Lisbon. Distributed through the UCI Machine Learning Repository.

Credit card fraud data: Dal Pozzolo, A., Caelen, O., Johnson, R. A. and Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE Symposium Series on Computational Intelligence. Released by the Machine Learning Group of the Université Libre de Bruxelles and Worldline.

Full references for the methods used are listed at the end of each notebook.
