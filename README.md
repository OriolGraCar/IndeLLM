# IndeLLM
Welcome to IndeLLM, an easy and free to use indel pathogenicity predictor based on Protein Language Models.

IndeLLM also provides per position protein fitness damage score to aid in interpreting the provided predictions. 

A plug and play google colab can be found at (to be updated)

The acompaining paper can be found at (to be published)

If you use IndeLLM please cite us as: (to be published)

![IndeLLM graphical abstract](img/graphicalabstract.png)

## Installation

To install IndeLLM first clone the repository.
The needed packages can be found on requirements.txt and can be installed with:
```
pip3 install -r requirements.txt
```
To use IndeLLM as a package add the repository to your PYTHONPATH.

## Folder structure and contents

*   The Analysis folder contains the data and jupyter notebooks needed to reproduce the analysis showcased in the manuscript.

*   The Data folder contains the csv files for all the predictions, sequences used and the train/test/validation data splits used.
More information can be found in the README found inside the Data folder.
*   The Examples folder contains the jupyter notebook and necessary files to reproduce the data interpretation examples showcased in the manuscript.
*   The colab folder contains the source code for the provided google colab notebook.
*   The models folder contains the trained IndeLLM siamese model.
*   The indellm folder contains the source code for the package. The code is structured in 4 files. berteval.py handles the protein language models; model.py contains the code for the training and inference of the siamese model; scorer.py contains the code for the zero-shot scoring using protein language models; utils.py contains the code to handle and process the sequences.
*   The folder scripts contains example scripts on how to run the different functionality of the code.

## Abstract

Protein language models (PLM) have revolutionised the variant effect prediction field. These models achieve high performance and accuracy without requiring complex architectures, sometimes outperforming dedicated methods even when using a zero-shot approach. From the realm of pathogenicity predictors, the study of in-frame insertion and deletions (indels) remains challenging to study with PLMs due to the differences in lengths and few available datasets. Despite that, approaches have emerged to leverage the advantages PLMs provide through transfer learning. However, there is no current consensus of which PLMs and transfer learning protocols better capture the features relevant to indel prediction, and these approaches lack interpretability for the provided predictions. In this work, we devise new scoring approaches (named IndeLLM) for zero-shot inference using PLMs on tasks that they were not trained for. Using this score we benchmarked different high-performing PLMs.  IndeLLM achieves similar performances to other pathogenicity predictors using only sequence information and a fraction of the computing time. We also constructed a simple transfer learning approach for a Siamese network that allowed for state-of-the-art performance (Matthews correlation coefficient = 0.77) on indel predictions. Finally, we design an approach that allows us to easily visualise the impact of the indels on the rest of the protein to aid in interpreting the results. Since PLMs are trained on protein sequence through evolution, IndeLLM is universally applicable across all species. 