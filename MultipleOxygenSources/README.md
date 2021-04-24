# Modeling of hypoxia in breast cancer
This model demonstrates the evaluation of the new hypotheses on the frequency and duration of phenotypic changes in cancer cells under the influence of hypoxic conditions.

This model is part of a collaboration between Dr. Macklin's lab and Dr. Gilkes's lab, through of a project granted by Jayne Koskinas Ted Giovanis (JKTG) Foundation for Health and Policy and the Breast Cancer Research Foundation (BCRF). It is also part of the education and outreach for the IU Engineered nanoBIO Node and the NCI-funded cancer systems biology grant U01CA232137. The model is built using PhysiCell (version 1.6.1): a C++ framework for multicellular systems biology.

![alt ensure executable](https://raw.githubusercontent.com/heberlr/Hypoxia_simulator/master/doc/model.png)

To compile and run [PhysiCell](http://physicell.mathcancer.org/) for this model:
```
# your compiler needs to support OpenMP
$ make
$ ./hypoxia
```

The cloud-hosted interactive model can be run at https://nanohub.org/tools/pc4tumorhypoxia
