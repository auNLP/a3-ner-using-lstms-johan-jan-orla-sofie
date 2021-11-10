
*see assignment_description.md for a description of the assignment*

# Summary

<!-- 
This should include a short description of which models you have tried and conclusions from comparing these models. This should be no longer than an abstract. This section can also include questions regarding the assignment.
-->

# Performance
<!-- 
This should include a table of performance metrics of different models. The performance metrics should at least include accuracy and F1-score.
 -->

## Project Organization
The organization of the project is as follows:

<!-- 
Correct this to reflect changes
-->

```
├── LICENSE                    <- the license of this code
├── README.md                  <- The top-level README for this project.
├── .github            
│   └── workflows              <- workflows to automatically run when code is pushed
│   │    └── pytest.yml        <- A workflow which runs pytests upon push
├── classification             <- The main folder for scripts
│   ├── tests                  <- The pytest test suite
│   │   └── ...
|   └── ...
├── .gitignore                 <- A list of files not uploaded to git
├── requirement.txt            <- A requirements file of the required packages.
└── assignment_description.md  <- the assignment description
```



## Running the code
You can run the reproduce all the experiments by cloning the GitHub repository and running the following:

<!-- 
Update the code below such that it runs all the experiments in the performance section and print the performances.
-->

```
pip install -r requirements.txt
python ner/main.py --epochs 10 --gensim_embedding glove-wiki-gigaword-50
```