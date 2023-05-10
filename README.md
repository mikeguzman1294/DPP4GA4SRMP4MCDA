# DPP4GA4SRMP4MCDA
## Determinantal Point Processes for Genetic Algorithms for Simple Ranking Multiple Profiles for Multi-criteria Decision Aid

***
This repository contains the implementation of the DPP parent selection logic for a genetic algorithm described in "*Utilisation de processus ponctuels déterminantaux pour la sélection de parents dans un algorithme génétique : application à l’aide multi-critère à la décision*". The aforementioned code can be found in the "**kendall-tau" branch** (this branch name is just a temporary placeholder - TO BE CHANGED).

The **main branch** contains the code implementation of Algorithm 1 from "*A metaheuristic for inferring a ranking model based on multiple reference profiles*" by Arwa Khannoussi, Patrick Meyer, Alexandru-Liviu Olteanu and Bastien Pasdeloup, submitted to Springer Annals of Mathematics and Artificial Intelligence, which is the baseline genetic algorithm used for our DPP experimentation. This baseline implementation can also be found in the original author's repository [https://github.com/BastienPasdeloup/learn_srmp].

The corresponding scientific paper submitted to GRETSI conference 2023 can be found in the Documentation folder under the file name "DPP4GA4SRMP4MCDA.pdf".

***

**Abstract** – *Multi-criteria decision aid is a research field aimed at providing decision-makers with tools to assist in decision-making
based on multivariate data. Among these tools, the Ranking with Multiple Profiles (RMP) model allows for ordering a set of
alternatives by pairwise comparison to reference profiles. The optimal configuration of such a model can be complex due to multiple
parameters involved. Research has shown the usefulness of genetic algorithms in determining the parameters of this model. These
algorithms aim to evolve a population of solutions through a series of crossover and mutation operators between parent solutions to
produce child solutions. However, genetic algorithms tend to converge to a highly homogeneous population of solutions. In this
context, we chose to study the contribution of Determinantal Point Processes (DPP) in parent selection. These probabilistic models
allow for random sampling within the population of solutions, with diversity properties among the selected solutions.*

***

If you wish to cite this, please use the following Bibtex entry: TODO

Contact regarding the code: miguel.guzman@imt-atlantique.fr