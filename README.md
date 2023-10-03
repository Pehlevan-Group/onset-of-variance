# onset-of-variance
This is the code to reproduce the experiments and figures in "The Onset of Variance Limited Behavior for Networks in the Lazy and Rich Regimes" https://arxiv.org/abs/2212.12147

The principal script is `Scripts/pn_sweeps.py`. This allows one to sweep over width, depth, dataset size, polynomial degree, initialization seed, dataset seed, number of ensembles. 
 
This also requires the [kernel_generalization](https://github.com/Pehlevan-Group/kernel-generalization/tree/main) package to be installed with

    !pip install -q git+https://github.com/Pehlevan-Group/kernel-generalization

The notebooks to reproduce the major figures are in `PaperFigures`
