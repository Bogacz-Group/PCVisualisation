# Visualising predictive coding networks

[![UI Screenshot](src/ui.png)](src/ui.png)

## Description
This repository visualizes inference and learning in predictive coding networks. The objective is to play around with the visualisation to get an intuition of how PCN perform inference and learn. This code can visualise deep linear PCNs with an arbitrary number of layers and one neuron per layer. The visualised PCN contains 

An input layer 
-   is shown at the bottom of the visualisation
-   mean and std of Gaussian inputs can be set

A latent layer 
-   is shown at the top of the visualisation
-   has a constant activity

Hidden layers
-   its activity change during inference 
-   are randomly initialised before the first inference step ($n=0$)

Neural activity, weights and biases:
-   the activity of layer is shown using a dot
-   weights and biases generate top-down prediction
-   the top-down predictions are shown using rods
-   springs show the top-down prediction errors in each layer

Inference dynamics modes
-   standard predictive coding inference
-   Monte Carlo predictive coding inference
-   mixed mcpc (in layer L-1) and pc inference
-   "posterior" inference is a fast iterative inference mode where the activity of a neuron is updated to the activity with (locally) minimal energy. The activity update is given by $x_{l, t+1} = (w_l x_{l-1} + b_l + w_{l-1}(x_{l-1} - b_{l-1}))/(w_{l-1}^2 + 1)$. This inference scheme is an faster alternative to pc inference however, it requires computing the inverse of weights.
-   "root" inference is another fast alternative to pc which does not require taking the inverse of weights. However, it is only stable for weights smaller than 1. This inference scheme is written as $x_{l, t+1} = w_l x_{l-1} + b_l + w_{l-1} (x_{l-1} - w_{l-1} x_{l} - b_{l-1})$ and is a [fixed point iteration method to find the point of minimal energy](https://en.wikipedia.org/wiki/Root-finding_algorithms#:~:text=fixed%20point%20iteration%20method).

Weight and bias dynamics
-   weights and biases are update after a fixed number of inference steps following pc updates (=mcpc updates)
-   parameters can also have decay and be bounded


## Installation
The visualisation is writen in python using numy and matplotlib. It can be downloaded using:
1. Clone the repository: `git clone https://github.com/Bogacz-Group/PCVisualisation.git`
2. Install the dependencies: `pip install install -r requirements.txt`

## Usage
Run `python pc_app.py` and give the UI a shot.

## Contributing
Contributions are welcome!.  Dont hesitate to modify, add, etc. to the repository by creating a branch and creating a pull request when you are happy with your changes. 


## Contact
For any questions or feedback, please contact [gaspard.oliviers@pmb.ox.ac.uk](mailto:gaspard.oliviers@pmb.ox.ac.uk).