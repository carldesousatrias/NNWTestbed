
# NNW-Testbed

NNW-Testbed is a Python project, designed to help people understand and benchmark neural network watermarking methods. \
The different watermarking methods, models, attacks, and datasets are stored in the NNWmethods, Architectures, Attacks, Data folders, respectively. \
The NNWResources folder contains subfolders linked to NN watermarking methods requiring specific external elements, as for example an image serving as watermark or a set of inputs serving as trigger dataset. \
The file utils.py regroups the support functions, transversal with respect to the NN watermarking methods, like reading input data or inference producing. \
The file main.py is the main function of the testbed, covering the NN watermarking workflow presented in the associated paper.\
\
**Note**: Pictures, details and illustration should be added uppon acceptance of the paper.
## Installation
unzip the NNWresources.zip folder. \

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the libraries.

```bash
pip install -r requirements.txt
```

## Usage
All the steps occured in the main.py file and should be run as follow:
```python
python main.py
```
### Structure
The first few lines of the file permit to select the architecture and the linked training hyperparameters. \
The bottom line of each NNWmethods should replace the designed part '#TBM' to modify the applied watermarking method. \
Finally the needed parameter for the Attacker section are details in Attacks/main_attack.py . \
Details can be found in the [paper](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9739).
### Datasets
The implemented datasets are MNIST, CIFAR10, and CIFAR100 but you can add any other using the ImageFolderClass in utils.py.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Acknowledgment

This work is carried out under the framework of the NewEmma project (Neural nEtwork Watermarking for Energy efficient Mobile Multimedia Applications) founded by the DIGICOSME laboratory of excellence in France.

## License

[CC](https://creativecommons.org/about/program-areas/software/)
