# Commom Calibration methods for multivariate calibration

This is a Python library for dealing with multivariate calibration, e.g., Near infrared spectra regression and classification tasks.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pynir

```bash
pip install pynir
```
## Usage
### Simulata NIR spectra (spc) and reference values (conc) or load your own data
```python
from pynir.utils import simulateNIR

spc, conc = simulateNIR()
```

## Demon
Feature selection demostration of corn sample near infrared (NIR) spectra by **Monte Carlo-uninformative variable elimination (MC-UVE)**, **randomization test(RT)**, [**Variable selection _via_ Combination (VC)**, and **multi-step VC（MSVC）**](https://link.springer.com/article/10.1007%2Fs11426-018-9368-9).
```python
python FeatureSelectionDemo_mcuve.py
python FeatureSelectionDemo_RT.py
python FeatureSelectionDemo_VC.py
python FeatureSelectionDemo_MSVC.py

```

## Ref
_1. Cai, W. S.;  Li, Y. K.; Shao, X. G., A variable selection method based on uninformative variable elimination for multivariate calibration of near-infrared spectra. Chemom. Intell. Lab. Syst. 2008, 90 (2), 188-194._

_2. Xu, H.;  Liu, Z. C.;  Cai, W. S.; Shao, X. G., A wavelength selection method based on randomization test for near-infrared spectral analysis. Chemom. Intell. Lab. Syst. 2009, 97 (2), 189-193._

[_3. Zhang, J.;  Cui, X. Y.;  Cai, W. S.; Shao, X. G., A variable importance criterion for variable selection in near-infrared spectral analysis. Sci. China Chem. 2018, 62, 271–279._](https://link.springer.com/article/10.1007%2Fs11426-018-9368-9)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
