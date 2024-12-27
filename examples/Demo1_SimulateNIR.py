from pathlib import Path
import sys

current_dir = Path(__file__).parent
parent_dir = current_dir.parent.resolve()
pynir_dir = parent_dir / 'src'
if str(pynir_dir) not in sys.path:
    sys.path.insert(0, str(pynir_dir))

import numpy as np
import matplotlib.pyplot as plt
from pynir.utils import simulateNIR
from pynir.Reader import spaReader, InnoSpectraNIRReader
import matplotlib
matplotlib.use('agg')

# Generating X, y and wv variables using simulateNIR function
X, y, wv = simulateNIR(nSample=200, n_components=10, noise=1e-5)

fig, ax = plt.subplots(2)
# Plotting intensity values against wavelength
ax[0].plot(wv, np.transpose(X))
ax[0].set_xlabel("wavelength (nm)")
ax[0].set_ylabel("Intensity (a.u.)")

# Plotting histogram of reference values
ax[1].hist(y)
ax[1].set_xlabel("Reference values")
ax[1].set_ylabel("Count")

# Save the figure to the tmp folder
fig.savefig(current_dir / 'tmp' / 'intensity_vs_wavelength.png')

# 读取spa光谱文件示例
spa_file_path = pynir_dir / 'pynir' / 'demo_data' / 'spa_unknown'
reader = spaReader(
                    fix_nan=True,
                    smooth=True,
                    abs_threshold=(0.0, 5.0),  # 根据实际数据调整阈值
                    window_length=7,
                    polyorder=2,
                    outlier_window=11,
                    z_score_threshold=3,
                    interpolation_method='linear'  # 可选 'spline', 'polynomial', etc.
                )
spectra_data = reader.read_spectra_from_directory(str(spa_file_path))
spectra_df = reader.spectra_to_dataframe(spectra_data)

plt.figure()
plt.plot(spectra_df.columns, spectra_df.T)
plt.title('Spectra from all files')
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.savefig(current_dir / 'tmp' / 'spa_spectra.png')
plt.close()


# 读取InnoSpeca光谱文件示例
innospec_file_path = pynir_dir / 'pynir' / 'demo_data' / 'csv_InnoSpectra'
innospec_reader = InnoSpectraNIRReader(
                                        fix_nan=True,
                                        smooth=True,
                                        signal_threshold=500,  # 根据实际数据调整阈值
                                        window_length=7,
                                        polyorder=2,
                                        outlier_window=11,
                                        z_score_threshold=3
                                    )
spectra_data = innospec_reader.read_spectra_from_directory(str(innospec_file_path))
spectra_df = innospec_reader.spectra_to_dataframe(spectra_data)

plt.figure()
plt.plot(spectra_df.columns, spectra_df.T)
plt.title('Spectra from all files')
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.savefig(current_dir / 'tmp' / 'innospectra_spectra.png')