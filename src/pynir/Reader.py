import os
import re
from datetime import datetime, timezone, timedelta
import struct
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class InnoSpectraNIRReader:
    def __init__(self, fix_nan=True, smooth=True, signal_threshold=500, window_length=7, polyorder=1,
                 outlier_window=11, z_score_threshold=3):
        """
        Initialize method with outlier detection parameters.
        :param outlier_window: Rolling window size for calculating local statistics.
        :param z_score_threshold: Z-score threshold; points exceeding this are considered outliers.
        :param signal_threshold: Signal intensity threshold; absorbance below this value is set to NaN.
        """
        self.fix_nan = fix_nan
        self.smooth = smooth
        self.signal_threshold = signal_threshold  # Signal intensity too low, possibly high noise, needs to be filtered out
        self.window_length = window_length
        self.polyorder = polyorder
        self.outlier_window = outlier_window
        self.z_score_threshold = z_score_threshold

    def read_spectrum(self, file_path):
        """
        Reads a single NIR spectrum CSV file and returns the parsed data.
        """
        raw_data = pd.read_csv(file_path, header=None)
        block_patterns = [
            "***Scan Config Information***",
            "***Reference Scan Information***",
            "***General Information***",
            "***Calibration Coefficients***",
            "***Lamp Usage ***",
            "***Device/Error/Activation Status***",
            "***Scan Data***"
        ]
        block_positions = self._find_blocks(raw_data, block_patterns, block_widths=[4]*len(block_patterns))
        parsed_results = {}

        for block_name, block_info in block_positions.items():
            block_data = self._extract_block(raw_data, block_info)
            if block_name == "***Scan Data***":
                parsed_results[block_name] = self._parse_scan_data_block(block_data)
            else:
                parsed_results[block_name] = self._parse_key_value_block(block_data)

        return parsed_results
    
    def read_spectra_from_directory(self, directory_path):
        """
        Reads all NIR spectrum CSV files in a directory and returns a dictionary of parsed data.
        """
        spectra_data = {}
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory_path, filename)
                spectra_data[filename] = self.read_spectrum(file_path)
        return spectra_data

    def spectra_to_dataframe(self, spectra_data):
        """
        Converts a dictionary of parsed spectra data into a pandas DataFrame containing only absorbance values.
        The DataFrame will have sample names as row indices and wavelengths as column names.
        """
        spectra_list = []
        sample_names = []
        wavelengths = None

        for sample_name, data in spectra_data.items():
            scan_data = data.get("***Scan Data***")
            if scan_data is not None:
                if wavelengths is None:
                    wavelengths = scan_data["Wavelength (nm)"].values
                absorbance = scan_data["Absorbance (AU)"].values
                spectra_list.append(absorbance)
                sample_names.append(sample_name)

        if wavelengths is None or not spectra_list:
            raise ValueError("No valid spectra data found.")

        df = pd.DataFrame(spectra_list, index=sample_names, columns=wavelengths)
        return df

    def _find_blocks(self, data, block_patterns, block_widths):
        """
        Identifies blocks in the data based on specified block patterns and widths.
        Returns a dictionary with block names as keys and start/end positions as values.
        """
        blocks = {}
        for pattern, width in zip(block_patterns, block_widths):
            match = data.apply(lambda x: x.apply(lambda y: isinstance(y, str) and pattern in y))
            if match.any().any():
                start_row, start_col = match.stack().idxmax()
                blocks[pattern] = {'start_row': start_row, 'start_col': start_col, 'block_width': width}

        block_names = list(blocks.keys())
        for i, block_name in enumerate(block_names):
            start_row = blocks[block_name]['start_row']
            start_col = blocks[block_name]['start_col']
            block_width = blocks[block_name]['block_width']
            end_col = start_col + block_width - 1
            end_row = start_row
            while end_row < len(data) - 1:
                next_row = end_row + 1
                if pd.isna(data.iloc[next_row, start_col]) or (data.iloc[next_row, start_col] == ''):
                    break
                end_row = next_row
            blocks[block_name].update({'end_row': end_row, 'end_col': end_col})
        return blocks

    def _extract_block(self, data, block_info):
        """
        Extracts a block of data based on the start and end rows/columns of the block.
        """
        start_row = block_info['start_row']
        end_row = block_info['end_row']
        start_col = block_info['start_col']
        end_col = block_info['end_col']
        block_data = data.iloc[start_row:end_row + 1, start_col:end_col + 1]
        return block_data

    def _parse_key_value_block(self, block):
        """
        Parses a key-value block into a dictionary.
        """
        parsed_data = {}
        for i in range(len(block)):
            row = block.iloc[i, :].dropna()
            if len(row) > 1:
                for j in range(0, len(row), 2):
                    key = row.iloc[j].strip() if j < len(row) else None
                    value = row.iloc[j + 1].strip() if (j + 1) < len(row) else None
                    if key and value:
                        parsed_data[key] = value
            else:
                row_str = row.values[0]
                if ":" in row_str:
                    key, value = row_str.split(":", 1)
                    parsed_data[key.strip()] = value.strip()
        return parsed_data

    def _parse_scan_data_block(self, block):
        """
        Parses the scan data block and processes absorbance data.
        """
        block.reset_index(drop=True, inplace=True)
        block.columns = block.iloc[1].values  # Assume the second row is the column names
        block = block.iloc[2:].reset_index(drop=True)  # Data starts from the third row
        block = block.apply(pd.to_numeric, errors='coerce')

        if self.fix_nan:
            if self.signal_threshold:
                block["Absorbance (AU)"] = block["Absorbance (AU)"].where(
                    block["Sample Signal (unitless)"] >= self.signal_threshold, np.nan)

            # Outlier detection: based on rolling window Z-score method
            absorbance = block["Absorbance (AU)"].values
            window = self.outlier_window
            half_window = window // 2

            # Calculate median and MAD using rolling window
            median = pd.Series(absorbance).rolling(window=window, center=True, min_periods=1).median()
            mad = pd.Series(absorbance).rolling(window=window, center=True, min_periods=1).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True)

            # Calculate Z-score
            z_score = (absorbance - median) / (mad + 1e-6)  # Prevent division by zero

            # Mark outliers
            outliers = np.abs(z_score) > self.z_score_threshold
            block["Absorbance (AU)"] = np.where(outliers, np.nan, absorbance)

            # Interpolate NaN values (including signal threshold and outliers)
            block["Absorbance (AU)"] = block["Absorbance (AU)"].interpolate(
                method='linear', limit_direction='both')

        if self.smooth:
            # Ensure window length is odd and not larger than data length
            actual_window_length = self.window_length if self.window_length % 2 == 1 else self.window_length + 1
            actual_window_length = min(actual_window_length, len(block["Absorbance (AU)"]) - (len(block["Absorbance (AU)"]) + 1) % 2)
            block["Absorbance (AU)"] = savgol_filter(block["Absorbance (AU)"], actual_window_length, self.polyorder)
        return block


class spaReader:
    def __init__(
        self,
        fix_nan=True,
        smooth=True,
        abs_threshold=None,       # e.g. [0.0, 3.0], if None then threshold filter is not applied
        window_length=7,
        polyorder=2,
        outlier_window=11,
        z_score_threshold=3,
        interpolation_method='linear'
    ):
        """
        Initialize the spaReader class with outlier detection and smoothing parameters.

        Parameters
        ----------
        fix_nan : bool
            Whether to fix NaN (threshold filtering + outlier detection + interpolation).
        smooth : bool
            Whether to apply Savitzky-Golay smoothing.
        abs_threshold : list or None
            If [min_val, max_val], set absorbance outside this range to NaN; if None, no threshold filtering.
        window_length : int
            Window length for Savitzky-Golay filter (must be an odd number).
        polyorder : int
            Polynomial order for Savitzky-Golay filter.
        outlier_window : int
            Rolling window size for Z-score calculation.
        z_score_threshold : float
            Z-score threshold; values exceeding this are considered outliers.
        interpolation_method : str
            Interpolation method such as 'linear', 'spline', 'polynomial', etc.
        Ref: https://github.com/spectrochempy/spectrochempy/blob/master/spectrochempy/core/readers/read_omnic.py
        """
        self.fix_nan = fix_nan
        self.smooth = smooth
        self.abs_threshold = abs_threshold
        self.window_length = window_length
        self.polyorder = polyorder
        self.outlier_window = outlier_window
        self.z_score_threshold = z_score_threshold
        self.interpolation_method = interpolation_method

    # --------------------- Main entry: read a single SPA file --------------------- #
    def read_spectrum(self, file_path):
        """
        Read a single SPA file and return the processed spectrum data 
        (DataFrame with ['Wavenumber', 'Absorbance']).
        """
        with open(file_path, 'rb') as fid:
            # 1) Read the original Omnic file name (for record)
            spa_name = self._readbtext(fid, pos=30, size=256)  # offset=0x1E=30

            # 2) Get acquisition timestamp (Omnic records time in seconds from 1899-12-31)
            fid.seek(296)  # 0x128=296
            raw_timestamp = self._fromfile(fid, "uint32", 1)
            acq_date = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=int(raw_timestamp))

            # 3) Starting from 304 (0x130), read “key-value” entries until reaching key=0 or key=1 or until spectrum data is found
            pos = 304
            intensities = None
            info = {}
            spa_comments = []
            spa_history = None

            while True:
                fid.seek(pos)
                key_bytes = fid.read(1)
                if not key_bytes or len(key_bytes) < 1:
                    # File ended
                    break

                key = key_bytes[0]  # uint8

                if key == 2:
                    # Spectrum header
                    fid.seek(pos + 2)
                    pos_header = self._fromfile(fid, "uint32", 1)
                    info = self._read_header_spa(fid, pos_header)
                elif key == 3:
                    # Actual spectrum data
                    intensities = self._getintensities(fid, pos)
                elif key == 4:
                    # User-defined text
                    fid.seek(pos + 2)
                    comments_pos = self._fromfile(fid, "uint32", 1)
                    fid.seek(pos + 6)
                    comments_len = self._fromfile(fid, "uint32", 1)
                    fid.seek(comments_pos)
                    spa_comments.append(fid.read(comments_len).decode("latin-1", "replace"))
                elif key == 27:
                    # History text
                    fid.seek(pos + 2)
                    hist_pos = self._fromfile(fid, "uint32", 1)
                    fid.seek(pos + 6)
                    hist_len = self._fromfile(fid, "uint32", 1)
                    spa_history = self._readbtext(fid, hist_pos, hist_len)
                elif key == 0 or key == 1:
                    # End
                    break

                pos += 16

            # If there is no spectrum data
            if intensities is None:
                raise ValueError(f"No valid spectrum data found in {file_path}.")

        # -- Construct Wavenumber/Absorbance DataFrame --
        # If the file does not contain header info, assign default values
        nx = info.get("nx", len(intensities))
        firstx = info.get("firstx", 4000.0)
        lastx = info.get("lastx", 400.0)
        xunits = info.get("xunits", "cm^-1")
        data_title = info.get("title", "Absorbance")

        # Generate x axis
        # Omnic often has firstx > lastx, but it could be reversed
        xvals = np.linspace(firstx, lastx, nx)
        if firstx < lastx:
            # x from small to large
            pass
        else:
            # x from large to small
            xvals = xvals[::-1]

        # Safe check if intensities length != nx
        if len(intensities) != nx:
            min_len = min(len(intensities), nx)
            xvals = xvals[:min_len]
            intensities = intensities[:min_len]

        df = pd.DataFrame({
            "Wavenumber": xvals,
            "Absorbance": intensities
        })

        # (Optional) NaN/outlier handling
        if self.fix_nan:
            df = self._fix_nan(df)

        # (Optional) smoothing
        if self.smooth:
            df = self._smooth_spectrum(df)

        return df

    # ------------- Batch read SPA files in a directory ------------- #
    def read_spectra_from_directory(self, directory_path):
        """
        Read all `.spa` files in a directory and return a dict {filename: DataFrame}.
        """
        spectra = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.spa'):
                file_path = os.path.join(directory_path, filename)
                try:
                    spectra[filename] = self.read_spectrum(file_path)
                except ValueError as e:
                    print(f"[Error] File {filename} read failed: {e}")
        return spectra

    # ------------- Combine into a DataFrame ------------- #
    def spectra_to_dataframe(self, spectra_dict):
        """
        Combine {filename: DataFrame} into a large DataFrame:
            - Row index = filename
            - Column index = Wavenumber
        """
        if not spectra_dict:
            raise ValueError("The input spectra_dict is empty and cannot be converted to a DataFrame.")

        first_key = next(iter(spectra_dict))
        wavenumbers = spectra_dict[first_key]["Wavenumber"].values

        combined_df = pd.DataFrame(index=spectra_dict.keys(), columns=wavenumbers, dtype=float)

        for fn, df_sp in spectra_dict.items():
            combined_df.loc[fn, :] = df_sp["Absorbance"].values

        return combined_df

    # ------------- Private helper: read header (refer to SpectroChemPy) ------------- #
    def _read_header_spa(self, fid, pos_header):
        """
        Read spectrum header info (refer to SpectroChemPy _read_header)
        Return a dict like {"nx":..., "firstx":..., "lastx":..., "units":..., ...}
        """
        info = {}
        fid.seek(pos_header)

        # nx (UInt32) at offset pos_header+4
        fid.seek(pos_header + 4)
        nx = self._fromfile(fid, "uint32", 1)
        info["nx"] = nx

        # xunits key (UInt8)
        fid.seek(pos_header + 8)
        xkey = self._fromfile(fid, "uint8", 1)
        if xkey == 1:
            info["xunits"] = "cm^-1"
            info["xtitle"] = "wavenumber"
        elif xkey == 2:
            info["xunits"] = None
            info["xtitle"] = "datapoints"
        elif xkey == 3:
            info["xunits"] = "nm"
            info["xtitle"] = "wavelength"
        elif xkey == 4:
            info["xunits"] = "um"
            info["xtitle"] = "wavelength"
        elif xkey == 32:
            info["xunits"] = "cm^-1"
            info["xtitle"] = "raman shift"
        else:
            info["xunits"] = None
            info["xtitle"] = "xaxis"

        # data units (UInt8)
        fid.seek(pos_header + 12)
        dkey = self._fromfile(fid, "uint8", 1)
        if dkey == 17:
            info["units"] = "absorbance"
            info["title"] = "Absorbance"
        elif dkey == 16:
            info["units"] = "percent"
            info["title"] = "Transmittance"
        elif dkey == 11:
            info["units"] = "percent"
            info["title"] = "Reflectance"
        elif dkey == 12:
            info["units"] = None
            info["title"] = "Log(1/R)"
        elif dkey == 20:
            info["units"] = None
            info["title"] = "Kubelka-Munk"
        elif dkey == 21:
            info["units"] = None
            info["title"] = "Reflectance"
        elif dkey == 22:
            info["units"] = "V"
            info["title"] = "Detector Signal"
        else:
            info["units"] = None
            info["title"] = "Intensity"

        # firstx, lastx
        fid.seek(pos_header + 16)
        info["firstx"] = self._fromfile(fid, "float32", 1)
        fid.seek(pos_header + 20)
        info["lastx"] = self._fromfile(fid, "float32", 1)

        return info

    # ------------- Private helper: read spectrum/interferogram data ------------- #
    def _getintensities(self, fid, pos):
        """
        According to SpectroChemPy `_getintensities`:
        - Skip 2 bytes, read data offset
        - Skip 4 more bytes, read data size
        - Jump to offset, read float32[nintensities]
        """
        fid.seek(pos + 2)
        intensity_pos = self._fromfile(fid, "uint32", 1)
        fid.seek(pos + 6)
        intensity_size = self._fromfile(fid, "uint32", 1)

        n_points = int(intensity_size // 4)  # each float32 occupies 4 bytes
        fid.seek(intensity_pos)
        data = self._fromfile(fid, "float32", n_points)
        return data

    # ------------- Private helper: read text at a specific position ------------- #
    def _readbtext(self, fid, pos, size):
        """
        Refer to SpectroChemPy _readbtext:
        Read size bytes from file fid at position pos.
        If size=None, read until \x00. For simplicity, we assume a fixed size here.
        """
        fid.seek(pos)
        if size is None:
            btext = b""
            while True:
                c = fid.read(1)
                if c == b"\x00" or c == b"":
                    break
                btext += c
        else:
            btext = fid.read(size)

        # Replace \x00+ with \n
        btext = re.sub(b"\x00+", b"\n", btext)

        # Strip \n at beginning and end
        if btext[:1] == b"\n":
            btext = btext[1:]
        if btext[-1:] == b"\n":
            btext = btext[:-1]

        # Attempt decoding
        try:
            text = btext.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = btext.decode("latin-1")
            except UnicodeDecodeError:
                text = btext.decode("utf-8", errors="ignore")
        return text.strip()

    # ------------- Private helper: batch read binary ------------- #
    def _fromfile(self, fid, dtype, count=1):
        """
        A simple wrapper using struct to read 'count' items of 'dtype' at once.
        dtype: 'uint32', 'uint8', 'float32', etc.
        """
        format_map = {
            "uint8": "B",
            "int8": "b",
            "uint16": "H",
            "int16": "h",
            "uint32": "I",
            "int32": "i",
            "float32": "f",
        }
        if dtype not in format_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        fmt = format_map[dtype]
        size_map = {
            "B": 1, "b": 1,
            "H": 2, "h": 2,
            "I": 4, "i": 4,
            "f": 4
        }
        nbytes = size_map[fmt] * count
        raw = fid.read(nbytes)
        if len(raw) < nbytes:
            # EOF or corrupt file
            if count == 1:
                return None
            else:
                return np.array([], dtype=np.float32)

        if count == 1:
            val = struct.unpack(fmt, raw)[0]
            return val
        else:
            val = struct.unpack(fmt * count, raw)
            return np.array(val, dtype=np.float32 if fmt == 'f' else np.int64)

    # ------------- Private: outlier detection + interpolation ------------- #
    def _fix_nan(self, df):
        """
        Set absorbance out of threshold to NaN,
        then do rolling window Z-score outlier detection and set outliers to NaN,
        finally interpolate.
        """
        if self.abs_threshold is not None and len(self.abs_threshold) == 2:
            min_abs, max_abs = self.abs_threshold
            df["Absorbance"] = df["Absorbance"].where(
                (df["Absorbance"] >= min_abs) & (df["Absorbance"] <= max_abs), np.nan
            )

        # Z-score detection
        absorbance = df["Absorbance"].values
        window = self.outlier_window

        median_series = pd.Series(absorbance).rolling(
            window=window, center=True, min_periods=1
        ).median()
        mad_series = pd.Series(absorbance).rolling(
            window=window, center=True, min_periods=1
        ).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)

        z_score = (absorbance - median_series) / (mad_series + 1e-12)
        outliers = np.abs(z_score) > self.z_score_threshold
        df.loc[outliers, "Absorbance"] = np.nan

        # Interpolation
        df["Absorbance"] = df["Absorbance"].interpolate(
            method=self.interpolation_method, limit_direction='both'
        )

        return df

    # ------------- Private: Savitzky-Golay smoothing ------------- #
    def _smooth_spectrum(self, df):
        """
        Apply Savitzky-Golay filter to Absorbance.
        """
        absorbance = df["Absorbance"].values
        n_points = len(absorbance)

        # Ensure the window length is odd and not larger than the data length
        wl = self.window_length if self.window_length % 2 == 1 else self.window_length + 1
        wl = min(wl, n_points - (n_points + 1) % 2)

        if wl <= self.polyorder:
            wl = max(self.polyorder + 2, 3)  # at least bigger than polyorder
            if wl > n_points:
                # data too short, just return
                return df

        try:
            smoothed = savgol_filter(absorbance, window_length=wl, polyorder=self.polyorder)
            df["Absorbance"] = smoothed
        except ValueError as e:
            print(f"[Warning] Smoothing failed: {e}. Using original data.")

        return df
