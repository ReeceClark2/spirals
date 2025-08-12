import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from validate import Val
import astropy.units as u
import re

import matplotlib.pyplot as plt

class RadioTrackingSpectrum:
    def __init__(self, file_path: str, ifnum, plnum, including_frequency_ranges, excluding_frequency_ranges, including_time_ranges, excluding_time_ranges):
        self.filepath = file_path

        with fits.open(self.filepath) as hdul:
            self.header = hdul[0].header
            self.data = Table(hdul[1].data)

            # Find total number of feeds and channels
            ifnums = np.unique(self.data['IFNUM'])
            plnums = np.unique(self.data['PLNUM'])

            # Find total number of channels
            channel_count = len(ifnums) * len(plnums)

            self.data = self.data[
                (self.data['IFNUM'] == ifnum) &
                (self.data['PLNUM'] == plnum)
            ]
            
            self.data_start_index = (int(self.header.get("DATAIND") / channel_count)) if self.header.get("DATAIND") is not None else None
            self.post_calibration_start_index = (int(self.header.get("POSTIND") / channel_count)) if self.header.get("POSTIND") is not None else None
            self.off_start_index = (int(self.header.get("ONOFFIND") / channel_count)) if self.header.get("ONOFFIND") is not None else None

        self.ifnum = ifnum
        self.plnum = plnum

        # Accept frequency ranges
        self.including_frequency_ranges = including_frequency_ranges
        self.excluding_frequency_ranges = excluding_frequency_ranges

        # Accept time ranges
        self.including_time_ranges = including_time_ranges
        self.excluding_time_ranges = excluding_time_ranges

        # Initialize the final spectrum product
        self.spectrum = None

    def parse_history(self):
        # Find all instances of the HISTORY card in the FITS header
        entries = self.header.get('HISTORY', [])
        if isinstance(entries, str):
            entries = [entries]

        parsed = {}
        extra_lines = []

        # Parse each HISTORY card
        for entry in entries:
            # Remove inline comments
            clean_entry = entry.split('/')[0].strip()  

            # Definte the parsing strategy
            match = re.match(r'^\s*([A-Za-z0-9_,]+(?: [A-Za-z0-9_,]+)*)\s+(.*)', clean_entry)
            if match:
                key = match.group(1).strip()
                val_str = match.group(2).strip()

                # Handle underscore-separated numeric values like "1355_1435"
                if re.fullmatch(r'\d+_\d+', val_str):
                    a, b = val_str.split('_')
                    parsed[key] = (float(a), float(b))
                    continue

                # Handle comma/space-separated numeric values
                parts = val_str.replace(',', ' ').split()
                try:
                    if all(re.fullmatch(r'-?\d+(\.\d+)?', p) for p in parts):
                        vals = [float(p) for p in parts]
                        parsed[key] = vals if len(vals) > 1 else vals[0]
                    else:
                        parsed[key] = val_str
                except ValueError:
                    parsed[key] = val_str
            elif clean_entry:
                extra_lines.append(entry.strip())

        if extra_lines:
            parsed["_extra"] = extra_lines

        return parsed

    def get_frequency_range(self):
        # Use the parse history function to get dictionary of sub FITS header
        history = self.parse_history()

        # Determine whether the file is LOW or HIGH resolution
        datamode = history.get('DATAMODE')

        # Find the valid channel range within the file
        low_channel = int(history.get('START,STOP channels')[0])
        high_channel = int(history.get('START,STOP channels')[1])

        # Find the number of channels used across the file
        channel_count = high_channel - low_channel + 1

        if datamode == 'HIRES':
            # For HIGH resolution files, retrieve the proper band center from the history field
            band_center = history.get('HIRES bands')[self.ifnum]
            # Process to find the band width
            band_width = self.header['OBSBW']

            # Find the lowest and highest frequencies of the file
            low_frequency = band_center - (band_width / 2)
            high_frequency = band_center + (band_width / 2)

            return low_frequency, high_frequency, channel_count

        elif datamode == 'LOWRES':
            # For LOW resolution files, retrieve the proper band center from the FITS header
            band_center = self.header['OBSFREQ']
            # Process to find the band width
            band_width = self.header['OBSBW']

            # Pull the low and high frequencies from the provided header
            low_frequency = history.get('RFFILTER')[0] # band_center - (band_width / 2)
            high_frequency = history.get('RFFILTER')[1] # band_center + (band_width / 2)
        
            return low_frequency, high_frequency, channel_count
        
        else:
            # If the file is not LOW or HIGH resolution then raise an exception
            # TODO add graceful error handling
            raise ValueError(f"Unknown datamode: {datamode}")

    def filter_time_ranges(self):
        # Create time array to be filtered
        t0 = Time(self.header["DATE"], format="isot")
        dt = Time(self.data["DATE-OBS"], format="isot") - t0
        times = dt.to_value(u.s)

        # If there are ranges to include then filter all times not within those ranges
        if self.including_time_ranges:
            include_mask = np.zeros(len(times), dtype=bool)

            # Use a mask to get all valid indices
            for start_time, end_time in self.including_time_ranges:
                include_mask |= (start_time < times) & (times < end_time)

            # Apply the mask
            self.data = self.data[include_mask]

        # If there are ranges to exclude then filter all times within those ranges
        if self.excluding_time_ranges:
            exclude_mask = np.ones(len(times), dtype=bool)

            # Use a mask to get all valid indices
            for start_time, end_time in self.excluding_time_ranges:
                exclude_mask &= ~((start_time < times) & (times < end_time))

            # Apply the mask
            self.data = self.data[exclude_mask]

    def filter_frequency_ranges(self):
        # Get the necessary frequency data
        low_frequency, high_frequency, n_channels = self.get_frequency_range()
        # Create an array of frequencies from the highest frequency to the lowest frequency of length of total channels
        # +1 included so no channel gets cropped
        self.frequencies = np.linspace(high_frequency, low_frequency, n_channels)

        # If there are ranges to include then filter all frequencies not within those ranges
        if self.including_frequency_ranges:
            include_freq_mask = np.zeros(len(self.frequencies), dtype=bool)

            # Use a mask to get all valid indices
            for fmin, fmax in self.including_frequency_ranges:
                low, high = sorted((fmin, fmax))
                include_freq_mask |= (self.frequencies > low) & (self.frequencies < high)

            # Update the frequencies and data to reflect the mask
            self.frequencies = self.frequencies[include_freq_mask]
            self.data['DATA'] = [row[include_freq_mask] for row in self.data['DATA']]

        if self.excluding_frequency_ranges:
            exclude_freq_mask = np.ones(len(self.frequencies), dtype=bool)

            # Use a mask to get all valid indices
            for fmin, fmax in self.excluding_frequency_ranges:
                low, high = sorted((fmin, fmax))
                exclude_freq_mask &= ~((self.frequencies > low) & (self.frequencies < high))

            # Update the frequencies and data to reflect the mask
            self.frequencies = self.frequencies[exclude_freq_mask]
            self.data['DATA'] = [row[exclude_freq_mask] for row in self.data['DATA']]

    def get_spectrum(self, data):
        # Get intensities and shape of provided Astropy data table snippet
        intensities = np.array(data['DATA']) 
        count = intensities.shape[1]
        channel_means = np.sum(intensities, axis=0) / count
        
        # Return the spectrum using the predefined frequencies and intensities
        return (self.frequencies, channel_means)

    def create_spectrum(self):
        # Begin by filtering any existing ranges
        self.filter_time_ranges()
        self.filter_frequency_ranges()

        # Form the spectrum
        idx = np.arange(len(self.data))

        # ON mask: keep everything except [off_start_index : post_calibration_start_index)
        on_mask = ~((idx >= self.off_start_index) & (idx < self.post_calibration_start_index))

        # OFF mask: keep everything except [data_start_index : off_start_index)
        off_mask = ~((idx >= self.data_start_index) & (idx < self.off_start_index))

        self.on_spectrum  = self.get_spectrum(self.data[on_mask])
        self.off_spectrum = self.get_spectrum(self.data[off_mask])
        
        intensities = (self.on_spectrum[1] - self.off_spectrum[1]) / self.off_spectrum[1]
        self.spectrum = [self.frequencies, intensities]
        

if __name__ == "__main__":
    file = Val("C:/Users/starb/Downloads/0138370.fits")
    file.validate()
    
    file = RadioTrackingSpectrum("C:/Users/starb/Downloads/0138370_validate.fits", 0, 0, [(1421.25, 1423.25)], None, None, None)
    file.create_spectrum()

    plt.plot(file.spectrum[0], file.spectrum[1])
    plt.show()