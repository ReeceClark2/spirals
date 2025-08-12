import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from scipy.stats import linregress
from validate import Val
import re
import rcr

class RadioTrackingContinuum:
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

        # Initialize the final continuum product
        self.continuum = None

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

    def linear(self, x, params): # model function
        return params[0] + x * params[1]

    def d_linear_1(self, x, params): # first model parameter derivative
        return 1

    def d_linear_2(self, x, params): # second model parameter derivative
        return x

    def perform_rcr(self, array):
        # Perform functional Robust Chauvenet Rejection (RCR) on a given dataset.
        
        # Parse 2D array into x and y arrays
        x = array[0]
        x_centered = x - np.average(x)

        y = array[1]

        # Using linear regression, create a guess function for RCR
        result = linregress(x_centered, y)
        m = result.slope
        b = result.intercept
        guess = [m, b]

        # Perform RCR
        model = rcr.FunctionalForm(self.linear,
            x_centered,
            y,
            [self.d_linear_1, self.d_linear_2],
            guess
        )
        
        r = rcr.RCR(rcr.SS_MEDIAN_DL) 
        r.setParametricModel(model)
        r.performBulkRejection(y)

        # Fetch indices
        indices = r.result.indices

        # Keep on valid indices
        x = np.array([x[i] for i in indices])
        y = np.array([y[i] for i in indices])

        best_fit_parameters = model.result.parameters
        
        # Using the best fit parameters, calculate a slope and y-intercept uncertainty
        sigma = (1 / (len(x) - 2)) * np.sum((y - best_fit_parameters[1] * x - best_fit_parameters[0]) ** 2)
        m_sd = np.sqrt(sigma / np.sum((x - np.mean(x)) ** 2))
        b_sd = np.sqrt(sigma * ((1 / len(x)) + ((np.mean(x) ** 2) / np.sum((x - np.mean(x)) ** 2))))
        uncertainties = (b_sd, m_sd)

        return best_fit_parameters, uncertainties

    def get_continuum(self, data):
        # Get intensities and shape of provided Astropy data table snippet
        intensities = np.array(data['DATA']) 
        count = intensities.shape[1]
        channel_means = np.sum(intensities, axis=1) / count

        # Use the headers start time to 0 the time array of the observation
        times = Time(data["DATE-OBS"], format='isot')
        t0 = Time(self.header["DATE"], format="isot")
        time_rel = (times - t0)

        return (time_rel.sec, channel_means)

    def get_cal_arrays(self):
        # Filter out the pre cal on and off arrays
        try:
            pre_cal_on_mask = self.data[:self.data_start_index][
                (self.data['CALSTATE'][:self.data_start_index] == 1) &
                (self.data['SWPVALID'][:self.data_start_index] == 0) 
            ]

            pre_cal_on_array = self.get_continuum(pre_cal_on_mask)
        except Exception:
            pre_cal_on_array = None

        try:
            pre_cal_off_mask = self.data[:self.data_start_index][
                (self.data['CALSTATE'][:self.data_start_index] == 0) &
                (self.data['SWPVALID'][:self.data_start_index] == 0) 
            ]

            pre_cal_off_array = self.get_continuum(pre_cal_off_mask)
        except Exception:
            pre_cal_off_array = None

        # Filter out the post cal on and off arrays
        try:
            post_cal_on_mask = self.data[self.post_calibration_start_index:][
                (self.data['CALSTATE'][self.post_calibration_start_index:] == 1) &
                (self.data['SWPVALID'][self.post_calibration_start_index:] == 0) 
            ]

            post_cal_on_array = self.get_continuum(post_cal_on_mask)
        except Exception:
            post_cal_on_array = None

        try:
            post_cal_off_mask = self.data[self.post_calibration_start_index:][
                (self.data['CALSTATE'][self.post_calibration_start_index:] == 0) &
                (self.data['SWPVALID'][self.post_calibration_start_index:] == 0) 
            ]

            post_cal_off_array = self.get_continuum(post_cal_off_mask)
        except Exception:
            post_cal_off_array = None

        # Return all arrays to be iterated over
        return [[pre_cal_on_array, pre_cal_off_array], [post_cal_on_array, post_cal_off_array]]

    def gain_calibrate(self):
        # Get the start time and array of times through the channel
        t0 = Time(self.header["DATE"], format="isot")
        time_array = (Time(self.data['DATE-OBS'], format='isot') - t0).sec

        # Get the calibration arrays
        cal_arrays = self.get_cal_arrays()

        for ind1, i in enumerate(cal_arrays):
            # For pre and post cal in cal_arrays try to compute the delta
            try:
                # i[0] is the on array for a calibration spike while i[1] is the off array. i[0][0] are the times of the on array and i[0][1] are the intensities of the on array
                if i[0][0].size != 0 and i[0][1].size != 0 and i[1][0].size != 0 and i[1][1].size != 0:
                    cal_on_params, on_uncertainty = self.perform_rcr(i[0])
                    cal_off_params, off_uncertainty = self.perform_rcr(i[1])

                    # Get the calibration average time between the on and off time arrays
                    cal_time = np.mean([np.mean(i[0][0]), np.mean(i[1][0])])

                    off_array_times = i[1][0]
                    largest_time_difference = -np.inf
                    split_index = None

                    # Find a split in time in off segments
                    average_time_difference = np.mean(np.diff(off_array_times))
                    for ind, j in enumerate(off_array_times):
                        if ind != 0:
                            time_difference = j - off_array_times[ind - 1]
                            if time_difference > largest_time_difference:
                                largest_time_difference = time_difference
                                split_index = ind

                    # Check if some difference contains likely points in between, locate on array in calibration
                    if largest_time_difference > 3 * average_time_difference:
                        off_times_prior = i[1][0][:split_index]
                        off_times_later = i[1][0][split_index:]

                        # If several off arrays exist then fix the time to the calibration spike edges
                        cal_time = np.clip(
                            cal_time,
                            (i[0][0][0] + off_times_prior[-1]) / 2,
                            (i[0][0][-1] + off_times_later[0]) / 2
                        )

                    # Calculate the calibration delta
                    cal_delta = np.abs(
                        (cal_on_params[1] * (cal_time - np.mean(i[0][0])) + cal_on_params[0]) -
                        (cal_off_params[1] * (cal_time - np.mean(i[1][0])) + cal_off_params[0])
                    )
                    # Calculate the calibration uncertainty
                    cal_uncertainty = np.sqrt(
                        on_uncertainty[0]**2 + off_uncertainty[0]**2 +
                        (on_uncertainty[1] * cal_time)**2 + (off_uncertainty[1] * cal_time)**2
                    )
                else:
                    cal_delta = None
                    cal_uncertainty = None
            # Catch any exceptions in calibration spike identification and assign None values
            except:
                cal_delta = None
                cal_uncertainty = None

            # If pre calibration then assign pre calibration values
            if ind1 == 0:
                pre_cal_delta = cal_delta
                pre_cal_uncertainty = cal_uncertainty

            # If post calibration then assign post calibration values 
            elif ind1 == 1:
                post_cal_delta = cal_delta
                post_cal_uncertainty = cal_uncertainty

        # If both calibrations exist then use both calibrations
        if pre_cal_delta and post_cal_delta:
            # Calculate the convoluted z value for both deltas
            z_value = abs(pre_cal_delta - post_cal_delta) / np.sqrt(
                pre_cal_uncertainty ** 2 + post_cal_uncertainty ** 2
            )
            
            # If the z value is not significant then average gain deltas
            if z_value < 0.6745:
                weights = np.array([1 / pre_cal_uncertainty, 1 / post_cal_uncertainty])
                delta = np.average([pre_cal_delta, post_cal_delta], weights=weights)
                self.data['DATA'] /= delta
            else:
            # If the z value is significant then interpolate between deltas for each time
                for ind, _ in enumerate(self.data['DATA']):
                    t = time_array[ind]
                    delta = pre_cal_delta + (post_cal_delta - pre_cal_delta) * (t - t0)
                    self.data['DATA'][ind] /= delta

        # If only pre calibration exist and not post calibration
        elif pre_cal_delta and not post_cal_delta:
            # Not enough data to interpolate
            self.data['DATA'] /= pre_cal_delta
        
        # If only post calibration exist and not pre calibration
        elif post_cal_delta and not pre_cal_delta:
            # Not enough data to interpolate
            self.data['DATA'] /= post_cal_delta

        self.data['DATA'] = self.data['DATA']

    def get_conversion_factor(self):
        # TODO add logic to get conversion factor
        # TODO maybe add future logic to interpolate between calibration frames
        # TODO on the display page, have a banner like "new calibration data available"

        conversion_factor = 3
        return conversion_factor

    def flux_calibrate(self):
        # Get the conversion factor
        conversion_factor = self.get_conversion_factor()

        # Refactor continuum with applied conversion factor
        x, y = self.continuum
        y /= conversion_factor 
        self.continuum = (x, y)

    def create_continuum(self):
        # Begin by filtering any existing ranges
        self.filter_time_ranges()
        self.filter_frequency_ranges()
        
        # Perform gain calibration
        self.gain_calibrate()

        # Using the gain calibrated data cube, form the continuum
        self.continuum = self.get_continuum(self.data)

        # Flux calibrate the now existing continuum (not existing currently)
        # self.flux_calibrate()

if __name__ == "__main__":
    file = Val("filepath")
    file.validate()

    file = RadioTrackingContinuum("filepath_validated", 0, 0, None, None, None, None)
    file.create_continuum()
