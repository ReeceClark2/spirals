# Third-party libraries
import os

import numpy as np
from astropy.io import fits
from astropy.table import Column
from astropy.table import Table
from astropy.time import Time


class Val:
    def __init__(self, file_path: str):
        self.filepath = file_path
        self.missing_values = []

        with fits.open(self.filepath) as hdul:
            self.header = hdul[0].header
            self.data = Table(hdul[1].data)

        self.data_start_index = None
        self.post_calibration_start_index = None
        self.off_start_index = None

    def validate_header_size(self):
        # Validate header's 2880 byte standard
        header_size = len(self.header.tostring())
        if 0 != header_size % 2880:
            raise ValueError(f"Header size {header_size} is not a multiple of 2880 bytes.")

    def validate_primary_header_cards(self):
        # Validate SIMPLE card is the first in the primary header
        if self.header.cards[0].keyword != "SIMPLE":
            raise ValueError("SIMPLE card is not the first card in the primary header.")

        # Ensure BITPIX and NAXIS exist (may need to include other required cards later)
        required_keys = ['BITPIX', 'NAXIS']
        for key in required_keys:
            if key not in self.header:
                raise ValueError(f"Required header card '{key}' is missing.")

        # Ensure BITPIX conforms to standard values
        if self.header.get('BITPIX') not in [8, 16, 32, 64, -32, -64]:
            raise ValueError(f"BITPIX value {self.header.get('BITPIX')} is not a standard value.")

        # Ensure all expected axes exist
        naxis = self.header.get('NAXIS')
        for i in range(1, naxis + 1):
            axis_key = f"NAXIS{i}"
            if axis_key not in self.header:
                raise ValueError(f"Required header card '{axis_key}' is missing.")
            if self.header[axis_key] < 0:
                raise ValueError(f"Value for '{axis_key}' cannot be negative: {self.header[axis_key]}.")

        # Ensure last card is the END card
        with open(self.filepath, 'rb') as f:
            block_size = 2880
            header_bytes = b""
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
                header_bytes += chunk
                if b'END' in chunk:
                    break

            if b'END' not in header_bytes:
                raise ValueError("Header does not end with 'END' card.")

    def validate_header_cards(self):
        # Ensure there are no duplicate cards (except for COMMENT and HISTORY cards)
        seen = set()
        for card in self.header.cards:
            key, value, comment = card
            if key in seen and key not in ('COMMENT', 'HISTORY'):
                raise ValueError(f"Duplicate header card found: {key}.")

            seen.add(key)

        # Find if any cards have missing values
        for card in self.header.cards:
            key, value, comment = card

            card_str = str(card)
            card_length = len(card_str)

            if card_length != 80:
                raise ValueError(f"Header card '{key}' does not conform to 80 character length: {card_str}")

            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"Header card '{key}' has a missing value: {card_str}")

    def validate_primary_header(self):
        # Run all necessary functions to validate the header extension
        self.validate_header_size()
        self.validate_primary_header_cards()
        self.validate_header_cards()

    def check_values(self):
        # Initialize data cube in SDFITS file
        data_array = np.ma.array(self.data['DATA'])

        # Ensure no NaN values in data cube
        mask = np.isnan(data_array)
        if np.any(mask):
            # TODO add logging to warn nan values
            # Mask values if NaN values exist
            self.data['DATA'] = np.ma.masked_array(data_array, mask=mask)

    def match_types(self, column):
        # Initialize the array for the given column
        column_data_array = self.data[column]

        unique_types = set(type(value) for value in column_data_array)

        # If more than one type exist within the given column then force the column to conform to the most frequent type
        if len(unique_types) > 1:
            common_type = max(unique_types, key=lambda t: sum(isinstance(value, t) for value in column_data_array))

            for ind, i in enumerate(column_data_array):
                if not isinstance(i, common_type):
                    try:
                        column_data_array[ind] = common_type(i)
                    except:
                        # If the data point in the data array can not be assigned the common type then mask it
                        if not hasattr(column_data_array, 'mask'):
                            column_data_array = np.ma.array(column_data_array)
                        column_data_array.mask[ind] = True

            # TODO add logging to state "column has mixed data types in X field"

        # If only one type exists then ensure all string and numpy array columns are valid
        else:
            common_type = next(iter(unique_types))
            try:
                if common_type is str:
                    if column_data_array.dtype.char == 'S':
                        column_data_array = np.char.decode(column_data_array, encoding='utf-8', errors='replace')
                    else:
                        column_data_array = np.array(column_data_array[column], dtype=str)

                elif common_type is np.ndarray:
                    if not all(isinstance(float(item), float) for subarray in column_data_array for item in subarray):
                        for subarray in column_data_array:
                            for ind, i in enumerate(subarray):
                                try:
                                    float(i)
                                except:
                                    raise ValueError(f"Cannot convert {i} to float in column {column}.")

                else:
                    self.data[column] = self.data[column].astype(common_type)

            except:
                raise ValueError("Failed to convert column data to common type.")

    def convert_to_datetime(self, column):
        # Initialize the array for the given column
        column_data_array = self.data[column]

        # Ensure DATE and TIME columns are valid timestamps
        if any(keyword in column.upper() for keyword in ["DATE", "TIME"]) or column.upper() == "LST":
            try:
                # Try converting with astropy Time
                type = column_data_array.dtype.type
                if type is str or type is np.str_:
                    for i, value in enumerate(column_data_array):
                        # Replace underscores with dashes and the first colon with 'T' for ISO format
                        new_value = value.replace('_', '-')

                        if 'T' not in new_value:
                            new_value = new_value.replace(':', 'T', 1)

                        self.data[column][i] = new_value
                self.data[column] = Time(column_data_array)

            except (ValueError, TypeError):
                try:
                    # Try to convert dates to float if date object fails
                    self.data[column] = Column(column_data_array.astype(float), dtype='f8')
                except ValueError:
                    try:
                        # Further try a string type
                        self.data[column] = Column(column_data_array.astype(str), dtype='U')
                    except:
                        raise ValueError("Cannot convert datetime")

        # Ensure that certain time columns can be written as string or float
        if any(keyword in column.upper() for keyword in
               ['DURATION', 'EXPOSURE', 'MJD', 'UTC', 'UTSECS', 'DATE-OBS']) or column.upper() == 'LST':
            try:
                # Try to convert dates to float if date object fails
                self.data[column] = Column(column_data_array.astype(float), dtype='f8')
            except ValueError:
                try:
                    # Further try a string type
                    self.data[column] = Column(column_data_array.astype(str), dtype='U')
                except:
                    raise ValueError(f"Cannot convert time column {column} to float or string.")

    def check_numbers(self, column):
        # Initialize the array for the given column
        column_data_array = self.data[column]

        if isinstance(self.data[column], Time):
            return

        if column_data_array.dtype.kind in {'U', 'S', 'O'}:
            def replace_nan(val):
                if isinstance(val, str) and val.strip().lower() == 'nan':
                    return np.nan

                return val

            self.data[column] = np.vectorize(replace_nan)(column_data_array)

        # Ensure that columns that should not have negative values do not have negative values
        if column.upper() in ["DURATION", "EXPOSURE", "TSYS", "TCAL", "LST", "ELEVATION", "TAMBIENT", "PRESSURE",
                              "HUMIDITY", "RESTFREQ", "FREQRES", "TRGTLONG", "MJD", "UTSECS"]:
            if np.any(column_data_array < 0):
                mask = column_data_array >= 0
                self.data = self.data[mask]

                # TODO add logging for time array less than zero

    def get_channels(self):
        for key, value in self.header.items():
            if key == ("HISTORY"):
                if value.startswith("START,STOP"):
                    # Extract all integers from the string
                    value = value.replace(",", " ").strip()
                    value = value.split(" ")
                    # Split the value into individual words and numbers

                    channels = []
                    for k in value:
                        k = str(k).strip()
                        try:
                            # If it's an integer add it to the channels list
                            k = int(k)
                            channels.append(k)
                        except ValueError:
                            # If it can't become a integer, skip it
                            continue
                    
                    # Remove all string type characters from the channels list
                    start_channel = int(channels[0])
                    stop_channel = int(channels[1])
        
        self.data['DATA'] = [row[start_channel:stop_channel + 1] for row in self.data['DATA']]

    def validate_data(self):
        # Run all necessary functions to validate the data extension
        self.check_values()

        # Iterate through each column in the data extension for validation
        for column in self.data.colnames:
            self.match_types(column)
            self.convert_to_datetime(column)
            self.check_numbers(column)

    def find_calibrations(self):
        # Find total number of feeds and channels
        ifnums = np.unique(self.data['IFNUM'])
        plnums = np.unique(self.data['PLNUM'])

        # Find total number of channels
        channel_count = len(ifnums) * len(plnums)

        # Initialize necessary indices
        data_start_ind = None
        post_cal_start_ind = None

        # Create a counter for valid data
        counter = 0

        # Initialize confirmation Booleans
        cal_started = False
        pre_cal_complete = False

        for ind, i in enumerate(self.data):
            # If the CALSTATE is equal to 1 then state that calibration has started
            if i['CALSTATE'] == 1:
                cal_started = True

            # If data begins being collected (CALSTATE = 0 and SWPVALID = 1) then state that pre calibration has started and the first data index can be recorded
            if cal_started and i["CALSTATE"] == 0 and i["SWPVALID"] == 1 and not pre_cal_complete:
                data_start_ind = ind
                pre_cal_complete = True

            # If the pre calibration is complete and the sweep is no longer valid then keep track of this as the post calibration start
            if ind > 0 and pre_cal_complete and i["SWPVALID"] == 0 and self.data[ind - 1]["SWPVALID"] == 0:
                if post_cal_start_ind is None:
                    post_cal_start_ind = ind - 1
            # Reset the post cal index to None if the above condition is False. 
            # This allows for sweeps to be invalid within the observation such as during on/off transition or invalid data blips.
            else:
                post_cal_start_ind = None

            # Keep track of contiguous data points
            if pre_cal_complete and i['CALSTATE'] == 0 and i['SWPVALID'] == 1:
                counter += 1

            # If 3 or less valid data points across all channels have been collected and the sweep becomes 
            # invalid then treat this section of data as invalid and reset necessary params.
            if counter <= 3 * channel_count and i['SWPVALID'] == 0 and data_start_ind:
                data_start_ind = None
                pre_cal_complete = False

            # When pre calibration is complete and a new cal spike begins, 
            # break the loop (post cal index will already be recorded if available)
            if pre_cal_complete and i['SWPVALID'] == 0 and i['CALSTATE'] == 1:
                break

        # Set indices
        self.data_start_index = data_start_ind

        if not pre_cal_complete:
            data_start_ind = 0

            for ind, i in enumerate(self.data):
                # If the pre calibration is complete and the sweep is no longer valid then keep track of this as the post calibration start
                if ind > 0 and pre_cal_complete and i["SWPVALID"] == 0 and self.data[ind - 1]["SWPVALID"] == 0:
                    if post_cal_start_ind is None:
                        post_cal_start_ind = ind - 1

                # Reset the post cal index to None if the above condition is False. 
                # This allows for sweeps to be invalid within the observation such as during on/off transition or invalid data blips.
                else:
                    post_cal_start_ind = None

                # When pre calibration is complete and a new cal spike begins, 
                # break the loop (post cal index will already be recorded if available)
                if i['SWPVALID'] == 0 and i['CALSTATE'] == 1:
                    break

        if not post_cal_start_ind:
            post_cal_start_ind = len(self.data) - 1

        self.post_calibration_start_index = post_cal_start_ind

        # If the file is an on/off file then find when the transition occurs and store an additional index
        if self.header['OBSMODE'] == 'onoff':
            for ind, i in enumerate(self.data):
                target = 'onoff:off'

                if target in i['OBSMODE']:
                    offstart = ind
                    self.off_start_index = offstart

                    break

        # Write the indices to the SDFITS header extension
        self.header['DATAIND'] = (self.data_start_index, 'Index of start of data')
        self.header['POSTIND'] = (self.post_calibration_start_index, 'Index of start of post-calibration data')
        self.header['ONOFFIND'] = (getattr(self, 'off_start_index', None), 'Index of start of OFF section in on/off mode')

    def save(self, output_path=None):
        if output_path is None:
            base, ext = os.path.splitext(self.filepath)
            output_path = f"{base}_validate{ext}"

        # Create Primary HDU with updated header
        primary_hdu = fits.PrimaryHDU(header=self.header)

        # Convert Table back to FITS HDU
        table_hdu = fits.BinTableHDU(data=self.data)

        # Combine HDUs
        hdulist = fits.HDUList([primary_hdu, table_hdu])

        # Write to file
        hdulist.writeto(output_path, overwrite=True)

    def validate(self):
        # Validate the primary header
        self.validate_primary_header()

        # Validate the data
        self.validate_data()

        # Find the calibration indices and get rid of poor channels
        self.find_calibrations()
        self.get_channels()

        # Save the new validated file under the original filepath + _validated
        self.save()


if __name__ == "__main__":
    filepath = "C:/Users/starb/Downloads/0116250.fits"
    
    v = Val(filepath)
    v.validate()
    
    filepath = "C:/Users/starb/Downloads/0117689.fits"

    v = Val(filepath)
    v.validate()
