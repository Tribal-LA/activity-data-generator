# Specialised generator for attendance statements
# Conform to jisc profile v1.0, with column names defined in data pipeline input/sideload_spec.py

# =============================================================================
# Activity Data Generator
# Copyright (C) 2018, Tribal Group
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

import pandas as pd
import numpy as np
from generator import Dataset

CAT = "ATTENDANCE_CATEGORY"
ATT = "EVENT_ATTENDED"
LATE = "ATTENDANCE_LATE"
VALID_CAT_CODES = ['A', 'L', 'P', '', np.nan]
LATE_CAT_CODES = {"1": ['L'],
                  "0": ['A', 'P']}
PRESENT_CAT_CODES = {"1": ['P', 'L'],
                     "0": ['A']}

class AttendanceDataset(Dataset):
    """
    Extend the base Dataset generators with busines rules for Attendance data
    """

    def __init__(self, statement_family, dataset_code,  output_dir, config_dir):
        Dataset.__init__(self, statement_family, dataset_code,  output_dir, config_dir)

    def apply_models(self):
        # core model-driven generation
        Dataset.apply_models(self)
        #super(AttendanceDataset, self).apply_models()

        for ay in self.academic_years:
            df = self.ay_df[ay]
            # checks on things we rely on below
            if sum(~df[CAT].isin(VALID_CAT_CODES)) > 0:
                raise Exception("{} codes not in expected list: {}".format(CAT, VALID_CAT_CODES))

            #  specialised business rules

            # A. additional logic for code/attended/late, which tolerates the model spec giving different ones. Note precedence order
            # if ATTENDANCE_CATEGORY is known, it trumps _ATTENDED and _LATE. NB logic here allows for missing attendance category
            for att, codes in PRESENT_CAT_CODES.iteritems():
                df.loc[df[CAT].isin(codes), ATT] = att
            for late, codes in LATE_CAT_CODES.iteritems():
                df.loc[df[CAT].isin(codes), LATE] = late

            # B. the timestamp gives the start time + add the duration, then clean up "__" temp col
            df["START_TIME"] = df.timestamp
            df["END_TIME"] = pd.to_datetime(df.timestamp) + pd.to_timedelta(df["__duration_mins"].astype(int), unit='m')
            df.drop(['__duration_mins'], axis=1, inplace=True)
