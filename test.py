# just a file to set up some manual runners for testing purposes.

from generators.generator import Memberships, Dataset
from generators.attendance import AttendanceDataset
import sys

# # for testing only. This always overwrites the memberships, which is generally right for testing but not for proper use.
if __name__ == '__main__':
    output_dir = "output/test"
    # create and save memberships CSV
    m = Memberships("mset_test", config_dir='config', output_dir=output_dir)
    m.generate_ids()
#
#     # # create and save a dataset. Could have several here...
#     # d = Dataset("ATTENDANCE", "test1")
#     # d.apply_models()
#     # d.to_csv()

# if __name__ == '__main__':
#     if len(sys.argv) > 1:
#         dataset_code = sys.argv[1]
#     else:
#         dataset_code = "test1"
#
#     output_dir = "output/test"
#     # create and save memberships CSV
#     m = Memberships("mset_test", config_dir='config', output_dir=output_dir)
#     m.generate_ids()
#
#     d = AttendanceDataset("ATTENDANCE", dataset_code, config_dir='config', output_dir=output_dir)
#     d.apply_models()
#     d.to_csv()

