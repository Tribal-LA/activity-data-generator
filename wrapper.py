# Simple wrapper to Membership &/or Dataset for command line use
# intended main use scenario = to generate membership csvs and several datasets which rely on them.
# This is driven by a JSON file. See configuration.md

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

from generators.generator import Memberships, Dataset

from os import path
import json


class Wrapper:
    def __init__(self, wrapper_code, output_dir='output', config_dir='config'):
        """

        :param wrapper_code: name of JSON wrapper configuration file without extension to look for in config_dir
        :type wrapper_code: str
        :param config_dir: location for the wrapper config files. May be absolute or relative to CWD. Defaults to 'config'
        :type config_dir: str
        :param output_dir: location for the output files files. May be absolute or relative to CWD. Defaults to 'output'
        :type output_dir: str
        """
        self.wrapper_code = wrapper_code
        self.config_dir = config_dir
        self.output_dir = output_dir

        config_file = path.join(self.config_dir, "{}.json".format(wrapper_code))
        if not path.exists(config_dir):
            raise Exception("Failed to find wrapper config file: {}.".format(config_file))

        with open(config_file, 'r') as f:
           self.config = json.load(f)

    def generate(self):
        # the wrapper code distinguishes output as a subdir
        output_dir = path.join(self.output_dir, self.wrapper_code)
        # generate memberships
        m_def = self.config["membership_def"]
        m = Memberships(m_def,
                        output_dir=output_dir,
                        config_dir=self.config_dir)
        m.generate_ids()

        # loop over datasets, using either a custom generators or the default generators
        for batch in self.config["generate_datasets"]:
            statement_family = batch["statement_family"]
            if 'custom_generator' in batch:
                # instantiate a custom generator
                (module_name, class_name) = batch["custom_generator"].split('.')
                module = __import__('generators.'+module_name, fromlist=[None])
                class_ = getattr(module, class_name)
                for dataset_code in batch["dataset_defs"]:
                    g = class_(statement_family, dataset_code, output_dir, self.config_dir)
                    g.apply_models()
                    g.to_csv()
            else:
                # generic generator
                for dataset_code in batch["dataset_defs"]:
                    g = Dataset(statement_family, dataset_code, output_dir, self.config_dir)
                    g.apply_models()
                    g.to_csv()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        wrapper_config = sys.argv[1]
        w = Wrapper(wrapper_config)  # config path defaults to 'config'
        w.generate()
    else:
        print "Supply an argument which identifies a JSON wrapper configuration file (file name without extension)"
