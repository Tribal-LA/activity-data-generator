# -------------------------------------------------------------------------------------------------
# This is the core generic generator. Create derived classes to implement custom business rules.
# The classes here should remain multi-purpose.
# -------------------------------------------------------------------------------------------------

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

from os import path
import os
import json
from zipfile import ZipFile
from numpy.random import random, choice, randint, normal, uniform
import numpy as np
import pandas as pd
import datetime
# from pipeline
from utilities import AcademicPeriod

# these are subdirectory names
MODEL_DEF_DIR = 'model_definitions'
DATASET_DEF_DIR = 'dataset_definitions'
MEMBERSHIP_DEF_DIR = 'membership_definitions'

def _make_membership_file_path(output_dir, membership_def, ay):
    dir = output_dir  # path.join(output_dir, membership_def)
    if not path.exists(dir):
        os.makedirs(dir)
    return path.join(dir, "memberships_{}_{}.csv".format(membership_def, ay))

def _make_dataset_file_path(output_dir, platform, statement_family, dataset_code, ay):
    dir = output_dir  # path.join(output_dir, membership_def)
    if not path.exists(dir):
        os.makedirs(dir)
    return path.join(dir, "{}.{}_{}_{}.csv".format(platform, statement_family, dataset_code, ay))

# keywords in model def json
MODEL_TYPE_EVENT = "event"
MODEL_TYPE_RS = "random-session"


class Models:
    """

    """
    def __init__(self, statement_family, profile_id, config_dir, week1_contains='09-01'):
        # starting week of academic years
        self._week1_contains = week1_contains

        self.config_dir = config_dir

        def_file = path.join(self.config_dir, MODEL_DEF_DIR, "{}_{}.json".format(statement_family, profile_id))
        if not path.exists(def_file):
            raise Exception("Failed to find file {}".format(def_file))
        with open(def_file, 'r') as f:
            model_def = json.load(f)

        self.data_columns = model_def["columns"]
        self.data_columns.append("timestamp")
        self.agent_id_col = model_def["agent_id_col"]
        self.group_id_col = model_def["group_id_col"]
        self.group_set_col = model_def.get("group_set_col", None)
        self.activity_id_col = model_def["activity_id_col"]
        self.n_activities = model_def["n_activities"]
        self.activity_id_pattern = model_def["activity_id_pattern"]

        # NB column order IS important; must match order in row, elsewhere
        if self.group_set_col is not None:
            self.id_columns = [self.agent_id_col, self.group_id_col, self.group_set_col, self.activity_id_col]
        else:
            [self.agent_id_col, self.group_id_col, self.activity_id_col]
        self.all_columns = self.id_columns + self.data_columns
        """columns to match the (inner) list items returned by try_generate()"""

        self.models = model_def["models"]
        self.model_type = model_def["models_type"]
        if self.model_type not in [MODEL_TYPE_EVENT, MODEL_TYPE_RS]:
            raise Exception("Invalid models_type in JSON: {}".format(self.model_type))
        # validation
        if self.model_type == MODEL_TYPE_RS:
            for m_code in self.models:
                if 'session_model' not in self.models[m_code]:
                    raise Exception("A models_type of '{}' was specified but model specification '{}' does not contain a 'session_model' definition".format(
                        MODEL_TYPE_RS, m_code))

        # renormalise intensities and day+pattern
        self.temporal = list()
        self.temporal_intensities = list()  # convenience direct access to list of intensities
        intensity_sum = 0.0
        weeks_covered = set()  # for checking weeks spec is non-overlapping
        # prep
        for t in model_def["temporal"]:
            t_weeks = set(t["weeks"])
            if not weeks_covered.isdisjoint(t_weeks):
                raise Exception("Invalid weeks specification, some week(s) appeared more than once: {}".format(
                    weeks_covered.intersection(t_weeks)))
            weeks_covered = weeks_covered.union(t_weeks)
            intensity_sum += t["intensity"]
        # renorm temporal
        for t in model_def["temporal"]:
            day_sum = sum(t["day_pattern"])
            renorm_intensity = float(t["intensity"]) / intensity_sum
            self.temporal_intensities.append(renorm_intensity)
            self.temporal.append({"note": t["note"],
                                  "weeks": t["weeks"],
                                  "intensity": renorm_intensity,
                                  "day_pattern": [float(d) / day_sum for d in t["day_pattern"]]})
        # and tod-pattern if present (which is per model)
        for m_code in self.models:
            if "tod_pattern" in self.models[m_code]:
                if self.models[m_code]["tod_pattern"]["tod_pattern_type"] is not None:
                    old_tod = self.models[m_code]["tod_pattern"]["tod"]
                    tod_fraction_sum = sum(old_tod.values())
                    self.models[m_code]["tod_pattern"]["tod"] = {k: float(v) / tod_fraction_sum for k, v in old_tod.iteritems()}
            else:
                self.models[m_code]["tod_pattern"] = {"tod_pattern_type": None}

        # renorm column option group proportions
        for m_code in self.models:
            if "option_group_prop" in self.models[m_code]:
                old_ogp = self.models[m_code]["option_group_prop"]
                option_fraction_sum = sum(old_ogp.values())
                self.models[m_code]["option_group_prop"] = {k: float(v) / option_fraction_sum for k, v in old_ogp.iteritems()}

        # some deeper renorm of categorical and column option group proportions
        def _renorm_cat(cv):
            if type(cv) is dict:
                if 'categorical' in cv:
                    old = cv["categorical"].copy()
                    fract_sum = sum(old.values())
                    cv["categorical"] = {code: float(old[code]) / fract_sum for code in old}
        for m_code in self.models:
            if "compound_model" not in self.models[m_code]:
                # plain column spec
                for c in self.models[m_code]["column_values"]:
                    _renorm_cat(self.models[m_code]["column_values"][c])
                # option-groups
                if "option_group_choice" in self.models[m_code]:
                    ogc = self.models[m_code]["option_group_choice"]
                    for og in ogc:
                        for c in ogc[og]:
                            _renorm_cat(ogc[og][c])

        # renorm compound models
        for m_code in self.models:
            if "compound_model" in self.models[m_code]:
                old = self.models[m_code]["compound_model"]
                p_sum = sum(old.values())
                self.models[m_code]["compound_model"] = {k: float(v) / p_sum for k, v in old.iteritems()}

        # for cache of academic periods
        self._ap = dict()

        # for test
        # print self.temporal_intensities
        # print self.temporal
        # print self.models

    def try_generate(self, model_code, academic_year, agent_id, group_id, group_set):
        """For simple models, generate a record subject to the 'p' value of the required model code.
        For compound models, select a simple model based on configured probabilities and then generate for that simple model.

        A value None is returned to signal no activity, otherwise a list comprising the values for self.columns,
        including the additional item which is the timestamp"""

        this_model = self.models[model_code]

        # check whether this is a compound model, and if so make a choice of the simple model to use and recurse
        if "compound_model" in this_model:
            simple_model = choice(this_model["compound_model"].keys(), p=this_model["compound_model"].values())
            return self.try_generate(simple_model, academic_year, agent_id, group_id, group_set)
        else:
            p = float(this_model["p"])

            if random() > p:  # cld speed up with check p!=1.0
                return None

            last_event, last_timestamp = self._generate_1(model_code, academic_year, agent_id, group_id, group_set)
            events = [last_event]

            # model will be either a single event generators or a session-event-set generators
            if self.model_type == MODEL_TYPE_EVENT:
                return events

            # model type is randoom session
            session_model = this_model["session_model"]
            while ((len(events) < session_model["min_length"]) or (random() < session_model["survival"])) and (len(events) < session_model["max_length"]):
                interval_model = session_model["interval"]
                delay_s = normal(interval_model["mu"], interval_model["sigma"])
                if delay_s >=0:  # avoid time travel. This means the data may not actually match the specified distribution but ...
                    next_timestamp = last_timestamp + datetime.timedelta(seconds=delay_s)
                    last_event, last_timestamp = self._generate_1(model_code, academic_year, agent_id, group_id, group_set, timestamp=next_timestamp)
                    events.append(last_event)
            return events


    def _generate_1(self, model_code, academic_year, agent_id, group_id, group_set, timestamp=None):

        # preliminaries for timestamp
        timestamp_format = '%Y-%m-%dT%H:%M:%S'  # default - may be overridden below
        time_offset = ""
        if academic_year in self._ap:
            # use cache
            ap = self._ap[academic_year]
        else:
            # populate cache
            ap = AcademicPeriod(ay=academic_year, week1_contains=self._week1_contains)
            self._ap[academic_year] = ap

        # timeetamp starts as date-only and MAY be embellished. This doies not apply within a session, whten timestamp is passed in
        if timestamp is None:
            # select one of the temporal patterns according to the intensity
            this_time = self.temporal[choice(range(0, len(self.temporal)), p=self.temporal_intensities)]
            # pick an academic week
            choose_from = this_time["weeks"]
            aw = choose_from[choice(range(0, len(choose_from)))]
            timestamp = ap.date_from_aw(aw)
            # pick a day of the week and add it on
            dow = choice(range(0, 7), p=this_time["day_pattern"])
            timestamp += datetime.timedelta(days=dow)
            # offset
            tod_pattern = self.models[model_code]["tod_pattern"]
            tod_pattern_type = tod_pattern["tod_pattern_type"]
            if tod_pattern_type is None:
                timestamp_format = '%Y-%m-%d'
            else:
                time_offset = choice(tod_pattern["tod"].keys(), p=tod_pattern["tod"].values())
                time_offset_split = [int(tos) for tos in time_offset.split(':')]
                timestamp += datetime.timedelta(hours=time_offset_split[0])
                if len(time_offset_split) > 1:
                    timestamp += datetime.timedelta(minutes=time_offset_split[1])
                if tod_pattern_type != 'choice':
                    dist = tod_pattern['distributions'][time_offset]
                    if tod_pattern_type == 'norm':
                        # normal
                        timestamp += datetime.timedelta(minutes=normal(0.0, dist))
                    elif tod_pattern_type == 'uniform':
                        #uniform
                        timestamp += datetime.timedelta(minutes=uniform(-dist, dist))
                    else:
                        raise Exception("Invalid tod_pattern_type {}. If provided, it must be one of 'choice', 'norm', or 'uniform'.".format(tod_pattern_type))

        # specified values for the model. may not cover all self.columns
        col_vals = self.models[model_code]["column_values"].copy()
        # if there are option groups, choose one of them. the chosen group will override the column_values
        if "option_group_prop" in self.models[model_code]:
            ogp = self.models[model_code]["option_group_prop"]
            chosen_group = choice(ogp.keys(), p=ogp.values())
            og = self.models[model_code]["option_group_choice"][chosen_group]
            for c in og:
                col_vals[c] = og[c]

        row = []
        for c in self.data_columns:
            if c == "timestamp":
                row.append(datetime.date.strftime(timestamp, timestamp_format))
            else:
                if c in col_vals:
                    cv = col_vals[c]
                    if type(cv) in [str, unicode]:
                        row.append(cv)
                    elif type(cv) is dict:
                        # missing value check
                        p_na = cv.get("p_na", 0.0)
                        if (p_na == 0.0) or (random() >= p_na):
                            if "categorical" in cv:
                                codes = cv["categorical"].keys()
                                code_freq = cv["categorical"].values()
                                row.append(choice(codes, p=code_freq))
                            elif "numerical" in cv:
                                distribution = cv["numerical"]["distribution"]
                                params = cv["numerical"]["params"]
                                if distribution == "normal":
                                    r_val = normal(params[0], params[1])
                                elif distribution == "uniform":
                                    r_val = uniform(params[0], params[1])
                                else:
                                    raise Exception("Invalid distribution '{}'. Only 'uniform' or 'normal' are allowed".format(distribution))
                                row.append(r_val)
                            else:
                                raise Exception("Bad model defintion (require 'categorical' or 'numerical': {}".format(cv.keys()))
                        else:
                            row.append(np.nan)  # p_na was hit!
                    else:
                        row.append(str(cv))
                else:
                    row.append(np.nan)
        format_vals = {self.group_id_col: group_id,
                       '__tod': time_offset,
                        '__n': randint(self.n_activities)}
        for i in range(0, len(row)):  #  note assumption that row[] generated by loop over data_columns
            format_vals[self.data_columns[i]] = row[i]
        activity_id = self.activity_id_pattern.format(**format_vals)
        if self.group_set_col is not None:
            row = [agent_id, group_id, group_set, activity_id] + row
        else:
            row = [agent_id, group_id, activity_id] + row

        return row, timestamp


class Dataset:
    """Single dataset spec and generators which wraps around Models"""
    def __init__(self, statement_family, dataset_code,  output_dir, config_dir):

        self.output_dir = output_dir
        self.config_dir = config_dir

        self.statement_family = statement_family
        self.dataset_code = dataset_code

        def_file = path.join(self.config_dir, DATASET_DEF_DIR, "{}_{}.json".format(statement_family, dataset_code))
        if not path.exists(def_file):
            raise Exception("Failed to find file {}".format(def_file))
        with open(def_file, 'r') as f:
            ds_def = json.load(f)

        self.profile_id = ds_def["profile_id"]
        self.platform = ds_def["platform"]
        self.n_records = ds_def["n_records"]
        self.prune_future = ds_def["prune_future"]

        # these are defined in the membership_definition. The AY drives the memberships used and the timestamps
        #   while we read the stereotype keys to check that they match the rules which map to model codes.
        self.membership_def = ds_def["membership_def"]
        m_def_file = path.join(config_dir, MEMBERSHIP_DEF_DIR, "{}.json".format(self.membership_def))
        if not path.exists(m_def_file):
            raise Exception("Failed to find membership definition file {}".format(m_def_file))
        with open(m_def_file, 'r') as f:
            m_def = json.load(f)
        self.academic_years = m_def["academic_years"]
        self.group_stereptype_keys = m_def["group_stereotype_proportions"].keys()
        self.agent_stereptype_keys = m_def["agent_stereotype_proportions"].keys()

        # validate the stereotype to model code mapping
        # first index is the group stereotype and the second is the agent stereotype
        self.model_map = ds_def["stereotype_models"]
        if set(self.model_map.keys()) != set(self.group_stereptype_keys):
            raise Exception("Did not find the required group stereotype keys in the dataset definition. Found {} but expected {} from the membership definition {}".format(
                self.model_map.keys(), self.group_stereptype_keys, self.membership_def))
        for gs in self.model_map:
            if set(self.model_map[gs].keys()) != set(self.agent_stereptype_keys):
                raise Exception("Did not find the required agent stereotype keys in the stereotype model for group type {}. Found {} but expected {} from the membership definition {}".format(
                        gs, self.model_map[gs].keys(), self.agent_stereptype_keys, self.membership_def))

        # there should be some saved memberships to used when generating
        self.membership_dfs = dict()
        for ay in self.academic_years:
            m_file = _make_membership_file_path(self.output_dir, self.membership_def, ay)
            if not path.exists(m_file):
                raise Exception("Failed to find membership data for dataset generation. Looked for: {}. You must generate memberships before datesets.".format(m_file))
            self.membership_dfs[ay] = pd.read_csv(m_file, dtype=str)
            print "Read memberships from {} OK.".format(m_file)
            # todo - add lookup of model code from stereotypes, adapting
            # gsm = self.gs_models[group_type]
            # all_model_codes += list(choice(gsm.keys(), size=group_load, p=gsm.values()))

        # for generated data
        self.ay_df = dict()

        # for test
        print "\nStereotype-Model Map:"
        print pd.DataFrame(self.model_map)
        print

    def _pick_agent_group(self, ay):
        membership_df = self.membership_dfs[ay]
        return tuple(membership_df.iloc[randint(len(membership_df))].sort_index())

    def apply_models(self):
        """Does the generation based on the models"""
        m = Models(self.statement_family, self.profile_id, self.config_dir)
        #generated = list()
        for ay in self.academic_years:
            generated_ay = list()
            while len(generated_ay) < self.n_records:
                # get ids and stereotypes,
                aid, agent_stereotype, gid, group_set, group_stereotype = self._pick_agent_group(ay)
                # look up a model code for the stereotypes
                model_code = self.model_map[group_stereotype][agent_stereotype]
                row_list = m.try_generate(model_code, academic_year=ay, agent_id=aid, group_id=gid, group_set=group_set)
                if row_list is not None:
                    generated_ay += row_list
            #generated += generated_ay

            df = pd.DataFrame(data=generated_ay, columns=m.all_columns)
            # pruning
            if self.prune_future:
                old_len = len(df)
                df = df[df.timestamp < datetime.datetime.now().isoformat()]
                if len(df) < old_len:
                    print "Pruned {}/{} records for AY={} which would have post-dated now".format(old_len - len(df), len(generated_ay), ay)

            self.ay_df[ay] = df

    def to_csv(self, zipped=True):
        if len(self.ay_df) == 0:
            raise Exception("Use apply_models() before to_csv().")
        for ay in self.academic_years:
            out_file = _make_dataset_file_path(self.output_dir, self.platform, self.statement_family, self.dataset_code, ay)
            self.ay_df[ay].to_csv(out_file, index=False, date_format='%Y-%m-%dT%H:%M:%S')
            if zipped:
                csv_file = out_file
                out_file = csv_file.replace('csv', 'zip')
                with ZipFile(out_file, 'w') as z:
                    z.write(csv_file)
            print "Stored {} records to: {}".format(len(self.ay_df[ay]), out_file)

        print '-' * 80
        print


class Memberships:
    """Deal with generation of person-group records or preparation of UDD source (probably from FakerMaker) TODO"""
    def __init__(self, membership_def, output_dir, config_dir):

        self.membership_def = membership_def
        self.output_dir = output_dir

        def_file = path.join(config_dir, MEMBERSHIP_DEF_DIR, "{}.json".format(membership_def))
        if not path.exists(def_file):
            raise Exception("Failed to find file {}".format(def_file))
        with open(def_file, 'r') as f:
            m_def = json.load(f)

        self.n_agents = m_def["n_agents"]
        self.n_groups = m_def["n_groups"]
        self.group_load = (m_def["group_load"]["min"], m_def["group_load"]["max"])
        self.academic_years = m_def["academic_years"]
        # steretype proportions
        def normalise_props(old):
            weight_sum = sum(old.values())
            return {stereotype_code: float(old[stereotype_code]) / weight_sum for stereotype_code
                                             in old}
        self.group_stereotype_proportions = normalise_props(m_def["group_stereotype_proportions"])
        self.agent_stereotype_proportions = normalise_props(m_def["agent_stereotype_proportions"])

        # for test
        # print self.group_stereotype_proportions
        # print self.agent_stereotype_proportions

    def generate_ids(self):
        # generate cache of agent ids and stereotype
        agent_ids = ["S{:05}".format(n) for n in range(0, self.n_agents)]
        agent_stereotypes = choice(self.agent_stereotype_proportions.keys(), size=self.n_agents,
                                   p=self.agent_stereotype_proportions.values())

        for ay in self.academic_years:
            # use the proportions of the group stereotypes, compensating for rounding
            group_ids = dict()
            keys = self.group_stereotype_proportions.keys()
            proportions = self.group_stereotype_proportions.values()
            for i in range(0, len(keys)):
                if i < len(keys) - 1:
                    n_gs = int(round(self.n_groups * proportions[i]))
                else:
                    n_gs = self.n_groups - len(group_ids)
                group_ids[keys[i]] = ["G{}{:03}{}".format(keys[i], n, ay) for n in range(0, n_gs)]

            # create agent-group pairs, (according to group load) and randomise a model code for each
            all_agent_ids = list()
            all_group_ids = list()
            all_agent_stereotypes = list()
            all_group_stereotypes = list()
            all_group_sets = list()
            for i in range(0, self.n_agents):
                agent_id = agent_ids[i]
                group_type = choice(keys, p=proportions)
                group_load = self.group_load[0] if self.group_load[0] ==  self.group_load[1] else randint(self.group_load[0], self.group_load[1])
                group_load = min(group_load, len(group_ids[group_type]))
                all_agent_ids += [agent_id] * group_load
                all_agent_stereotypes += [agent_stereotypes[i]] * group_load
                all_group_ids += list(choice(group_ids[group_type], size=group_load, replace=False))  # no replacement otherwise we get duplicates
                all_group_stereotypes += [group_type] * group_load
                all_group_sets += ["{}/{}".format(group_type, ay)] * group_load

            # eliminate duplicate agent/group id records and store to instance var
            membership_df = pd.DataFrame({"agent_id": all_agent_ids,
                                          "group_id": all_group_ids,
                                          "agent_stereotype": all_agent_stereotypes,
                                          "group_stereotype": all_group_stereotypes,
                                          "group_set": all_group_sets  # can be used like a course instance code
                                          })
            membership_df.drop_duplicates(subset=['agent_id', 'group_id'], inplace=True)

            # save
            m_file = _make_membership_file_path(self.output_dir, self.membership_def, ay)
            membership_df.to_csv(m_file, index=False)
            print "Saved {} memberships for {} to {}".format(len(membership_df), ay, m_file)

        print '=' * 80
        print
