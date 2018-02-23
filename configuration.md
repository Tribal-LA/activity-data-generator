## Activity Data Generator JSON config files
This is driven by a series of JSON files. These are described from the bottom up.  
__NB:__ any files in the config/ directory tree should not be considered to be definitive; they are only examples (at some point, we might create a separate repository for standard model definitions).

## Membership Definition
- "n\_agents", the number of unique agents (students) to generate for (NB there is no guarantee that all agents will have recorded activitity)
- "n\_groups", the number of unique group ids (typically a module-instance) to generate for (NB there is no guarantee that there will be activity records for all groups). This must be large enough that n\_groups * the smallest group stereotype proportion exceeds the minimum group load (see below), otherwise the minimum load will not be obeyed.
- "group\_load", the number of groups a student should have. Make sure this is sensible in relation to n\_agents and n\_records. Generally expect that there will be several to very-many records per student-group in realistic data. "group\_load" is uniform over min/max (inclusive):
 - "min"
 - "max"
- "group\_stereptype\_proportions", keys are the stereotype group type codes and values are the proportion of "n\_groups" in each type. Proportion is renormalised.
- "agent\_stereotype\_proportions", as for "group\_stereotype\_proportions" but related to agents. In some scenarios, these would be correlated with outcomes.
- "academic\_years", containing a list of integers for the academic years to generate data for

Example:
```
{
	"n_agents": 5,
	"n_groups": 5,
	"group_load": {"min": 2, "max": 6},
	"academic_years": [2016, 2017],
	"group_stereotype_proportions": {
		"a": 0.3,
		"b": 0.7
	},
	"agent_stereotype_proportions": {
		"risky": 0.2,
		"steady": 0.5,
		"safe": 0.3
	}
}
```

## Model Definition
The configuration files are in the form: model\_defintions/xxxxx\_ppppp.json (where xxxxx is the name of a statement family in the same form as we'd use in pipeline tenant config and ppppp is the profile\_id). The intention is that this file contains ALL of the generator models which would apply for a statement family for a given xAPI profile (as defined in xapi/xapi\_profile\_rules.py of the data pipeline). When generating other than for the data pipeline, profile\_id may be freely chosen.

This defines one or more stereotype behavioural models via an associative array. The top level keys used:
- "models\_type" - may be 'event' for a single event model, 'random-session' for a series of events with random distribution determined on a per-model basis (see below)
- "columns" - the names of columns to be generated (excluding the id columns with special treatment - see below). Columns not listed in "column\_values" (see below) will be all-NA. A column called timestamp is always added by the generator. By convention, start columns which will be consumed by the application-specific py code with "\__". Include columns which will not appear in "column\_values" but which will be created in the application-specific business rules (e.g. in attendance.py)
- "agent\_id\_col" - column name for the agent. This may link to UDD (__TODO__)
- "group\_id\_col" - column name for the group, generally a module-instance level thing. This may link to UDD (__TODO__)
- "group\_set\_col" - column name for the group stereotype. Optional. May be usable to simulate a higher-level group (e.g. a course instance when the group is a module, or a faculty when the group is a course)
- "activity\_id\_col" - the id for the object of the statement. This will be generated using "activity\_id\_pattern".
- "activity\_id\_pattern" - a string formatter containing {key} named replacements, where key is a data column name (including "timestamp"), the group id column name, or the reseved key "\__n" for a random number between 0 and "n\_activities", or the reserved key "\__tod" for non-sessional models with a "tod\_pattern" in operation, when the value substituted will be one of the keys in "tod". e.g. "A{MOD\_INSTANCE\_ID}{\_\_tod}\_{\__n:03}"
- "n\_activities" - see "activity\_id\_pattern", this is generally a per-group value
- "temporal" - contains a list of 1 or more members, which apply to all models, where each member is an associative array defined by these keys:
 - "note" - an optional string to describe the member of "temporal"
 - "weeks" - a list of relative (academic week numbers) weeks
 - "intensity" - a numeric for the relative level of intensity of this member of "temporal"; activity will be equally spread among the weeks specified
 - "day\_pattern" - a list of length 7 for the relative intensity on each day of the week (first element is monday)
- "models" - contains an associative array as described below.

Intensity is renormalised:
- "day\_pattern" will be re-normalised across each list of 7 days, so a pattern [1,1,1,1,1,0,0] would be acceptable
- "intensity" will be renormalised across all entries in "temporal"

The content of "models" is:
- the top level key is the model code (identifier) within which there should be further associative arrays. For a simple model, these are:
- "p" - relative probability of activitity being recorded. For some statement families this will be 1.0 across all models (e.g. attendance, since a statement is created for non-attendance), whereas for others, only one model will have a value of 1.0 and others lower according to the relative chance that activity will occur in a given time interval.
- "column\_values" - contains a dict with an entry for each column which is not all-NA, where they key name is the name of the column to be created.
- "option\_group\_prop" - relative probability of the option-group to be used, specified using a dict with keys to match the keys in "option\_group\_choice" and values = the relative probability (which will be renormalised). "option\_group\_prop" may be omitted, in which case "option\_group\_choice" will be ignored
- "option\_group\_choice" - contains option-groups (option-groups are used to achieve conditional dependency between a set of columns) of column specifications, as a dictionary (keys may be chosen arbitrarily) with values being further dictionaries defined in the same way as the contents of "column\_values". These do not have to have the same keys as the master "column\_values", but if present in a chosen group then they will override anything specified in the "column\_values" which are outside the "option\_group\_choice" section. If a column appeas in all option groups then it is not required in "column\_values"
- "tod\_pattern" - settings to control time of day for the timestamp. May be missing to cause generator to provide date-only timestamps, or a dictionary with
 - "tod\_pattern\_type" - with value being one of null, 'choice' or one of 'norm' or 'uniform' if a distribution-based timestamp should be generated. null indicates date-only timestamps will be created (and other settings ignored).
 - "tod" - a dictionary with the key being an hour:minute specifier strings and the value being a relative probability. this is used in the same way as a categorical column specification if 'choice' is given as the pattern type, and a timestamp created using one of the specified hour:minute keys. If the pattern type is 'distribution' then a random offset will be applied once the initial choice has been made.
 - "distributions" - (only used if the tod pattern type indicates) a dict with the same keys as "tod" which specifies the distribution of a random (per-record) offset to be added to the hour:minute it is aligned with. Units are minutes. The value is a standard deviation for a normal distribution or the half-width for a uniform distribution.
- "session\_model" - only applicable when the "models\_type" is 'random-session', a dict defined by:
  - "interval" - the distribution of time intervals between events in the session (in seconds), currently a dict of three keys e.g. {"dist": "norm", "mu": 60.0, "sigma": 15.0} (currently "dist" is ignored and a normal distribution is always used).
  - "min\_length" - min number of events in a session
  - "survival" - the proportion of cases where another event is generated in the session (in the other case, the session ends), applied after each event in the session is generated, once min\_length is reached
  - "max\_length" - max number of events in a session

_Currently, the generator cannot simulate a session which has activity according to a variety different model definitions. In this case, the columns would differ._

The content of each column\_values specification must be one of the following:
- a string value - the column will be created with all members containing the value. Use string values in the JSON, even if the actual value is a numerical one.
- a dictionary - the column will contain randomly generated content according to the dictionary

The content of the randomisation-control dictionay must "p\_na" and one of "categorical" or "numerical":
- "p\_na" - the probability of a missing value. This applies for both numerical or categorical cases. For categorical, an alternative is to specify null as one of the categories. May be omitted.
- for a categorical, under the key "categorical", a dict to indicate the relative proportion of each category:
 - "code": fraction, where the fraction indicates the relative proportion of records which should be generated to contain code. Codes will always be generated as strings. Missing values should be coded ""
- for numerical, under the key "numerical":
 - "distribution", ("normal", or "uniform" only)
 - "params" - params according to the distibution = [mu, sd] for "norm", or [min, max] for uniform

Compound, as opposed to Simple models may be specified. When a compound model is used, each time a record is produced, it is actually produced by one of the simple models, chosen based on the specified simple model probabilities. The specification for a compound model has as its key a model code, which may be used from a Dataset Definition, and rather than the entries outlined above contains a dictionary:
- "compound\_model" containing a dict of simple model codes as keys and relative probabilities as values (which are renormalised)

Examples:
__Session Model__  
```
"session_model": {
	"interval": {"dist": "norm", "mu": 60.0, "sigma": 15.0},
	"min_length": 3,
	"survival": 0.75,
	"max_length": 30
			}
```

## Dataset Definition
dataset\_definitions/xxxxx\_yyyyy.json (where xxxxx is as above and yyyyy is some code to distinguish between different datasets).

A dataset is defined as arising from a statement of the required number of records and the fraction of each of one or more stereotypes of group (typically it is a module instance). Each group stereotype is defined by the fraction of each of one or more models applicable to the statement family xxxxx. It is not required to use all models.

The dataset definition JSON comprises an associative array containing the following required keys:
- "profile\_id" - select the statement family profile version - use e.g. "jisc\_v1.0.1". This does not (currently) have any direct link into the xAPI Profile Rules file, but it is used to form the name of the model configuration file which is consulted.
- "platform" - name for the platform as used in xAPI pipeline sections (e.g. "UxAPI", "Moodle", etc). This is only used for the file name, which is used in the data pipeline, hence the value may be freely chosen in other circumstances.
- "n\_records", containing an integer number of records required (for each academic year). This may be exceeded if a session-based model is used
- "membership\_def" - gives the name of a membership definition (used as the JSON file name in membership\_definitions/ and as the name of output CSV generated by Membership class)
- "prune\_future" - whether to remove events timestamped in the future (if any, this will lead to less than n\_records)
- "stereotype\_models", containing a dict for each group stereotype which maps agent stereotypes to a model code. The effect of this is equivalent to a table with row and column headings (the stereotype keys) and single entries in each cell which indicates a model to use. __The keys must exactly match the keys provided in the membership\_definitions indicated by "membership\_def"__. These are single models; use a compound model to get bimodel (etc) student behaviour.


The code will re-normalise the fractions, so these do not have to sum to 1.0, and they may be whole numbers.

Example:
```

```

## Wrapper Config
- "membership\_def" - required, name of the membership definition JSON
- "generate\_datasets" - a list of dictionaries:
	- "custom\_generator" - optional, the module.class name of a custom class derived from Dataset class. e.g. "attendance.AttendanceDataset". If missing, the generic generator.Dataset class will be used.
	- "statement\_family" - as defined elsewhere, e.g. "ATTENDANCE",
	- "dataset\_defs" - a list of dataset defintion codes
