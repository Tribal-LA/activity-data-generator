# How To... with the Activity Data Generator
__Very much an initial start at a document...__

This outlines various use cases and the configuration settings to achieve each. The use cases are all of the form "I want the activity to look like X for a subset of agents/groups/contexts", but this isn't repeated.

## Simulate missing data
### Whole columns
Edit: model definition  
Section: top level of the definition, under "columns"
Setting: include the column in this list but only there

### Randomly for certain groups/students
First create two model definitions (say "A" and "B") which are the same except that "B" has missing data for whole columns (above) relative to "A". Maybe also create a new agent or group stereotype in the memberships definition file which will be used.

Edit: dataset definition  
Section: top level of the definition, under "stereotype_models"
Setting: assign model B to some agent/group stereotype cells.

### Randomly distributed
Edit: model definition  
Section: under the column-level randomisation control
Setting: use "p_na". The not-NA distribution will be drawn as specified.

## Different levels of activity in some contexts
Edit: model definition  
Section: under a model code  
Setting: use "p", setting at least one model to 1.0 and others with lower values to suppress activity.

## Non-independence between columns #1
Edit: model definition  
Section: under a model code  
Setting: use "option_group_prop" and "option_group_choice"

Examples:
- optional events get different distribution of attendance levels
- columns are related by an if-then logic
- later submissions tend to get lower marks