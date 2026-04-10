# Timeline Review — project-level manual evaluation
#
# Run interactively in R. Edit the two config lines below, then source.
# The active dataset is printed to the console on load.
#
# Console navigation:
#   show()                        re-display current project
#   nxt() / prv()                 next / previous project
#   goto(250)                     jump to index 250
#   goto_id("abc")                jump to a specific project_id
#   rnd()                         random project
#   use_stage("post_run", "ea")   switch stage + process (drops old data from memory)
#   use_stage("regex", "ce")      review regex candidate cache
#   use("ea_llm")                 switch by shortcut directly
#   datasets()                    list all available shortcuts
# ─────────────────────────────────────────────────────────────────────────────

rm(list = ls())

# ── Config: edit these two lines ─────────────────────────────────────────────
#
#   dataset_step  │  regex      post_training   post_run   post_llm
#   ──────────────┼─────────────────────────────────────────────────
#   ce            │  regex_ce   test_ce         ce         (none)
#   ea            │  regex_ea   test_ea         ea         ea_llm
#   eis           │  regex_eis  test_eis        eis        eis_llm
#
#dataset_step <- "post_training"   # regex | post_training | post_run | post_llm
dataset_step <- "psot"   # regex | post_training | post_run | post_llm
process      <- "ce"              # ce | ea | eis
# ─────────────────────────────────────────────────────────────────────────────

library(here)
source(here::here("code", "utils", "timeline_review.R"))
options(width = 10000)

load_timeline_navigator(.resolve_stage(dataset_step, process))

show()
