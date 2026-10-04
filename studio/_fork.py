# SPDX-License-Identifier: AGPL-3.0-only
# Fork marker. The presence of this module marks the installed package as the beaglemoo fork
# of Unsloth: every update path reads it to refuse upstream (PyPI, unslothai) sources.
#
# Stdlib only and constants only, so any caller (CLI, backend, scripts) can import it cheaply.
# Rebasing onto upstream keeps this file (it does not exist there). Never delete it.
"""beaglemoo/unsloth fork marker."""

FORK_REPO_URL = "https://github.com/beaglemoo/unsloth"
FORK_BRANCH = "feat/omlx-ds4-engines"

# Where the fork is updated from, relative to a checkout of FORK_REPO_URL, and the path the
# refusal message prints.
UPDATE_SCRIPT_REL = "studio/scripts/update-fork.sh"
UPDATE_SCRIPT_HINT = "~/Homelab/unsloth/unsloth/studio/scripts/update-fork.sh"
# The pinned third-party sources the fork installs (see studio/fork-pins.toml).
PINS_FILE_REL = "studio/fork-pins.toml"
