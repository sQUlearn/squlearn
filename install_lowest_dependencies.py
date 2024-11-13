"""This script installs the package with the lowest dependencies."""

import re
import subprocess
import sys
from typing import List, Optional, Tuple

import toml

# Load the pyproject.toml
pyproject = toml.load("pyproject.toml")

# Extract dependencies
dependencies: List[str] = pyproject.get("project", {}).get("dependencies", []) + pyproject.get(
    "project", {}
).get("optional-dependencies", {}).get("examples", [])


# Function to get exactly the minimal specified version
def get_lowest_version(dependency_string: str) -> str:
    """Get the lowest version of a dependency."""
    pattern = re.compile(r"([\w-]+)(?:>=(\d*(?:\.\d*(?:\.\d*)?)?))?")
    match = pattern.match(dependency_string)
    if match:
        groups: Tuple[Optional[str], Optional[str]] = match.groups()
        if groups[1]:
            return "==".join(groups)
        return groups[0]
    return dependency_string


# Install the main package without dependencies
subprocess.run([sys.executable, "-m", "pip", "install", ".", "--no-deps"], check=True)

# Get the lowest version of pennylane
PENNYLANE_VERSION = None
for dependency in dependencies:
    if dependency.startswith("pennylane"):
        PENNYLANE_VERSION = get_lowest_version(dependency).split("==")[1]
        break

dependencies = [get_lowest_version(dependency) for dependency in dependencies]
if PENNYLANE_VERSION:
    dependencies.append(f"pennylane-lightning=={PENNYLANE_VERSION}")
subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies, check=True)
