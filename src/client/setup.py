import setuptools
from dunamai import Pattern, Version

setuptools.setup(
    name="deepchecks-llm-client",
    version=Version.from_any_vcs(pattern=Pattern.DefaultUnprefixed).serialize(),
    include_package_data=True,
)
