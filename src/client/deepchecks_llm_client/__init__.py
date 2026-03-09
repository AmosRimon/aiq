import dunamai as _dunamai

__version__ = _dunamai.get_version("deepchecks_llm_client",
                                   third_choice=_dunamai.Version.from_any_vcs).serialize(style=_dunamai.Style.Pep440)
