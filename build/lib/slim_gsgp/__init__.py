import warnings

# Force DeprecationWarning to be shown, even if it's silenced by default
warnings.simplefilter('always', DeprecationWarning)

warnings.warn(
    "The `gsgp_slim` package is deprecated. Please use the `slim_gsgp` package instead: https://pypi.org/project/slim_gsgp/",
    DeprecationWarning,
    stacklevel=2  # Ensures it points to the user's code
)